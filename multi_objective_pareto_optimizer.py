# ═══════════════════════════════════════════════════════════════════════════════
# multi_objective_pareto_optimizer.py — Feature 12: MOPO
# Simultaneously optimises Cost / Time / Quality / Risk using Pareto analysis.
# No single-score winner — delivers the full Pareto frontier.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

MOPO_OBJECTIVES       = ["cost", "time", "quality", "risk"]
MOPO_MAX_ITERATIONS   = 50     # NSGA-II style iterations
MOPO_POPULATION_SIZE  = 20
MOPO_CROSSOVER_RATE   = 0.70
MOPO_MUTATION_RATE    = 0.15
MOPO_DEFAULT_WEIGHTS  = {"cost": 0.25, "time": 0.25, "quality": 0.30, "risk": 0.20}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ObjectiveVector:
    plan_id:      str
    plan_text:    str
    agent:        str
    cost_score:   float   # 0–100, higher = cheaper (inverted cost)
    time_score:   float   # 0–100, higher = faster
    quality_score:float   # 0–100, higher = better quality
    risk_score:   float   # 0–100, higher = lower risk
    rank:         int     = 0    # Pareto rank (1 = Pareto-optimal)
    crowding_dist:float   = 0.0  # NSGA-II crowding distance

    @property
    def objectives(self) -> Dict[str, float]:
        return {
            "cost":    self.cost_score,
            "time":    self.time_score,
            "quality": self.quality_score,
            "risk":    self.risk_score,
        }

    def dominates(self, other: "ObjectiveVector") -> bool:
        """self dominates other if it is >= on all and > on at least one."""
        objs_self  = self.objectives
        objs_other = other.objectives
        at_least_as_good = all(objs_self[o] >= objs_other[o] for o in MOPO_OBJECTIVES)
        strictly_better  = any(objs_self[o] >  objs_other[o] for o in MOPO_OBJECTIVES)
        return at_least_as_good and strictly_better

    def weighted_score(self, weights: Dict[str, float] = None) -> float:
        w = weights or MOPO_DEFAULT_WEIGHTS
        return sum(self.objectives[obj] * w.get(obj, 0.25) for obj in MOPO_OBJECTIVES)


@dataclass
class ParetoFrontier:
    frontier:        List[ObjectiveVector]   # Pareto-rank-1 solutions
    all_solutions:   List[ObjectiveVector]   # full population
    dominated_count: int
    knee_point:      Optional[ObjectiveVector]  # closest to utopia point
    ideal_point:     Dict[str, float]
    nadir_point:     Dict[str, float]


@dataclass
class MOPOResult:
    frontier:          ParetoFrontier
    recommended_plan:  ObjectiveVector
    preference_map:    Dict[str, ObjectiveVector]  # "fastest"|"cheapest"|"safest" → plan
    iterations_run:    int
    convergence_delta: float


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_MOPO_OBJECTIVE_SCORER = """You are a MULTI-OBJECTIVE PLAN EVALUATOR.

AIM: {aim}
PLAN (agent={agent}):
{plan_text}

Score this plan on 4 independent objectives (0–100 each):

  COST     — 100 = negligible cost, 0 = extremely expensive
  TIME     — 100 = very fast execution, 0 = extremely slow
  QUALITY  — 100 = comprehensive, complete, innovative, 0 = superficial
  RISK     — 100 = minimal risk, well-mitigated, 0 = extremely risky

Be calibrated: most plans should score 40–75. Reserve 85+ for exceptional plans.

Respond ONLY with valid JSON (no markdown):
{{"cost_score":0.0,"time_score":0.0,"quality_score":0.0,"risk_score":0.0,
  "cost_rationale":"...","time_rationale":"...","quality_rationale":"...","risk_rationale":"..."}}"""

PROMPT_MOPO_PREFERENCE_ELICITOR = """You are a PREFERENCE ANALYST for multi-objective optimisation.

AIM: {aim}
PARETO FRONTIER ({n_solutions} solutions):
{frontier_json}

A decision-maker must choose ONE plan. Help identify:
  1. The "balanced" plan closest to the utopia point (all objectives maximised)
  2. The "fastest" plan (maximise TIME score)
  3. The "cheapest" plan (maximise COST score)
  4. The "safest" plan (maximise RISK score)
  5. The "highest quality" plan (maximise QUALITY score)

Respond ONLY with valid JSON:
{{"balanced_plan_id":"...","fastest_plan_id":"...","cheapest_plan_id":"...","safest_plan_id":"...","highest_quality_plan_id":"...","preference_rationale":"..."}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class MultiObjectiveParetoOptimizer:
    """
    NSGA-II inspired multi-objective optimizer.

    Instead of collapsing all objectives into a single score,
    maintains a Pareto frontier of non-dominated solutions.
    Lets the decision-maker choose based on their preference.
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        agent:    str = "gemini",
        weights:  Optional[Dict[str, float]] = None,
    ):
        self.call_fn = call_fn
        self.agent   = agent
        self.weights = weights or MOPO_DEFAULT_WEIGHTS
        self._all_solutions: List[ObjectiveVector] = []

    # ── Scoring ───────────────────────────────────────────────────────────────

    async def score_plan(
        self,
        plan_id:   str,
        plan_text: str,
        agent:     str,
        aim:       str,
    ) -> ObjectiveVector:
        """LLM scores one plan on 4 objectives independently."""
        prompt = PROMPT_MOPO_OBJECTIVE_SCORER.format(
            aim       = aim,
            agent     = agent,
            plan_text = plan_text[:1500],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return ObjectiveVector(
                plan_id       = plan_id,
                plan_text     = plan_text,
                agent         = agent,
                cost_score    = float(data.get("cost_score",    50.0)),
                time_score    = float(data.get("time_score",    50.0)),
                quality_score = float(data.get("quality_score", 50.0)),
                risk_score    = float(data.get("risk_score",    50.0)),
            )
        except Exception as e:
            logger.warning(f"[MOPO] Scoring failed for {plan_id}: {e}")
            return ObjectiveVector(
                plan_id=plan_id, plan_text=plan_text, agent=agent,
                cost_score=50.0, time_score=50.0,
                quality_score=50.0, risk_score=50.0,
            )

    async def score_all_plans(
        self,
        plans: Dict[str, str],
        aim:   str,
    ) -> List[ObjectiveVector]:
        """Score all agent plans in parallel."""
        tasks = [
            self.score_plan(f"{agent}_p0", text, agent, aim)
            for agent, text in plans.items()
        ]
        results = await asyncio.gather(*tasks)
        self._all_solutions = list(results)
        return list(results)

    # ── Pareto Ranking ────────────────────────────────────────────────────────

    def compute_pareto_ranks(
        self,
        solutions: Optional[List[ObjectiveVector]] = None,
    ) -> List[ObjectiveVector]:
        """
        NSGA-II fast non-dominated sort.
        Assigns rank 1 = Pareto-optimal, rank 2 = dominated only by rank-1, etc.
        """
        sols = solutions or self._all_solutions
        n    = len(sols)
        dominated_by: List[Set] = [set() for _ in range(n)]
        dominates:    List[List[int]] = [[] for _ in range(n)]
        rank:         List[int] = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sols[i].dominates(sols[j]):
                    dominates[i].append(j)
                elif sols[j].dominates(sols[i]):
                    dominated_by[i].add(j)

        # Rank 1: not dominated by anyone
        current_front = [i for i in range(n) if len(dominated_by[i]) == 0]
        for i in current_front:
            rank[i] = 1

        r = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominates[i]:
                    dominated_by[j].discard(i)
                    if len(dominated_by[j]) == 0:
                        rank[j] = r + 1
                        next_front.append(j)
            r += 1
            current_front = next_front

        for i, sol in enumerate(sols):
            sol.rank = rank[i]

        # Compute crowding distance within each rank
        self._compute_crowding_distances(sols)
        return sols

    def _compute_crowding_distances(self, sols: List[ObjectiveVector]) -> None:
        """NSGA-II crowding distance for diversity preservation."""
        by_rank: Dict[int, List[int]] = {}
        for i, sol in enumerate(sols):
            by_rank.setdefault(sol.rank, []).append(i)

        for rank_group in by_rank.values():
            for obj in MOPO_OBJECTIVES:
                sorted_idx = sorted(rank_group, key=lambda i: getattr(sols[i], f"{obj}_score"))
                sols[sorted_idx[0]].crowding_dist  = float('inf')
                sols[sorted_idx[-1]].crowding_dist = float('inf')
                obj_range = (
                    getattr(sols[sorted_idx[-1]], f"{obj}_score") -
                    getattr(sols[sorted_idx[0]],  f"{obj}_score")
                ) or 1.0
                for k in range(1, len(sorted_idx) - 1):
                    dist = (
                        getattr(sols[sorted_idx[k+1]], f"{obj}_score") -
                        getattr(sols[sorted_idx[k-1]], f"{obj}_score")
                    ) / obj_range
                    sols[sorted_idx[k]].crowding_dist += dist

    # ── Frontier Construction ─────────────────────────────────────────────────

    def build_pareto_frontier(
        self,
        solutions: Optional[List[ObjectiveVector]] = None,
    ) -> ParetoFrontier:
        """Extract rank-1 solutions and compute ideal/nadir points."""
        sols     = solutions or self._all_solutions
        ranked   = self.compute_pareto_ranks(sols)
        frontier = [s for s in ranked if s.rank == 1]

        ideal = {obj: max(getattr(s, f"{obj}_score") for s in sols)
                 for obj in MOPO_OBJECTIVES}
        nadir = {obj: min(getattr(s, f"{obj}_score") for s in sols)
                 for obj in MOPO_OBJECTIVES}

        # Knee point: minimum distance to utopia (ideal)
        def utopia_distance(s: ObjectiveVector) -> float:
            return math.sqrt(sum(
                (ideal[obj] - getattr(s, f"{obj}_score")) ** 2
                for obj in MOPO_OBJECTIVES
            ))

        knee = min(frontier, key=utopia_distance) if frontier else None
        dominated = len([s for s in ranked if s.rank > 1])

        return ParetoFrontier(
            frontier        = frontier,
            all_solutions   = ranked,
            dominated_count = dominated,
            knee_point      = knee,
            ideal_point     = ideal,
            nadir_point     = nadir,
        )

    # ── Evolutionary Improvement ──────────────────────────────────────────────

    async def evolve_frontier(
        self,
        plans:         Dict[str, str],
        aim:           str,
        n_iterations:  int = MOPO_MAX_ITERATIONS,
    ) -> MOPOResult:
        """
        NSGA-II style evolution:
        1. Score initial population
        2. Select parents from Pareto front
        3. Generate offspring via crossover (plan text blending)
        4. Re-score and merge populations
        5. Truncate to population size by rank + crowding distance
        """
        # Initial scoring
        population = await self.score_all_plans(plans, aim)
        prev_ideal_sum = 0.0
        conv_delta     = 0.0

        for iteration in range(min(n_iterations, 5)):  # cap at 5 LLM iterations
            ranked   = self.compute_pareto_ranks(population)
            frontier = [s for s in ranked if s.rank == 1]

            # Generate offspring via crossover of Pareto-optimal pairs
            offspring = await self._generate_offspring(frontier, aim, n=2)
            population = ranked + offspring

            # Truncate: prefer rank 1, then crowding distance
            population.sort(key=lambda s: (s.rank, -s.crowding_dist))
            population = population[:MOPO_POPULATION_SIZE]

            # Convergence check
            new_ideal_sum = sum(
                max(getattr(s, f"{obj}_score") for s in population)
                for obj in MOPO_OBJECTIVES
            )
            conv_delta    = abs(new_ideal_sum - prev_ideal_sum)
            prev_ideal_sum= new_ideal_sum

            if conv_delta < 0.5 and iteration > 0:
                logger.info(f"[MOPO] Converged at iteration {iteration+1}")
                break

        final_frontier = self.build_pareto_frontier(population)
        preference_map = await self._elicit_preferences(final_frontier, aim)

        return MOPOResult(
            frontier          = final_frontier,
            recommended_plan  = final_frontier.knee_point or population[0],
            preference_map    = preference_map,
            iterations_run    = min(n_iterations, 5),
            convergence_delta = conv_delta,
        )

    async def _generate_offspring(
        self,
        frontier: List[ObjectiveVector],
        aim:      str,
        n:        int = 2,
    ) -> List[ObjectiveVector]:
        """Generate offspring by blending two Pareto-optimal plans."""
        if len(frontier) < 2:
            return []
        offspring = []
        for _ in range(n):
            p1, p2 = random.sample(frontier[:min(len(frontier), 6)], 2)
            # Blend: first half of p1 + second half of p2
            lines1 = [l for l in p1.plan_text.split('\n') if l.strip()]
            lines2 = [l for l in p2.plan_text.split('\n') if l.strip()]
            mid1   = max(1, len(lines1) // 2)
            mid2   = max(1, len(lines2) // 2)
            blended_lines = lines1[:mid1] + lines2[mid2:]
            blended_text  = "\n".join(blended_lines)
            ov = await self.score_plan(
                plan_id   = f"offspring_{len(offspring)}",
                plan_text = blended_text,
                agent     = f"blend_{p1.agent}_{p2.agent}",
                aim       = aim,
            )
            offspring.append(ov)
        return offspring

    async def _elicit_preferences(
        self,
        frontier: ParetoFrontier,
        aim:      str,
    ) -> Dict[str, ObjectiveVector]:
        """Build preference map: fastest / cheapest / safest / best quality."""
        if not frontier.frontier:
            return {}

        # Rule-based preference map (no LLM needed — deterministic)
        f = frontier.frontier
        result = {
            "balanced":   min(f, key=lambda s: sum(
                (frontier.ideal_point[obj] - getattr(s, f"{obj}_score"))**2
                for obj in MOPO_OBJECTIVES
            )),
            "fastest":   max(f, key=lambda s: s.time_score),
            "cheapest":  max(f, key=lambda s: s.cost_score),
            "safest":    max(f, key=lambda s: s.risk_score),
            "highest_quality": max(f, key=lambda s: s.quality_score),
        }
        return result

    def get_mopo_report(self, result: MOPOResult) -> Dict:
        """Structured report for Excel/frontend output."""
        def ov_dict(ov: ObjectiveVector) -> Dict:
            return {
                "plan_id":      ov.plan_id,
                "agent":        ov.agent,
                "cost_score":   round(ov.cost_score, 1),
                "time_score":   round(ov.time_score, 1),
                "quality_score":round(ov.quality_score, 1),
                "risk_score":   round(ov.risk_score, 1),
                "weighted_score":round(ov.weighted_score(self.weights), 1),
                "rank":         ov.rank,
                "crowding_dist":round(ov.crowding_dist, 3),
            }

        return {
            "frontier_size":     len(result.frontier.frontier),
            "dominated_count":   result.frontier.dominated_count,
            "ideal_point":       result.frontier.ideal_point,
            "nadir_point":       result.frontier.nadir_point,
            "knee_point":        ov_dict(result.frontier.knee_point) if result.frontier.knee_point else None,
            "frontier_solutions":[ov_dict(s) for s in result.frontier.frontier],
            "preferences": {
                k: ov_dict(v) for k, v in result.preference_map.items()
            },
            "iterations_run":    result.iterations_run,
            "convergence_delta": round(result.convergence_delta, 3),
        }


def _parse_json(raw: str) -> Dict:
    raw   = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
