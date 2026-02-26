# ═══════════════════════════════════════════════════════════════════════════════
# genetic_plan_mutator.py — Feature 5: Genetic Plan Mutator (GPM)
# Replaces step2_improvement_chain() with a full evolutionary optimizer
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import copy
import logging
import random
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

from system_config import (
    GPM_CROSSOVER_PROB, GPM_MUTATION_PROB_PER_STEP, GPM_ELITISM_K,
    GPM_MAX_GENERATIONS, GPM_PLATEAU_PATIENCE, GPM_IMMIGRANT_RATE,
    GPM_TOURNAMENT_SIZE, GPM_MIN_POPULATION
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PlanChromosome:
    """A planning document represented as a sequence of step-genes."""
    chromosome_id:   str
    parent_agent:    str
    steps:           List[str]       # genes
    fitness_score:   float           = 0.0
    generation:      int             = 0
    lineage:         List[str]       = field(default_factory=list)   # parent IDs
    mutation_count:  int             = 0
    crossover_count: int             = 0
    is_immigrant:    bool            = False

    def to_plan_string(self) -> str:
        """Serialise back to a numbered plan string."""
        return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(self.steps))

    def clone(self, new_id: Optional[str] = None) -> "PlanChromosome":
        c = copy.deepcopy(self)
        c.chromosome_id = new_id or f"chr_{uuid.uuid4().hex[:8]}"
        c.lineage = [self.chromosome_id] + self.lineage[:5]
        return c


@dataclass
class GenerationStats:
    """Metrics for one completed generation."""
    generation:         int
    population_size:    int
    max_fitness:        float
    mean_fitness:       float
    min_fitness:        float
    best_chromosome_id: str
    plateau_count:      int
    fitness_delta:      float           # vs previous generation
    operations:         Dict[str, int]  = field(default_factory=dict)


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_STEP_MUTATOR = """You are the Genetic Mutation Operator for an AI Planning System.
Rewrite ONE step while preserving logical context.

AIM: {aim}
ALL STEPS (for context only):
{all_steps}

TARGET STEP #{step_idx}: "{current_step}"
PRECEDING STEP: "{prev_step}"
FOLLOWING STEP:  "{next_step}"

REWRITE RULES:
- Make it MORE specific, actionable, and measurable
- Must logically follow from preceding step
- Must logically enable following step  
- Introduce a novel approach not present currently
- Keep same length (±30%)
- Use imperative form (action verb first)
- Do NOT change WHAT the step accomplishes

Respond ONLY with valid JSON (no markdown):
{{"mutated_step":"...", "improvement_rationale":"...", "mutation_type":"specificity|methodology|technology|sequencing"}}"""

PROMPT_CROSSOVER_COHERENCE = """You are the Crossover Coherence Validator for a Genetic Planning System.

AIM: {aim}
CROSSOVER POINT: step {crossover_point}

FIRST HALF (from Parent A, steps 1-{crossover_point}):
{first_half}

SECOND HALF (from Parent B, steps {crossover_point_plus1} onward):
{second_half}

TASK: Does this combined plan make logical sense end-to-end?

Respond ONLY with valid JSON:
{{"is_coherent":true,"seam_quality":"seamless|minor_gap|major_discontinuity","bridge_step_needed":false,"bridge_step_text":null,"estimated_fitness":75.0,"better_than_parents":true}}"""

PROMPT_IMMIGRANT = """You are a Diversity Injection Agent for an Evolutionary Planning System.
The evolutionary process is STUCK — generate a radically different plan.

Current best fitness: {best_fitness}
Plateau duration: {plateau_gens} generations

CURRENT BEST PLAN (DO NOT repeat its approach):
{current_best}

AIM: {aim}
INITIAL STEPS: {initial_steps}

Generate a COMPLETELY DIFFERENT plan that:
1. Uses a different structural framework or methodology
2. Challenges assumptions in the current best plan
3. Introduces unconventional sequencing or priorities
4. Still achieves the AIM

Output steps ONLY, one per line, starting with "- ":"""


# ── Engine ────────────────────────────────────────────────────────────────────

class GeneticPlanMutator:
    """
    Evolutionary optimizer that replaces the linear improvement chain.

    Key operations:
    • Tournament selection (k=3)
    • Single-point crossover with coherence validation
    • Per-step point mutation via LLM rewriter
    • Diversity injection (immigrant chromosomes)
    • Plateau detection with adaptive stopping
    • Elitism: top-K always survive

    Usage:
        gpm = GeneticPlanMutator(score_fn, mutate_fn, generate_fn, aim, steps)
        best, history = await gpm.evolve()
    """

    def __init__(
        self,
        score_fn:    Callable[[str, str], Awaitable[float]],       # (plan_text, aim) → score
        mutate_fn:   Callable[[str, str], Awaitable[Dict]],         # (prompt, agent) → response dict
        generate_fn: Callable[[str, List[str], str], Awaitable[str]], # (aim, steps, agent) → plan_text
        aim:         str,
        initial_steps: List[str],
    ):
        """
        Args:
            score_fn:    async function that scores a plan (0-100)
            mutate_fn:   async function that calls an LLM with a mutation prompt
            generate_fn: async function that generates a fresh plan (for immigrants)
            aim:         The planning aim string
            initial_steps: Original user-provided steps
        """
        self.score_fn      = score_fn
        self.mutate_fn     = mutate_fn
        self.generate_fn   = generate_fn
        self.aim           = aim
        self.initial_steps = initial_steps
        self.population:   List[PlanChromosome] = []
        self.history:      List[GenerationStats] = []
        self.best_ever:    Optional[PlanChromosome] = None
        self._plateau_count = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize_population(self, ai_plans: Dict[str, str]) -> List[PlanChromosome]:
        """Convert the step1 agent plans into an initial chromosome population."""
        self.population = []
        for agent_name, plan_text in ai_plans.items():
            steps = self._parse_steps_from_plan(plan_text)
            if len(steps) < 2:
                continue
            chrom = PlanChromosome(
                chromosome_id = f"chr_{agent_name}_{uuid.uuid4().hex[:6]}",
                parent_agent  = agent_name,
                steps         = steps,
                generation    = 0,
            )
            self.population.append(chrom)

        # Enforce minimum population
        if len(self.population) < GPM_MIN_POPULATION:
            logger.warning(
                f"[GPM] Population too small ({len(self.population)}). "
                f"Need at least {GPM_MIN_POPULATION}."
            )

        logger.info(f"[GPM] Initialized population: {len(self.population)} chromosomes")
        return self.population

    @staticmethod
    def _parse_steps_from_plan(plan_text: str) -> List[str]:
        """
        Extract individual steps from a plan string.
        Handles: "Step N: text", "N. text", "- text", bullet lines.
        """
        lines = plan_text.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip common prefixes
            cleaned = re.sub(r"^(step\s*\d+[:.\s]+|\d+[.)\s]+|[-•*]\s*)", "", line, flags=re.IGNORECASE).strip()
            if len(cleaned) > 15:   # ignore very short lines (headers etc)
                steps.append(cleaned)
        return steps

    # ── Fitness Evaluation ────────────────────────────────────────────────────

    async def evaluate_fitness_async(
        self,
        chromosomes: List[PlanChromosome],
    ) -> List[PlanChromosome]:
        """Score all chromosomes in parallel."""
        async def score_one(chrom: PlanChromosome) -> PlanChromosome:
            try:
                chrom.fitness_score = await self.score_fn(chrom.to_plan_string(), self.aim)
            except Exception as e:
                logger.warning(f"[GPM] Fitness eval failed for {chrom.chromosome_id}: {e}")
                chrom.fitness_score = 0.0
            return chrom

        scored = await asyncio.gather(*[score_one(c) for c in chromosomes])
        return list(scored)

    # ── Selection ─────────────────────────────────────────────────────────────

    def tournament_selection(
        self,
        population:  List[PlanChromosome],
        n_select:    int,
        k:           int = GPM_TOURNAMENT_SIZE,
    ) -> List[PlanChromosome]:
        """
        Run n_select tournaments of size k.
        Each tournament: pick k random individuals, return the fittest.
        """
        selected = []
        for _ in range(n_select):
            candidates = random.sample(population, min(k, len(population)))
            winner     = max(candidates, key=lambda c: c.fitness_score)
            selected.append(winner)
        return selected

    # ── Crossover ─────────────────────────────────────────────────────────────

    async def single_point_crossover(
        self,
        parent_a: PlanChromosome,
        parent_b: PlanChromosome,
        validate: bool = True,
    ) -> Tuple[PlanChromosome, PlanChromosome]:
        """
        Split both parents at a random crossover point and swap halves.
        Optionally validates coherence via PROMPT_CROSSOVER_COHERENCE.
        Returns two child chromosomes.
        """
        min_len = min(len(parent_a.steps), len(parent_b.steps))
        if min_len < 2:
            return parent_a.clone(), parent_b.clone()

        cp = random.randint(1, min_len - 1)

        child_a_steps = parent_a.steps[:cp] + parent_b.steps[cp:]
        child_b_steps = parent_b.steps[:cp] + parent_a.steps[cp:]

        child_a = parent_a.clone()
        child_a.steps          = child_a_steps
        child_a.crossover_count = parent_a.crossover_count + 1

        child_b = parent_b.clone()
        child_b.steps          = child_b_steps
        child_b.crossover_count = parent_b.crossover_count + 1

        # Coherence validation — inject bridge step if needed
        if validate:
            for child in (child_a, child_b):
                try:
                    child = await self._validate_and_repair_crossover(child, cp)
                except Exception as e:
                    logger.debug(f"[GPM] Crossover validation skipped: {e}")

        return child_a, child_b

    async def _validate_and_repair_crossover(
        self,
        child: PlanChromosome,
        cp:    int,
    ) -> PlanChromosome:
        """Request LLM coherence check and inject bridge step if needed."""
        first_half  = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(child.steps[:cp]))
        second_half = "\n".join(f"  {cp+i+1}. {s}" for i, s in enumerate(child.steps[cp:]))

        prompt = PROMPT_CROSSOVER_COHERENCE.format(
            aim              = self.aim,
            crossover_point  = cp,
            first_half       = first_half,
            second_half      = second_half,
            crossover_point_plus1 = cp + 1,
        )
        try:
            raw = await self.mutate_fn(prompt, child.parent_agent)
            if isinstance(raw, str):
                import json as _json
                raw = _json.loads(raw)

            if raw.get("bridge_step_needed") and raw.get("bridge_step_text"):
                bridge_idx = raw.get("insert_at_index", cp)
                child.steps.insert(bridge_idx, raw["bridge_step_text"])
                logger.debug(f"[GPM] Bridge step inserted at idx {bridge_idx}")
        except Exception as e:
            logger.debug(f"[GPM] Crossover coherence check failed: {e}")
        return child

    # ── Mutation ──────────────────────────────────────────────────────────────

    async def point_mutate(
        self,
        chromosome:    PlanChromosome,
        mutation_rate: float = GPM_MUTATION_PROB_PER_STEP,
    ) -> PlanChromosome:
        """
        For each step, with probability mutation_rate:
        call PROMPT_STEP_MUTATOR to rewrite that single step.
        All mutations are run in parallel.
        """
        mutated = chromosome.clone()
        indices_to_mutate = [
            i for i in range(len(mutated.steps))
            if random.random() < mutation_rate
        ]

        if not indices_to_mutate:
            return mutated

        async def mutate_step(idx: int) -> Tuple[int, str]:
            prev_s = mutated.steps[idx - 1] if idx > 0 else "(first step)"
            next_s = mutated.steps[idx + 1] if idx < len(mutated.steps) - 1 else "(last step)"
            all_steps_text = "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(mutated.steps)
            )
            prompt = PROMPT_STEP_MUTATOR.format(
                aim          = self.aim,
                all_steps    = all_steps_text,
                step_idx     = idx + 1,
                current_step = mutated.steps[idx],
                prev_step    = prev_s,
                next_step    = next_s,
            )
            try:
                raw = await self.mutate_fn(prompt, mutated.parent_agent)
                if isinstance(raw, str):
                    import json as _json
                    raw = _json.loads(raw)
                new_step = raw.get("mutated_step", mutated.steps[idx])
                return idx, new_step
            except Exception as e:
                logger.debug(f"[GPM] Step mutation failed at idx {idx}: {e}")
                return idx, mutated.steps[idx]

        mutations = await asyncio.gather(*[mutate_step(i) for i in indices_to_mutate])
        for idx, new_step in mutations:
            if new_step != mutated.steps[idx]:
                mutated.steps[idx] = new_step
                mutated.mutation_count += 1

        logger.debug(
            f"[GPM] Mutated {mutated.mutation_count} steps in {mutated.chromosome_id}"
        )
        return mutated

    # ── Immigrant Injection ───────────────────────────────────────────────────

    async def inject_immigrant(self) -> Optional[PlanChromosome]:
        """
        Generate a completely fresh plan to escape local fitness maxima.
        Uses a randomly selected agent from the initial population.
        """
        best = self.best_ever
        agents_available = list({c.parent_agent for c in self.population})
        if not agents_available:
            return None

        agent = random.choice(agents_available)
        best_plan_text = best.to_plan_string() if best else "None"

        prompt = PROMPT_IMMIGRANT.format(
            best_fitness  = f"{best.fitness_score:.1f}" if best else "N/A",
            plateau_gens  = self._plateau_count,
            current_best  = best_plan_text[:800],
            aim           = self.aim,
            initial_steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.initial_steps)),
        )

        try:
            raw_text = await self.generate_fn(self.aim, self.initial_steps, agent)
            steps = self._parse_steps_from_plan(raw_text)
            if len(steps) < 2:
                return None

            immigrant = PlanChromosome(
                chromosome_id = f"chr_immigrant_{uuid.uuid4().hex[:6]}",
                parent_agent  = agent,
                steps         = steps,
                generation    = len(self.history),
                is_immigrant  = True,
            )
            logger.info(f"[GPM] Immigrant injected: {len(steps)} steps from {agent}")
            return immigrant
        except Exception as e:
            logger.warning(f"[GPM] Immigrant generation failed: {e}")
            return None

    # ── Plateau Detection ─────────────────────────────────────────────────────

    def detect_plateau(self, patience: int = GPM_PLATEAU_PATIENCE) -> Tuple[bool, int]:
        """
        Plateau = max_fitness unchanged across `patience` consecutive generations.
        Returns (plateau_detected: bool, consecutive_flat_gens: int).
        """
        if len(self.history) < patience:
            return False, self._plateau_count

        recent = self.history[-patience:]
        max_scores = [g.max_fitness for g in recent]

        if max(max_scores) - min(max_scores) < 0.5:   # within 0.5 points = plateau
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        return self._plateau_count >= patience, self._plateau_count

    # ── Master Evolution Loop ─────────────────────────────────────────────────

    async def evolve(
        self,
        max_generations: int = GPM_MAX_GENERATIONS,
    ) -> Tuple[PlanChromosome, List[GenerationStats]]:
        """
        Main evolutionary loop.

        Each generation:
        1. Evaluate fitness of entire population
        2. Update best_ever
        3. Detect plateau → inject immigrant if stuck
        4. Elitism: reserve top GPM_ELITISM_K individuals
        5. Tournament selection of parents
        6. Crossover with probability GPM_CROSSOVER_PROB
        7. Mutate offspring
        8. Form next generation: elites + offspring
        9. Collect generation stats

        Returns (best_chromosome_ever, generation_history).
        """
        if not self.population:
            raise ValueError("[GPM] Population is empty. Call initialize_population() first.")

        logger.info(
            f"[GPM] Starting evolution: {len(self.population)} individuals, "
            f"max {max_generations} generations"
        )

        for gen_num in range(max_generations):
            gen_start = time.monotonic()
            operations = {"crossover": 0, "mutation": 0, "immigrant": 0, "elitism": 0}

            # ── 1. Evaluate fitness ──────────────────────────────────────────
            self.population = await self.evaluate_fitness_async(self.population)
            self.population.sort(key=lambda c: c.fitness_score, reverse=True)

            # ── 2. Track best_ever ───────────────────────────────────────────
            current_best = self.population[0]
            if self.best_ever is None or current_best.fitness_score > self.best_ever.fitness_score:
                self.best_ever = current_best.clone()
                logger.info(
                    f"[GPM] Gen {gen_num}: New best! {self.best_ever.chromosome_id} "
                    f"score={self.best_ever.fitness_score:.1f}"
                )

            # ── 3. Stats + plateau check ────────────────────────────────────
            scores = [c.fitness_score for c in self.population]
            prev_max = self.history[-1].max_fitness if self.history else 0.0
            stats = GenerationStats(
                generation         = gen_num,
                population_size    = len(self.population),
                max_fitness        = max(scores),
                mean_fitness       = statistics.mean(scores),
                min_fitness        = min(scores),
                best_chromosome_id = current_best.chromosome_id,
                plateau_count      = self._plateau_count,
                fitness_delta      = max(scores) - prev_max,
                operations         = operations,
            )
            self.history.append(stats)

            plateau_detected, plateau_count = self.detect_plateau()

            # ── 4. Elitism ───────────────────────────────────────────────────
            elites = [c.clone() for c in self.population[:GPM_ELITISM_K]]
            operations["elitism"] = len(elites)

            # ── 5. Early stopping ────────────────────────────────────────────
            if plateau_detected and gen_num >= 2:
                logger.info(
                    f"[GPM] Plateau detected at gen {gen_num} "
                    f"(flat for {plateau_count} gens). "
                    f"Injecting immigrant then stopping."
                )
                immigrant = await self.inject_immigrant()
                if immigrant:
                    immigrant = await self.evaluate_fitness_async([immigrant])
                    immigrant = immigrant[0]
                    if immigrant.fitness_score > self.best_ever.fitness_score:
                        self.best_ever = immigrant
                        logger.info(f"[GPM] Immigrant beats best! score={immigrant.fitness_score:.1f}")
                break

            # ── 6. Parent selection ──────────────────────────────────────────
            n_offspring = len(self.population) - GPM_ELITISM_K
            parents     = self.tournament_selection(self.population, n_offspring * 2)

            # ── 7. Crossover + mutation ──────────────────────────────────────
            offspring: List[PlanChromosome] = []
            parent_pairs = [
                (parents[i * 2], parents[i * 2 + 1])
                for i in range(min(n_offspring // 2, len(parents) // 2))
            ]

            crossover_tasks = []
            for pa, pb in parent_pairs:
                if random.random() < GPM_CROSSOVER_PROB:
                    crossover_tasks.append(self.single_point_crossover(pa, pb))
                    operations["crossover"] += 1
                else:
                    offspring.extend([pa.clone(), pb.clone()])

            if crossover_tasks:
                child_pairs = await asyncio.gather(*crossover_tasks)
                for ca, cb in child_pairs:
                    offspring.extend([ca, cb])

            mutation_tasks = [self.point_mutate(c) for c in offspring]
            offspring = list(await asyncio.gather(*mutation_tasks))
            operations["mutation"] = sum(c.mutation_count for c in offspring)

            # Mark generation
            for c in offspring:
                c.generation = gen_num + 1

            # ── 8. Immigrant injection (per-generation rate) ─────────────────
            n_immigrants = max(1, int(len(offspring) * GPM_IMMIGRANT_RATE))
            for _ in range(n_immigrants):
                immigrant = await self.inject_immigrant()
                if immigrant:
                    offspring.append(immigrant)
                    operations["immigrant"] += 1

            # ── 9. Next generation ───────────────────────────────────────────
            self.population = elites + offspring[:len(self.population) - GPM_ELITISM_K]

            logger.info(
                f"[GPM] Gen {gen_num} complete | "
                f"max={max(scores):.1f} mean={statistics.mean(scores):.1f} | "
                f"ops={operations} | time={time.monotonic()-gen_start:.1f}s"
            )

        # Final evaluation pass
        self.population = await self.evaluate_fitness_async(self.population)
        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        final_best = self.population[0]

        if self.best_ever is None or final_best.fitness_score > self.best_ever.fitness_score:
            self.best_ever = final_best

        logger.info(
            f"[GPM] Evolution complete. Best ever: {self.best_ever.chromosome_id} "
            f"score={self.best_ever.fitness_score:.1f} "
            f"agent={self.best_ever.parent_agent} "
            f"mutations={self.best_ever.mutation_count}"
        )
        return self.best_ever, self.history

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_evolution_report(self) -> Dict:
        """Return structured evolution history for Excel/frontend."""
        return {
            "generations":      len(self.history),
            "final_best_score": self.best_ever.fitness_score if self.best_ever else 0,
            "initial_best_score": self.history[0].max_fitness if self.history else 0,
            "improvement_delta": (
                (self.best_ever.fitness_score - self.history[0].max_fitness)
                if self.best_ever and self.history else 0
            ),
            "best_agent":       self.best_ever.parent_agent if self.best_ever else "N/A",
            "total_mutations":  sum(g.operations.get("mutation", 0) for g in self.history),
            "total_crossovers": sum(g.operations.get("crossover", 0) for g in self.history),
            "immigrants_used":  sum(g.operations.get("immigrant", 0) for g in self.history),
            "plateau_detected": self._plateau_count >= GPM_PLATEAU_PATIENCE,
            "generation_history": [
                {
                    "gen":       g.generation,
                    "max":       round(g.max_fitness, 2),
                    "mean":      round(g.mean_fitness, 2),
                    "min":       round(g.min_fitness, 2),
                    "delta":     round(g.fitness_delta, 2),
                    "ops":       g.operations,
                }
                for g in self.history
            ],
        }
