# ═══════════════════════════════════════════════════════════════════════════════
# temporal_execution_simulator.py — Feature 8: TES
# Monte Carlo timeline simulation, conflict detection, Gantt data.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

from system_config import (
    TES_MONTE_CARLO_N, TES_DURATION_VARIANCE,
    TES_PARALLEL_MIN_SAVINGS_PCT, TES_CONFLICT_DETECTION_ENABLED,
    TES_DEFAULT_DOMAIN
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class TemporalStep:
    step_id:                  str
    step_text:                str
    estimated_hrs_p10:        float   = 2.0
    estimated_hrs_p50:        float   = 4.0
    estimated_hrs_p90:        float   = 8.0
    resources_required:       List[str] = field(default_factory=list)
    can_parallelise_with:     List[str] = field(default_factory=list)
    parallel_group:           Optional[int] = None
    earliest_start_hrs:       float   = 0.0
    latest_finish_hrs:        float   = 0.0
    float_hrs:                float   = 0.0   # slack before becoming critical
    hard_deadline_hrs:        Optional[float] = None
    estimation_confidence:    str     = "medium"
    is_on_critical_path:      bool    = False


@dataclass
class ResourceConflict:
    conflict_id:         str
    resource:            str
    step_a:              str
    step_b:              str
    overlap_start_hrs:   float
    overlap_end_hrs:     float
    severity:            str   # critical | major | minor


@dataclass
class MonteCarloResult:
    n_simulations:       int
    p10_hrs:             float
    p50_hrs:             float
    p90_hrs:             float
    mean_hrs:            float
    std_hrs:             float
    min_hrs:             float
    max_hrs:             float
    simulated_totals:    List[float]   # raw samples (truncated to 100 for storage)


@dataclass
class SimulationReport:
    steps:               List[TemporalStep]
    monte_carlo:         MonteCarloResult
    conflicts:           List[ResourceConflict]
    gantt_data:          List[Dict]
    parallelism_savings_pct: float
    sequential_total_hrs:float
    parallel_total_hrs:  float
    critical_path_hrs:   float


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_TES_DURATION_ESTIMATOR = """You are an EXECUTION TIME ANALYST for a planning system.
Use domain expertise to estimate realistic execution times.

AIM: {aim}
DOMAIN: {domain}
PLAN STEPS:
{numbered_steps}

For each step estimate using 3-point estimation:
  P10 = Optimistic (10th percentile — best case)
  P50 = Realistic  (50th percentile — most likely)
  P90 = Pessimistic (90th percentile — delayed/rework)

Also identify required resources and parallelisation opportunities.

Respond ONLY with valid JSON (no markdown):
{{"steps":[
  {{
    "step_id":"step_1",
    "estimated_hrs_p10":4.0,
    "estimated_hrs_p50":8.0,
    "estimated_hrs_p90":16.0,
    "resources_required":["senior_developer","AWS_account"],
    "can_parallelise_with":["step_2"],
    "hard_deadline_hrs":null,
    "estimation_confidence":"high|medium|low"
  }}
],
"total_p50_hrs":0.0,
"estimated_calendar_days":0,
"parallelism_savings_pct":0.0}}"""

PROMPT_TES_CONFLICT_RESOLVER = """You are a RESOURCE CONFLICT RESOLVER.

AIM: {aim}
DETECTED CONFLICTS:
{conflicts_json}

For each conflict propose a concrete resolution:
  resequence | add_resource | reduce_scope | split_step

Respond ONLY with valid JSON:
{{"resolutions":[
  {{
    "conflict_id":"conflict_1",
    "conflicting_steps":["step_3","step_5"],
    "root_cause":"...",
    "resolution_type":"resequence",
    "resolution_action":"...",
    "cost_hrs":0.0,
    "risk_introduced":"low|medium|high"
  }}
],
"revised_total_p50_hrs":0.0}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class TemporalExecutionSimulator:
    """
    Assigns durations, identifies parallel clusters, runs Monte Carlo,
    detects resource conflicts, and produces Gantt-compatible output.
    """

    def __init__(
        self,
        call_fn:       Callable[[str, str], Awaitable[str]],
        primary_agent: str = "gemini",
    ):
        self.call_fn       = call_fn
        self.primary_agent = primary_agent
        self._steps:       List[TemporalStep]       = []
        self._conflicts:   List[ResourceConflict]   = []
        self._mc_result:   Optional[MonteCarloResult] = None

    # ── Duration Estimation ───────────────────────────────────────────────────

    async def estimate_step_durations(
        self,
        steps:  List[str],
        aim:    str,
        domain: str = TES_DEFAULT_DOMAIN,
    ) -> List[TemporalStep]:
        """LLM estimates P10/P50/P90 + resources for all steps."""
        numbered = "\n".join(f"  step_{i+1}: {s}" for i, s in enumerate(steps))
        prompt   = PROMPT_TES_DURATION_ESTIMATOR.format(
            aim           = aim,
            domain        = domain,
            numbered_steps= numbered,
        )
        try:
            raw  = await self.call_fn(self.primary_agent, prompt)
            data = _parse_json(raw)
            ts_list = []
            for sd in data.get("steps", []):
                ts = TemporalStep(
                    step_id                = sd.get("step_id", f"step_{len(ts_list)+1}"),
                    step_text              = steps[len(ts_list)] if len(ts_list) < len(steps) else "",
                    estimated_hrs_p10      = float(sd.get("estimated_hrs_p10", 2.0)),
                    estimated_hrs_p50      = float(sd.get("estimated_hrs_p50", 4.0)),
                    estimated_hrs_p90      = float(sd.get("estimated_hrs_p90", 8.0)),
                    resources_required     = sd.get("resources_required", []),
                    can_parallelise_with   = sd.get("can_parallelise_with", []),
                    hard_deadline_hrs      = sd.get("hard_deadline_hrs"),
                    estimation_confidence  = sd.get("estimation_confidence", "medium"),
                )
                ts_list.append(ts)

            # Pad with heuristic estimates if LLM returned fewer entries
            while len(ts_list) < len(steps):
                i = len(ts_list)
                ts_list.append(TemporalStep(
                    step_id  = f"step_{i+1}",
                    step_text= steps[i],
                    estimated_hrs_p10= 2.0,
                    estimated_hrs_p50= 4.0,
                    estimated_hrs_p90= 8.0,
                ))

            self._steps = ts_list
            logger.info(f"[TES] Duration estimates: {len(ts_list)} steps, "
                        f"total P50={sum(s.estimated_hrs_p50 for s in ts_list):.1f}h")
            return ts_list

        except Exception as e:
            logger.warning(f"[TES] Duration estimation failed: {e}. Using heuristic.")
            return self._heuristic_durations(steps)

    def _heuristic_durations(self, steps: List[str]) -> List[TemporalStep]:
        ts_list = []
        for i, step in enumerate(steps):
            # Rough heuristic: longer step text = more complex = more time
            words = len(step.split())
            base  = max(2.0, min(16.0, words * 0.4))
            ts = TemporalStep(
                step_id           = f"step_{i+1}",
                step_text         = step,
                estimated_hrs_p10 = base * 0.5,
                estimated_hrs_p50 = base,
                estimated_hrs_p90 = base * 2.0,
            )
            ts_list.append(ts)
        self._steps = ts_list
        return ts_list

    # ── Parallel Clusters ─────────────────────────────────────────────────────

    def identify_parallel_clusters(
        self,
        steps:      Optional[List[TemporalStep]] = None,
        causal_dag: Optional[Dict] = None,
    ) -> Dict[int, List[str]]:
        """
        Groups steps that can run simultaneously.
        Uses can_parallelise_with hints from LLM estimation,
        augmented by causal_dag independence if available.
        """
        steps = steps or self._steps
        clusters: Dict[int, List[str]] = {}
        assigned = set()
        group_id = 0

        for ts in steps:
            if ts.step_id in assigned:
                continue
            if ts.can_parallelise_with:
                group = [ts.step_id] + [
                    pid for pid in ts.can_parallelise_with
                    if pid not in assigned
                ]
                for sid in group:
                    assigned.add(sid)
                    # Find and mark the TemporalStep
                    for s in steps:
                        if s.step_id == sid:
                            s.parallel_group = group_id
                clusters[group_id] = group
                group_id += 1

        return clusters

    # ── Critical Path Timeline ────────────────────────────────────────────────

    def compute_critical_path_timeline(
        self,
        steps:         Optional[List[TemporalStep]] = None,
        prerequisites: Optional[Dict[str, List[str]]] = None,
    ) -> List[TemporalStep]:
        """
        Computes earliest_start_hrs and latest_finish_hrs using
        forward/backward pass (CPM algorithm).
        """
        steps = steps or self._steps
        if not steps:
            return steps

        prereqs = prerequisites or {}
        earliest_start: Dict[str, float] = {}
        earliest_finish: Dict[str, float] = {}

        # Forward pass
        for ts in steps:
            preds = prereqs.get(ts.step_id, [])
            es = max(
                (earliest_finish.get(p, 0.0) for p in preds),
                default=0.0
            )
            earliest_start[ts.step_id]  = es
            earliest_finish[ts.step_id] = es + ts.estimated_hrs_p50
            ts.earliest_start_hrs       = es

        project_finish = max(earliest_finish.values(), default=0.0)

        # Backward pass
        latest_start:  Dict[str, float] = {}
        latest_finish: Dict[str, float] = {ts.step_id: project_finish for ts in steps}

        for ts in reversed(steps):
            successors = [s.step_id for s in steps
                          if ts.step_id in prereqs.get(s.step_id, [])]
            lf = min((latest_start.get(s, project_finish) for s in successors),
                     default=project_finish)
            latest_finish[ts.step_id] = lf
            latest_start[ts.step_id]  = lf - ts.estimated_hrs_p50
            ts.latest_finish_hrs      = lf
            ts.float_hrs              = lf - ts.estimated_hrs_p50 - earliest_start[ts.step_id]
            ts.is_on_critical_path    = ts.float_hrs <= 0.01

        return steps

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def run_monte_carlo(
        self,
        steps:        Optional[List[TemporalStep]] = None,
        n_simulations:int = TES_MONTE_CARLO_N,
        variance:     float = TES_DURATION_VARIANCE,
    ) -> MonteCarloResult:
        """
        Samples step durations N times using triangular distribution
        parameterised by (P10, P50, P90). Returns completion time distribution.
        """
        steps = steps or self._steps
        totals = []

        for _ in range(n_simulations):
            total = 0.0
            for ts in steps:
                # Triangular distribution: low=p10, mode=p50, high=p90
                sample = random.triangular(
                    ts.estimated_hrs_p10,
                    ts.estimated_hrs_p90,
                    ts.estimated_hrs_p50,
                )
                total += max(0.1, sample)
            totals.append(total)

        totals.sort()
        n    = len(totals)
        p10i = max(0, int(n * 0.10) - 1)
        p50i = max(0, int(n * 0.50) - 1)
        p90i = max(0, int(n * 0.90) - 1)

        result = MonteCarloResult(
            n_simulations    = n_simulations,
            p10_hrs          = totals[p10i],
            p50_hrs          = totals[p50i],
            p90_hrs          = totals[p90i],
            mean_hrs         = statistics.mean(totals),
            std_hrs          = statistics.stdev(totals) if n > 1 else 0.0,
            min_hrs          = totals[0],
            max_hrs          = totals[-1],
            simulated_totals = totals[::max(1, n//100)],  # store ~100 samples
        )
        self._mc_result = result
        logger.info(
            f"[TES] Monte Carlo ({n_simulations} runs): "
            f"P10={result.p10_hrs:.1f}h P50={result.p50_hrs:.1f}h P90={result.p90_hrs:.1f}h"
        )
        return result

    # ── Conflict Detection ────────────────────────────────────────────────────

    def detect_resource_conflicts(
        self,
        steps: Optional[List[TemporalStep]] = None,
    ) -> List[ResourceConflict]:
        """
        Identifies steps that compete for the same resource in overlapping windows.
        Uses earliest_start_hrs + p50 as the time window per step.
        """
        if not TES_CONFLICT_DETECTION_ENABLED:
            return []

        steps = steps or self._steps
        conflicts: List[ResourceConflict] = []
        cid = 0

        for i, sa in enumerate(steps):
            for sb in steps[i+1:]:
                shared_resources = set(sa.resources_required) & set(sb.resources_required)
                if not shared_resources:
                    continue

                # Check time window overlap
                a_start = sa.earliest_start_hrs
                a_end   = a_start + sa.estimated_hrs_p50
                b_start = sb.earliest_start_hrs
                b_end   = b_start + sb.estimated_hrs_p50

                overlap_start = max(a_start, b_start)
                overlap_end   = min(a_end, b_end)

                if overlap_end > overlap_start:
                    overlap_hrs = overlap_end - overlap_start
                    severity = (
                        "critical" if overlap_hrs > sa.estimated_hrs_p50 * 0.5 else
                        "major"    if overlap_hrs > 2.0 else
                        "minor"
                    )
                    for resource in shared_resources:
                        cid += 1
                        conflicts.append(ResourceConflict(
                            conflict_id       = f"conflict_{cid}",
                            resource          = resource,
                            step_a            = sa.step_id,
                            step_b            = sb.step_id,
                            overlap_start_hrs = overlap_start,
                            overlap_end_hrs   = overlap_end,
                            severity          = severity,
                        ))

        self._conflicts = conflicts
        if conflicts:
            logger.warning(f"[TES] {len(conflicts)} resource conflicts detected")
        return conflicts

    # ── Gantt Data ────────────────────────────────────────────────────────────

    def build_gantt_data(
        self,
        steps: Optional[List[TemporalStep]] = None,
    ) -> List[Dict]:
        """Returns Gantt-chart-compatible JSON for frontend rendering."""
        steps = steps or self._steps
        return [
            {
                "step_id":             ts.step_id,
                "step_text":           ts.step_text[:80],
                "start_hrs":           round(ts.earliest_start_hrs, 2),
                "end_hrs":             round(ts.earliest_start_hrs + ts.estimated_hrs_p50, 2),
                "duration_p50_hrs":    round(ts.estimated_hrs_p50, 2),
                "duration_p10_hrs":    round(ts.estimated_hrs_p10, 2),
                "duration_p90_hrs":    round(ts.estimated_hrs_p90, 2),
                "resources":           ts.resources_required,
                "parallel_group":      ts.parallel_group,
                "is_critical":         ts.is_on_critical_path,
                "float_hrs":           round(ts.float_hrs, 2),
                "confidence":          ts.estimation_confidence,
            }
            for ts in steps
        ]

    # ── Parallelism Savings ───────────────────────────────────────────────────

    def compute_parallelism_savings(
        self,
        steps: Optional[List[TemporalStep]] = None,
    ) -> Tuple[float, float, float]:
        """
        Returns (sequential_total_hrs, parallel_total_hrs, savings_pct).
        Sequential = sum of all P50 durations.
        Parallel   = sum of max(P50) per parallel group + ungrouped steps.
        """
        steps = steps or self._steps
        sequential = sum(ts.estimated_hrs_p50 for ts in steps)

        # Group by parallel_group
        groups: Dict[int, float] = {}
        ungrouped = 0.0
        for ts in steps:
            if ts.parallel_group is not None:
                g = ts.parallel_group
                groups[g] = max(groups.get(g, 0.0), ts.estimated_hrs_p50)
            else:
                ungrouped += ts.estimated_hrs_p50

        parallel = sum(groups.values()) + ungrouped
        savings  = max(0.0, (sequential - parallel) / max(sequential, 1) * 100)
        return sequential, parallel, savings

    # ── Master Pipeline ───────────────────────────────────────────────────────

    async def simulate(
        self,
        steps:         List[str],
        aim:           str,
        domain:        str = TES_DEFAULT_DOMAIN,
        prerequisites: Optional[Dict[str, List[str]]] = None,
    ) -> SimulationReport:
        """
        Full TES pipeline:
        estimate → cluster → CPM → Monte Carlo → conflicts → Gantt
        """
        ts_list = await self.estimate_step_durations(steps, aim, domain)
        self.identify_parallel_clusters(ts_list)
        self.compute_critical_path_timeline(ts_list, prerequisites)
        mc      = self.run_monte_carlo(ts_list)
        confl   = self.detect_resource_conflicts(ts_list)
        gantt   = self.build_gantt_data(ts_list)
        seq, par, savings = self.compute_parallelism_savings(ts_list)
        cp_hrs  = sum(ts.estimated_hrs_p50 for ts in ts_list if ts.is_on_critical_path)

        report = SimulationReport(
            steps                = ts_list,
            monte_carlo          = mc,
            conflicts            = confl,
            gantt_data           = gantt,
            parallelism_savings_pct = savings,
            sequential_total_hrs = seq,
            parallel_total_hrs   = par,
            critical_path_hrs    = cp_hrs,
        )
        logger.info(
            f"[TES] Simulation complete. seq={seq:.1f}h par={par:.1f}h "
            f"savings={savings:.1f}% conflicts={len(confl)}"
        )
        return report

    def get_report_dict(self, report: Optional[SimulationReport] = None) -> Dict:
        """Serialise for Excel/frontend output."""
        if not report:
            return {}
        mc = report.monte_carlo
        return {
            "p10_hrs":                mc.p10_hrs,
            "p50_hrs":                mc.p50_hrs,
            "p90_hrs":                mc.p90_hrs,
            "mean_hrs":               mc.mean_hrs,
            "sequential_total_hrs":   report.sequential_total_hrs,
            "parallel_total_hrs":     report.parallel_total_hrs,
            "parallelism_savings_pct":report.parallelism_savings_pct,
            "critical_path_hrs":      report.critical_path_hrs,
            "conflict_count":         len(report.conflicts),
            "conflicts":              [
                {"id": c.conflict_id, "resource": c.resource,
                 "steps": [c.step_a, c.step_b], "severity": c.severity}
                for c in report.conflicts
            ],
            "gantt_data":             report.gantt_data,
            "step_count":             len(report.steps),
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
