# ═══════════════════════════════════════════════════════════════════════════════
# execution_simulation_sandbox.py — Feature 23: ESS
# Before delivering the plan, simulates executing it step-by-step in a
# virtual environment. Predicts failure points, resource exhaustion, and
# deadline violations — before they happen in reality.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

ESS_SIMULATION_RUNS     = 20     # Monte Carlo execution trials
ESS_FAILURE_RATE_DEFAULT= 0.05   # 5% per-step random failure baseline
ESS_MAX_STEPS_TO_SIM    = 15     # cap simulation depth
ESS_RESOURCE_POOL = {            # default resource pool
    "developers":  3,
    "budget_usd":  50_000,
    "days":        90,
    "servers":     2,
    "testers":     1,
}


class StepOutcome(Enum):
    SUCCESS    = "success"
    PARTIAL    = "partial"       # completed but took longer
    FAILED     = "failed"        # step failed, recovery needed
    BLOCKED    = "blocked"       # resource unavailable
    SKIPPED    = "skipped"       # dependency failed, step skipped


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SimulatedStep:
    step_id:          str
    step_text:        str
    outcome:          StepOutcome
    actual_hrs:       float
    resources_consumed:Dict[str, float]
    failure_reason:   Optional[str]
    recovery_action:  Optional[str]
    cumulative_hrs:   float
    cumulative_cost:  float


@dataclass
class ExecutionTrial:
    trial_id:         int
    steps:            List[SimulatedStep]
    final_outcome:    str       # completed | deadline_missed | resource_exhausted | critical_failure
    total_hrs:        float
    total_cost:       float
    steps_failed:     int
    steps_completed:  int
    deadline_met:     bool
    failure_cascade:  Optional[str]   # which step triggered cascade failure


@dataclass
class SandboxReport:
    plan_text:             str
    trials:                List[ExecutionTrial]
    completion_rate:       float        # fraction of trials that completed
    avg_total_hrs:         float
    p50_hrs:               float
    p90_hrs:               float
    deadline_success_rate: float
    most_failed_step:      str
    resource_bottleneck:   Optional[str]
    predicted_failure_points: List[Dict]
    pre_emptive_fixes:     List[str]    # suggested plan amendments
    sandbox_verdict:       str          # safe_to_deliver | needs_hardening | high_risk


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_ESS_STEP_SIMULATOR = """You are an EXECUTION SIMULATOR.

AIM: {aim}
STEP TO SIMULATE: "{step_text}"
TRIAL CONTEXT:
  - Steps completed so far: {steps_done}/{total_steps}
  - Resources remaining: {resources_json}
  - Cumulative hours spent: {cumulative_hrs:.1f}h
  - Previous failures: {prev_failures}

Simulate executing THIS step realistically. Consider:
  • What commonly goes wrong in this type of task?
  • Are resources sufficient?
  • Does this step depend on a previous step that may have failed?

Respond ONLY with valid JSON:
{{
  "outcome": "success|partial|failed|blocked|skipped",
  "actual_hrs": 0.0,
  "resources_consumed": {{"developers": 0, "budget_usd": 0}},
  "failure_reason": null,
  "recovery_action": null,
  "notes": "..."
}}"""

PROMPT_ESS_FAILURE_ANALYSER = """You are a FAILURE PATTERN ANALYST.

AIM: {aim}
PLAN:
{plan_text}

SIMULATION RESULTS ({n_trials} trials):
  - Completion rate: {completion_rate:.0%}
  - Most failed step: {most_failed}
  - Resource bottleneck: {bottleneck}
  - Average duration: {avg_hrs:.1f}h (P90: {p90_hrs:.1f}h)

FAILURE PATTERNS:
{failure_patterns}

Propose 3–5 CONCRETE plan amendments to improve the simulation success rate.
Each amendment should directly address a failure pattern.

Respond ONLY with valid JSON:
{{
  "pre_emptive_fixes": [
    "Add a resource pre-acquisition step before step_N",
    "Split step_M into two parallel tasks to reduce bottleneck",
    "Add a fallback path after step_K in case of API unavailability"
  ],
  "sandbox_verdict": "safe_to_deliver|needs_hardening|high_risk",
  "verdict_rationale": "..."
}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class ExecutionSimulationSandbox:
    """
    Runs N Monte Carlo execution trials of the plan in a virtual sandbox.

    Each trial:
    1. Initialises virtual resource pool
    2. Executes each step via LLM simulation (or heuristic for speed)
    3. Tracks resource consumption, failures, recovery actions
    4. Checks deadline + resource constraints at each step
    5. Records final outcome

    After N trials: identifies failure patterns, bottlenecks, pre-emptive fixes.
    """

    def __init__(
        self,
        call_fn:        Callable[[str, str], Awaitable[str]],
        agent:          str = "gemini",
        resource_pool:  Optional[Dict[str, float]] = None,
        deadline_hrs:   Optional[float] = None,
        n_trials:       int = ESS_SIMULATION_RUNS,
        use_llm_steps:  bool = False,   # True = LLM per step (slow), False = heuristic (fast)
    ):
        self.call_fn       = call_fn
        self.agent         = agent
        self.resource_pool = resource_pool or dict(ESS_RESOURCE_POOL)
        self.deadline_hrs  = deadline_hrs or self.resource_pool.get("days", 90) * 8
        self.n_trials      = n_trials
        self.use_llm_steps = use_llm_steps

    # ── Step Simulation ───────────────────────────────────────────────────────

    async def simulate_step_llm(
        self,
        step_text:      str,
        step_id:        str,
        step_idx:       int,
        total_steps:    int,
        resources:      Dict[str, float],
        cumulative_hrs: float,
        prev_failures:  int,
        aim:            str,
    ) -> SimulatedStep:
        """LLM-powered step simulation (accurate but slow)."""
        prompt = PROMPT_ESS_STEP_SIMULATOR.format(
            aim            = aim,
            step_text      = step_text[:200],
            steps_done     = step_idx,
            total_steps    = total_steps,
            resources_json = json.dumps({k: round(v, 1) for k, v in resources.items()}),
            cumulative_hrs = cumulative_hrs,
            prev_failures  = prev_failures,
        )
        try:
            raw    = await self.call_fn(self.agent, prompt)
            data   = _parse_json(raw)
            outcome_str = data.get("outcome", "success").lower()
            try:
                outcome = StepOutcome(outcome_str)
            except ValueError:
                outcome = StepOutcome.SUCCESS

            actual_hrs = float(data.get("actual_hrs", 4.0))
            consumed   = data.get("resources_consumed", {})

            return SimulatedStep(
                step_id           = step_id,
                step_text         = step_text,
                outcome           = outcome,
                actual_hrs        = actual_hrs,
                resources_consumed= consumed,
                failure_reason    = data.get("failure_reason"),
                recovery_action   = data.get("recovery_action"),
                cumulative_hrs    = cumulative_hrs + actual_hrs,
                cumulative_cost   = consumed.get("budget_usd", 0),
            )
        except Exception:
            return self._heuristic_step(step_text, step_id, cumulative_hrs, resources, prev_failures)

    def _heuristic_step(
        self,
        step_text:      str,
        step_id:        str,
        cumulative_hrs: float,
        resources:      Dict[str, float],
        prev_failures:  int,
    ) -> SimulatedStep:
        """Fast statistical step simulation — no LLM."""
        # Failure probability increases with step complexity and prior failures
        complexity    = min(1.0, len(step_text.split()) / 50)
        failure_prob  = ESS_FAILURE_RATE_DEFAULT + complexity * 0.05 + prev_failures * 0.02

        # Resource check
        devs_needed   = max(0.5, complexity * 2)
        blocked       = resources.get("developers", 0) < devs_needed

        if blocked:
            outcome = StepOutcome.BLOCKED
            actual_hrs = 0.0
            failure_reason = "Insufficient developer capacity"
        elif random.random() < failure_prob:
            outcome    = StepOutcome.FAILED if random.random() < 0.5 else StepOutcome.PARTIAL
            actual_hrs = random.uniform(4, 16)
            failure_reason = random.choice([
                "Integration error", "Dependency unavailable",
                "Scope misunderstanding", "Technical blocker"
            ])
        else:
            outcome    = StepOutcome.SUCCESS
            base_hrs   = max(1.0, len(step_text.split()) * 0.3)
            actual_hrs = random.triangular(base_hrs * 0.7, base_hrs * 2.0, base_hrs)
            failure_reason = None

        cost = actual_hrs * 100  # $100/hr
        return SimulatedStep(
            step_id            = step_id,
            step_text          = step_text,
            outcome            = outcome,
            actual_hrs         = actual_hrs,
            resources_consumed = {"developers": devs_needed, "budget_usd": cost},
            failure_reason     = failure_reason,
            recovery_action    = "Re-attempt after resource resolution" if blocked else None,
            cumulative_hrs     = cumulative_hrs + actual_hrs,
            cumulative_cost    = cost,
        )

    # ── Trial Execution ───────────────────────────────────────────────────────

    async def run_trial(
        self,
        trial_id:    int,
        steps:       List[str],
        aim:         str,
    ) -> ExecutionTrial:
        """Execute one complete trial of the plan."""
        resources  = dict(self.resource_pool)
        sim_steps: List[SimulatedStep] = []
        cum_hrs    = 0.0
        cum_cost   = 0.0
        failures   = 0
        cascade    = None

        for i, step_text in enumerate(steps[:ESS_MAX_STEPS_TO_SIM]):
            step_id = f"step_{i+1}"

            if self.use_llm_steps and i < 3:  # LLM for first 3 steps max
                sim_step = await self.simulate_step_llm(
                    step_text, step_id, i, len(steps),
                    resources, cum_hrs, failures, aim
                )
            else:
                sim_step = self._heuristic_step(
                    step_text, step_id, cum_hrs, resources, failures
                )

            sim_steps.append(sim_step)

            # Update resource pool
            for resource, amount in sim_step.resources_consumed.items():
                resources[resource] = max(0, resources.get(resource, 0) - amount)

            cum_hrs  += sim_step.actual_hrs
            cum_cost += sim_step.cumulative_cost

            if sim_step.outcome == StepOutcome.FAILED:
                failures += 1

            # Check cascade failure
            if failures >= 3 and sim_step.outcome == StepOutcome.FAILED:
                cascade = step_id
                # Skip remaining steps
                for remaining in steps[i+1:ESS_MAX_STEPS_TO_SIM]:
                    sim_steps.append(SimulatedStep(
                        step_id=f"step_{len(sim_steps)+1}", step_text=remaining,
                        outcome=StepOutcome.SKIPPED, actual_hrs=0, resources_consumed={},
                        failure_reason="Cascade failure", recovery_action=None,
                        cumulative_hrs=cum_hrs, cumulative_cost=0,
                    ))
                break

            # Check deadline
            if cum_hrs > self.deadline_hrs:
                break

            # Check resource exhaustion
            if resources.get("budget_usd", 1) <= 0:
                break

        # Determine final outcome
        completed_count = sum(1 for s in sim_steps if s.outcome == StepOutcome.SUCCESS)
        failed_count    = sum(1 for s in sim_steps if s.outcome == StepOutcome.FAILED)

        if cascade:
            final_outcome = "critical_failure"
        elif cum_hrs > self.deadline_hrs:
            final_outcome = "deadline_missed"
        elif resources.get("budget_usd", 1) <= 0:
            final_outcome = "resource_exhausted"
        elif completed_count >= len(steps) * 0.85:
            final_outcome = "completed"
        else:
            final_outcome = "partial_completion"

        return ExecutionTrial(
            trial_id       = trial_id,
            steps          = sim_steps,
            final_outcome  = final_outcome,
            total_hrs      = cum_hrs,
            total_cost     = cum_cost,
            steps_failed   = failed_count,
            steps_completed= completed_count,
            deadline_met   = cum_hrs <= self.deadline_hrs,
            failure_cascade= cascade,
        )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _analyse_trials(
        self,
        trials:    List[ExecutionTrial],
        steps:     List[str],
    ) -> Dict:
        """Aggregate trial results into failure patterns."""
        n = len(trials)

        # Step-level failure counts
        step_failures: Dict[str, int] = {}
        for trial in trials:
            for step in trial.steps:
                if step.outcome in (StepOutcome.FAILED, StepOutcome.BLOCKED):
                    step_failures[step.step_id] = step_failures.get(step.step_id, 0) + 1

        most_failed     = max(step_failures, key=step_failures.get) if step_failures else "none"
        most_failed_pct = step_failures.get(most_failed, 0) / max(n, 1)

        # Resource bottleneck
        resource_shortages: Dict[str, int] = {}
        for trial in trials:
            for step in trial.steps:
                if step.outcome == StepOutcome.BLOCKED and step.failure_reason:
                    for resource in self.resource_pool:
                        if resource.lower() in step.failure_reason.lower():
                            resource_shortages[resource] = resource_shortages.get(resource, 0) + 1
        bottleneck = max(resource_shortages, key=resource_shortages.get) \
                     if resource_shortages else None

        # Duration distribution
        durations = sorted([t.total_hrs for t in trials])
        p50 = durations[len(durations)//2] if durations else 0
        p90 = durations[int(len(durations)*0.9)] if durations else 0

        completion_rate = sum(1 for t in trials if t.final_outcome == "completed") / max(n, 1)
        deadline_rate   = sum(1 for t in trials if t.deadline_met) / max(n, 1)

        # Build failure pattern list
        patterns = []
        for step_id, count in sorted(step_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
            idx = int(step_id.replace("step_", "")) - 1
            text = steps[idx] if idx < len(steps) else step_id
            patterns.append({
                "step_id":    step_id,
                "step_text":  text[:80],
                "fail_count": count,
                "fail_rate":  round(count / max(n, 1), 2),
            })

        return {
            "completion_rate":   round(completion_rate, 3),
            "deadline_rate":     round(deadline_rate, 3),
            "avg_hrs":           round(sum(durations)/max(len(durations),1), 1),
            "p50_hrs":           round(p50, 1),
            "p90_hrs":           round(p90, 1),
            "most_failed":       most_failed,
            "most_failed_pct":   round(most_failed_pct, 2),
            "bottleneck":        bottleneck,
            "failure_patterns":  patterns,
        }

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def simulate(
        self,
        plan_text: str,
        aim:       str,
    ) -> SandboxReport:
        """Full ESS pipeline: N trials → analysis → fixes → verdict."""
        steps = _parse_steps(plan_text)
        if not steps:
            steps = [plan_text[:200]]

        logger.info(f"[ESS] Running {self.n_trials} trials on {len(steps)} steps...")

        # Run trials (mostly heuristic for speed, LLM optional)
        tasks  = [self.run_trial(i, steps, aim) for i in range(self.n_trials)]
        trials = await asyncio.gather(*tasks)

        stats = self._analyse_trials(list(trials), steps)

        # LLM analyses results and proposes fixes
        prompt = PROMPT_ESS_FAILURE_ANALYSER.format(
            aim             = aim,
            plan_text       = plan_text[:800],
            n_trials        = self.n_trials,
            completion_rate = stats["completion_rate"],
            most_failed     = stats["most_failed"],
            bottleneck      = stats["bottleneck"] or "none",
            avg_hrs         = stats["avg_hrs"],
            p90_hrs         = stats["p90_hrs"],
            failure_patterns= json.dumps(stats["failure_patterns"], indent=2)[:600],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            fixes   = data.get("pre_emptive_fixes", [])
            verdict = data.get("sandbox_verdict", "needs_hardening")
        except Exception:
            fixes   = [f"Add pre-check for resource availability before {stats['most_failed']}"]
            verdict = "needs_hardening" if stats["completion_rate"] < 0.8 else "safe_to_deliver"

        # Predicted failure points
        predicted = [
            {
                "step":     p["step_id"],
                "text":     p["step_text"],
                "fail_rate":p["fail_rate"],
                "severity": "critical" if p["fail_rate"] > 0.4 else
                            "major"    if p["fail_rate"] > 0.2 else "minor",
            }
            for p in stats["failure_patterns"]
        ]

        report = SandboxReport(
            plan_text              = plan_text,
            trials                 = list(trials),
            completion_rate        = stats["completion_rate"],
            avg_total_hrs          = stats["avg_hrs"],
            p50_hrs                = stats["p50_hrs"],
            p90_hrs                = stats["p90_hrs"],
            deadline_success_rate  = stats["deadline_rate"],
            most_failed_step       = stats["most_failed"],
            resource_bottleneck    = stats["bottleneck"],
            predicted_failure_points= predicted,
            pre_emptive_fixes      = fixes,
            sandbox_verdict        = verdict,
        )

        logger.info(
            f"[ESS] Simulation complete. completion={stats['completion_rate']:.0%} "
            f"p90={stats['p90_hrs']:.1f}h verdict={verdict}"
        )
        return report

    def get_ess_report(self, report: SandboxReport) -> Dict:
        return {
            "sandbox_verdict":      report.sandbox_verdict,
            "completion_rate_pct":  round(report.completion_rate * 100, 1),
            "deadline_success_pct": round(report.deadline_success_rate * 100, 1),
            "avg_duration_hrs":     report.avg_total_hrs,
            "p50_hrs":              report.p50_hrs,
            "p90_hrs":              report.p90_hrs,
            "most_failed_step":     report.most_failed_step,
            "resource_bottleneck":  report.resource_bottleneck,
            "predicted_failures":   report.predicted_failure_points[:5],
            "pre_emptive_fixes":    report.pre_emptive_fixes,
            "trials_run":           len(report.trials),
            "outcome_distribution": {
                outcome: sum(1 for t in report.trials if t.final_outcome == outcome)
                for outcome in ["completed","deadline_missed","resource_exhausted","critical_failure","partial_completion"]
            },
        }


def _parse_steps(plan_text: str) -> List[str]:
    lines = plan_text.strip().split('\n')
    steps = []
    for line in lines:
        clean = re.sub(r'^\s*(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '', line, flags=re.IGNORECASE).strip()
        if clean and len(clean) > 5:
            steps.append(clean)
    return steps or [plan_text]


def _parse_json(raw: str) -> Dict:
    raw   = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
