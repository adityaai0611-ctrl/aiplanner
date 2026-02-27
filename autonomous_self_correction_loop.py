# ═══════════════════════════════════════════════════════════════════════════════
# autonomous_self_correction_loop.py — Feature 10: ASCL
# Closed-loop feedback: monitor execution, detect deviations, auto-correct.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

from system_config import (
    ASCL_ESCALATION_THRESHOLD, ASCL_MICROPLAN_MAX_STEPS,
    ASCL_HEALTH_DECAY_PER_DEVIATION, ASCL_TIMELINE_SLIP_TRIGGER_PCT,
    ASCL_WEBHOOK_ENDPOINT
)

logger = logging.getLogger(__name__)


# ── Enums & Data Structures ───────────────────────────────────────────────────

class DeviationType(Enum):
    TIMELINE_SLIP       = "timeline_slip"
    RESOURCE_SHORTAGE   = "resource_shortage"
    SCOPE_CREEP         = "scope_creep"
    DEPENDENCY_FAILURE  = "dependency_failure"
    EXTERNAL_BLOCKER    = "external_blocker"
    STEP_ABANDONED      = "step_abandoned"


class StepStatus(Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    BLOCKED     = "blocked"
    FAILED      = "failed"
    MODIFIED    = "modified"
    SKIPPED     = "skipped"


@dataclass
class ExecutionReport:
    """Status update submitted by the executor for one plan step."""
    step_id:              str
    step_text:            str
    status:               StepStatus
    actual_duration_hrs:  Optional[float]  = None
    blocker_description:  Optional[str]    = None
    modifications_made:   Optional[str]    = None
    completion_pct:       float            = 0.0    # 0–100
    resources_used:       List[str]        = field(default_factory=list)
    timestamp:            str              = field(default_factory=lambda: datetime.utcnow().isoformat())
    reported_by:          str              = "system"


@dataclass
class DeviationRecord:
    """A detected deviation and its correction history."""
    deviation_id:         str
    step_id:              str
    deviation_type:       DeviationType
    root_cause:           str
    severity:             str             # critical | significant | minor
    delivery_impact_hrs:  float
    correction_urgency:   str             # immediate | within_24hrs | can_wait
    microplan:            List[str]       = field(default_factory=list)
    correction_applied:   bool            = False
    correction_succeeded: Optional[bool]  = None
    attempts:             int             = 0
    timestamp:            str             = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CorrectionResult:
    deviation:            DeviationRecord
    microplan:            List[str]
    insert_before_step:   Optional[str]
    estimated_recovery_hrs: float
    correction_confidence:str
    if_correction_fails:  str   # escalate | try_alternative | abandon_step
    validated:            bool  = False


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_ASCL_DIAGNOSER = """You are an EXECUTION DEVIATION ANALYST.

ORIGINAL AIM: {aim}
ORIGINAL PLAN:
{original_plan}

EXECUTION REPORT:
  Step: "{step_text}"
  Status: {status}
  Actual Duration: {actual_hrs}hrs (forecast was {forecast_hrs}hrs)
  Blocker: {blocker_description}
  Modifications Made: {modifications_made}
  Completion %: {completion_pct}%

Classify this deviation and diagnose root cause.

Respond ONLY with valid JSON (no markdown):
{{"deviation_type":"timeline_slip|resource_shortage|scope_creep|dependency_failure|external_blocker|step_abandoned","root_cause":"...","severity":"critical|significant|minor","downstream_steps_at_risk":["step_N"],"delivery_date_impact_hrs":0.0,"correction_urgency":"immediate|within_24hrs|can_wait","recommended_correction_approach":"..."}}"""

PROMPT_ASCL_MICROPLAN = """You are an AUTONOMOUS CORRECTION ENGINE.

ORIGINAL AIM: {aim}
DEVIATION TYPE: {deviation_type}
ROOT CAUSE: {root_cause}
AFFECTED STEP: "{affected_step}"
REMAINING PLAN:
{remaining_steps}

AVAILABLE RESOURCES: {available_resources}
DEADLINE REMAINING: {deadline_hrs}hrs

Generate a CORRECTIVE MICRO-PLAN of 2–5 steps that:
  1. Directly resolves the root cause
  2. Does NOT repeat the failed approach
  3. Restores the original plan's momentum
  4. Stays within resources and deadline

Respond ONLY with valid JSON (no markdown):
{{"correction_steps":[{{"step_id":"correction_1","action":"...","targets_root_cause":true,"estimated_hrs":0.0,"resources_needed":["..."],"success_indicator":"..."}}],"insert_before_step":"step_N","estimated_recovery_hrs":0.0,"correction_confidence":"high|medium|low","if_correction_fails":"escalate|try_alternative|abandon_step"}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class AutonomousSelfCorrectionLoop:
    """
    Receives execution reports post-delivery, detects deviations,
    diagnoses root causes, and generates corrective micro-plans.

    Usage:
        ascl = AutonomousSelfCorrectionLoop(call_fn, original_plan, temporal_plan)
        report = ExecutionReport(step_id="step_3", status=StepStatus.BLOCKED, ...)
        correction = await ascl.process_report(report, aim)
    """

    def __init__(
        self,
        call_fn:        Callable[[str, str], Awaitable[str]],
        original_plan:  str,
        temporal_plan:  Optional[Dict] = None,   # TES SimulationReport dict
        agent:          str = "gemini",
        available_resources: Optional[List[str]] = None,
        total_deadline_hrs:  Optional[float] = None,
    ):
        self.call_fn             = call_fn
        self.original_plan       = original_plan
        self.temporal_plan       = temporal_plan or {}
        self.agent               = agent
        self.available_resources = available_resources or ["team", "budget", "tools"]
        self.total_deadline_hrs  = total_deadline_hrs or 168.0  # 1 week default

        self._deviations:   List[DeviationRecord] = []
        self._reports:      List[ExecutionReport]  = []
        self._health_score: float                  = 100.0
        self._start_time:   float                  = time.time()
        self._escalated:    bool                   = False

    # ── Report Processing ─────────────────────────────────────────────────────

    async def process_report(
        self,
        report: ExecutionReport,
        aim:    str,
    ) -> Optional[CorrectionResult]:
        """
        Main entry point. Receives one ExecutionReport.
        Returns CorrectionResult if a deviation is detected, else None.
        """
        self._reports.append(report)
        logger.info(f"[ASCL] Report received: {report.step_id} → {report.status.value}")

        deviation_type = self.detect_deviation(report)
        if not deviation_type:
            logger.info(f"[ASCL] {report.step_id}: no deviation detected")
            return None

        logger.warning(f"[ASCL] Deviation detected: {deviation_type.value} on {report.step_id}")

        # Diagnose
        diagnosis = await self.diagnose_root_cause(report, aim)

        # Update health score
        severity_decay = {
            "critical":    ASCL_HEALTH_DECAY_PER_DEVIATION * 2,
            "significant": ASCL_HEALTH_DECAY_PER_DEVIATION,
            "minor":       ASCL_HEALTH_DECAY_PER_DEVIATION * 0.5,
        }
        self._health_score = max(
            0.0,
            self._health_score - severity_decay.get(diagnosis.get("severity", "minor"), ASCL_HEALTH_DECAY_PER_DEVIATION)
        )

        # Build deviation record
        dev = DeviationRecord(
            deviation_id        = f"dev_{len(self._deviations)+1}",
            step_id             = report.step_id,
            deviation_type      = deviation_type,
            root_cause          = diagnosis.get("root_cause", "Unknown"),
            severity            = diagnosis.get("severity", "minor"),
            delivery_impact_hrs = float(diagnosis.get("delivery_date_impact_hrs", 0.0)),
            correction_urgency  = diagnosis.get("correction_urgency", "can_wait"),
        )
        self._deviations.append(dev)

        # Check escalation
        should_esc, esc_reason = self.should_escalate()
        if should_esc:
            self._escalated = True
            logger.error(f"[ASCL] ESCALATING: {esc_reason}")
            return None

        # Generate micro-plan
        correction = await self.generate_corrective_microplan(dev, report, aim)
        return correction

    # ── Deviation Detection ───────────────────────────────────────────────────

    def detect_deviation(
        self,
        report: ExecutionReport,
    ) -> Optional[DeviationType]:
        """
        Rule-based deviation detection using report fields + TES forecast.
        Returns DeviationType or None if execution is nominal.
        """
        if report.status in (StepStatus.FAILED, StepStatus.STEP_ABANDONED if hasattr(StepStatus, 'STEP_ABANDONED') else StepStatus.SKIPPED):
            return DeviationType.STEP_ABANDONED

        if report.status == StepStatus.BLOCKED:
            blocker = (report.blocker_description or "").lower()
            if any(kw in blocker for kw in ["budget", "resource", "personnel", "tool", "access"]):
                return DeviationType.RESOURCE_SHORTAGE
            if any(kw in blocker for kw in ["dependency", "prerequisite", "waiting", "blocked by"]):
                return DeviationType.DEPENDENCY_FAILURE
            if any(kw in blocker for kw in ["external", "vendor", "regulatory", "approval", "weather"]):
                return DeviationType.EXTERNAL_BLOCKER
            return DeviationType.DEPENDENCY_FAILURE

        if report.status == StepStatus.MODIFIED:
            return DeviationType.SCOPE_CREEP

        if report.status == StepStatus.COMPLETED and report.actual_duration_hrs is not None:
            # Check against TES forecast
            forecast = self._get_forecast_hrs(report.step_id)
            if forecast and report.actual_duration_hrs > forecast * ASCL_TIMELINE_SLIP_TRIGGER_PCT:
                return DeviationType.TIMELINE_SLIP

        return None

    def _get_forecast_hrs(self, step_id: str) -> Optional[float]:
        """Look up P50 forecast from TES temporal plan."""
        gantt = self.temporal_plan.get("gantt_data", [])
        for entry in gantt:
            if entry.get("step_id") == step_id:
                return entry.get("duration_p50_hrs")
        return None

    # ── Root Cause Diagnosis ──────────────────────────────────────────────────

    async def diagnose_root_cause(
        self,
        report: ExecutionReport,
        aim:    str,
    ) -> Dict:
        """LLM diagnoses why this deviation occurred and its downstream impact."""
        forecast = self._get_forecast_hrs(report.step_id) or 0.0
        prompt   = PROMPT_ASCL_DIAGNOSER.format(
            aim                = aim,
            original_plan      = self.original_plan[:1000],
            step_text          = report.step_text[:200],
            status             = report.status.value,
            actual_hrs         = report.actual_duration_hrs or "N/A",
            forecast_hrs       = f"{forecast:.1f}",
            blocker_description= report.blocker_description or "None",
            modifications_made = report.modifications_made or "None",
            completion_pct     = report.completion_pct,
        )
        try:
            raw = await self.call_fn(self.agent, prompt)
            return _parse_json(raw)
        except Exception as e:
            logger.warning(f"[ASCL] Diagnosis failed: {e}")
            return {
                "root_cause":             "Undiagnosed deviation",
                "severity":               "significant",
                "delivery_date_impact_hrs": 4.0,
                "correction_urgency":     "within_24hrs",
            }

    # ── Micro-Plan Generation ─────────────────────────────────────────────────

    async def generate_corrective_microplan(
        self,
        deviation: DeviationRecord,
        report:    ExecutionReport,
        aim:       str,
    ) -> CorrectionResult:
        """Generates 2–5 targeted correction steps."""
        remaining  = self._get_remaining_steps(report.step_id)
        elapsed    = (time.time() - self._start_time) / 3600
        deadline   = max(0.0, self.total_deadline_hrs - elapsed)

        prompt = PROMPT_ASCL_MICROPLAN.format(
            aim                = aim,
            deviation_type     = deviation.deviation_type.value,
            root_cause         = deviation.root_cause,
            affected_step      = report.step_text[:200],
            remaining_steps    = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(remaining[:8])),
            available_resources= ", ".join(self.available_resources),
            deadline_hrs       = f"{deadline:.1f}",
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            steps = [cs.get("action", "") for cs in data.get("correction_steps", [])]
            steps = [s for s in steps if s][:ASCL_MICROPLAN_MAX_STEPS]

            if not steps:
                steps = [f"Re-evaluate root cause: {deviation.root_cause}",
                         "Implement targeted fix and validate outcome"]

            deviation.microplan        = steps
            deviation.correction_applied = True
            deviation.attempts         += 1

            result = CorrectionResult(
                deviation               = deviation,
                microplan               = steps,
                insert_before_step      = data.get("insert_before_step"),
                estimated_recovery_hrs  = float(data.get("estimated_recovery_hrs", 4.0)),
                correction_confidence   = data.get("correction_confidence", "medium"),
                if_correction_fails     = data.get("if_correction_fails", "escalate"),
            )
            logger.info(
                f"[ASCL] Micro-plan generated: {len(steps)} steps, "
                f"recovery={result.estimated_recovery_hrs:.1f}h, "
                f"confidence={result.correction_confidence}"
            )
            return result

        except Exception as e:
            logger.warning(f"[ASCL] Micro-plan generation failed: {e}")
            fallback = [
                f"Investigate root cause: {deviation.root_cause}",
                "Implement corrective action based on investigation findings",
                "Validate correction and resume original plan from next step",
            ]
            deviation.microplan        = fallback
            deviation.correction_applied = True
            deviation.attempts         += 1
            return CorrectionResult(
                deviation             = deviation,
                microplan             = fallback,
                insert_before_step    = None,
                estimated_recovery_hrs= 8.0,
                correction_confidence = "low",
                if_correction_fails   = "escalate",
            )

    def record_correction_outcome(
        self,
        deviation_id: str,
        succeeded:    bool,
    ) -> None:
        """Called by executor to report whether the micro-plan resolved the issue."""
        for dev in self._deviations:
            if dev.deviation_id == deviation_id:
                dev.correction_succeeded = succeeded
                logger.info(f"[ASCL] Correction {deviation_id} outcome: {'✅' if succeeded else '❌'}")
                return

    # ── Escalation ────────────────────────────────────────────────────────────

    def should_escalate(self) -> Tuple[bool, str]:
        """
        Escalate if:
        - 3+ consecutive corrections failed
        - Health score < 20
        - Critical deviation with no recovery plan
        """
        if self._escalated:
            return True, "Already escalated"

        recent_failures = 0
        for dev in reversed(self._deviations):
            if dev.correction_succeeded is False:
                recent_failures += 1
            elif dev.correction_succeeded is True:
                break   # reset streak
            if recent_failures >= ASCL_ESCALATION_THRESHOLD:
                return True, f"{recent_failures} consecutive correction failures"

        if self._health_score < 20.0:
            return True, f"Plan health critically low: {self._health_score:.0f}/100"

        critical_unresolved = [
            d for d in self._deviations
            if d.severity == "critical" and d.correction_succeeded is None
        ]
        if len(critical_unresolved) >= 2:
            return True, f"{len(critical_unresolved)} unresolved critical deviations"

        return False, ""

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_remaining_steps(self, current_step_id: str) -> List[str]:
        """Extract steps after current_step_id from original plan."""
        lines  = self.original_plan.strip().split('\n')
        steps  = []
        found  = False
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            if current_step_id in clean or (not found and steps):
                found = True
                continue
            if found:
                steps.append(re.sub(r'^(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '', clean, flags=re.IGNORECASE))
        return steps

    def get_plan_health_score(self) -> float:
        """0–100: decays with unresolved deviations."""
        return round(self._health_score, 1)

    def get_correction_history(self) -> List[Dict]:
        """Full audit trail of all deviations and corrections."""
        return [
            {
                "deviation_id":         d.deviation_id,
                "step_id":              d.step_id,
                "deviation_type":       d.deviation_type.value,
                "severity":             d.severity,
                "root_cause":           d.root_cause,
                "delivery_impact_hrs":  d.delivery_impact_hrs,
                "correction_urgency":   d.correction_urgency,
                "microplan":            d.microplan,
                "correction_applied":   d.correction_applied,
                "correction_succeeded": d.correction_succeeded,
                "attempts":             d.attempts,
                "timestamp":            d.timestamp,
            }
            for d in self._deviations
        ]

    def get_ascl_report(self) -> Dict:
        """Summary report for Excel/frontend output."""
        total = len(self._deviations)
        resolved = sum(1 for d in self._deviations if d.correction_succeeded is True)
        failed   = sum(1 for d in self._deviations if d.correction_succeeded is False)
        pending  = total - resolved - failed
        by_type  = {}
        for d in self._deviations:
            k = d.deviation_type.value
            by_type[k] = by_type.get(k, 0) + 1

        return {
            "plan_health_score":         self._health_score,
            "total_deviations":          total,
            "resolved_deviations":       resolved,
            "failed_corrections":        failed,
            "pending_deviations":        pending,
            "escalated":                 self._escalated,
            "total_reports_received":    len(self._reports),
            "deviation_breakdown":       by_type,
            "total_impact_hrs":          sum(d.delivery_impact_hrs for d in self._deviations),
            "correction_success_rate":   round(resolved / max(total, 1) * 100, 1),
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
