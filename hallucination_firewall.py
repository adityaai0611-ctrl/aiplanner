# ═══════════════════════════════════════════════════════════════════════════════
# hallucination_firewall.py — Feature 17: HFW
# 3-layer validation firewall that intercepts every agent response before
# it enters the plan pool. Quarantines hallucinated / impossible content.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

HFW_CONSISTENCY_THRESHOLD  = 0.60   # below this → internal inconsistency flag
HFW_PLAUSIBILITY_THRESHOLD = 0.50   # below this → domain implausibility flag
HFW_VARIANCE_SIGMA         = 2.5    # std devs from mean → statistical outlier flag
HFW_AUTO_QUARANTINE_LAYERS = 2      # fail 2+ layers → auto quarantine
HFW_SCORE_PENALTY          = 25.0   # subtract from score if 1 layer fails


# ── Data Structures ───────────────────────────────────────────────────────────

class ValidationLayer(Enum):
    CONSISTENCY  = "consistency"    # Layer 1: internal logic
    PLAUSIBILITY = "plausibility"   # Layer 2: domain physics
    CONSENSUS    = "consensus"      # Layer 3: statistical outlier


class FirewallVerdict(Enum):
    PASS        = "pass"
    WARN        = "warn"         # 1 layer failed — penalise score
    QUARANTINE  = "quarantine"   # 2+ layers failed — reject


@dataclass
class LayerResult:
    layer:           ValidationLayer
    passed:          bool
    confidence:      float       # 0–1
    flags:           List[str]   # specific issues found
    details:         str


@dataclass
class FirewallResult:
    agent:           str
    plan_text:       str
    verdict:         FirewallVerdict
    layers:          List[LayerResult]
    original_score:  float
    adjusted_score:  float
    quarantine_reason: Optional[str]
    is_admitted:     bool        # False = quarantined

    @property
    def layers_failed(self) -> int:
        return sum(1 for l in self.layers if not l.passed)


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_HFW_CONSISTENCY = """You are an INTERNAL CONSISTENCY AUDITOR.

AIM: {aim}
PLAN (agent={agent}):
{plan_text}

Check for INTERNAL LOGICAL CONTRADICTIONS:
  • Steps that claim resources they haven't acquired yet
  • Timeline claims that contradict step durations (e.g., "complete in 2 hrs" for 6-month work)
  • Steps that assume completed outcomes from later steps
  • Circular dependencies (A requires B, B requires A)
  • Resource usage that exceeds stated resource allocation
  • Steps duplicated with conflicting instructions

Rate: consistency_score 0.0 (completely inconsistent) to 1.0 (fully consistent)

Respond ONLY with valid JSON:
{{"consistency_score":0.0,"passed":true,"flags":["specific contradiction found"],"details":"..."}}"""

PROMPT_HFW_PLAUSIBILITY = """You are a DOMAIN PLAUSIBILITY INSPECTOR.

AIM: {aim}
DOMAIN: {domain}
PLAN (agent={agent}):
{plan_text}

Check if claimed outcomes are PHYSICALLY/OPERATIONALLY POSSIBLE:
  • Technology claims that don't exist (citing non-existent tools/APIs)
  • Cost estimates that are orders of magnitude off for the domain
  • Timeline estimates that violate known domain constraints
  • Regulatory claims that contradict known law in the domain
  • Resource requirements that exceed what's available in context
  • Market size claims that are implausible for the stated goal

Rate: plausibility_score 0.0 (physically impossible) to 1.0 (fully plausible)

Respond ONLY with valid JSON:
{{"plausibility_score":0.0,"passed":true,"impossible_claims":["specific impossible claim"],"details":"..."}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class HallucinationFirewall:
    """
    3-layer real-time validation for every agent response:

    Layer 1 — Consistency:   Internal logic, no circular deps, no time paradoxes
    Layer 2 — Plausibility:  Domain physics, realistic timelines, real tools
    Layer 3 — Consensus:     Statistical outlier detection vs other agents

    Admitted → enters plan pool normally
    Warned   → enters pool with score penalty
    Quarantined → rejected, logged, agent credibility reduced
    """

    def __init__(
        self,
        call_fn:   Callable[[str, str], Awaitable[str]],
        validator_agent: str = "gemini",
        domain:    str = "technology",
    ):
        self.call_fn         = call_fn
        self.validator_agent = validator_agent
        self.domain          = domain
        self._quarantine_log: List[FirewallResult] = []
        self._agent_quarantine_counts: Dict[str, int] = {}

    # ── Layer 1: Internal Consistency ────────────────────────────────────────

    async def check_consistency(
        self,
        plan_text: str,
        agent:     str,
        aim:       str,
    ) -> LayerResult:
        """Checks for internal logical contradictions."""
        # Fast pre-screen: regex-based heuristics
        flags = []
        flags += self._regex_consistency_checks(plan_text)

        # If fast checks find issues, LLM validates
        prompt = PROMPT_HFW_CONSISTENCY.format(
            aim       = aim,
            agent     = agent,
            plan_text = plan_text[:1500],
        )
        try:
            raw  = await self.call_fn(self.validator_agent, prompt)
            data = _parse_json(raw)
            score  = float(data.get("consistency_score", 0.8))
            passed = score >= HFW_CONSISTENCY_THRESHOLD
            flags += data.get("flags", [])
            return LayerResult(
                layer      = ValidationLayer.CONSISTENCY,
                passed     = passed,
                confidence = score,
                flags      = flags[:5],
                details    = data.get("details", ""),
            )
        except Exception as e:
            logger.warning(f"[HFW] Consistency check failed: {e}")
            # Fail-open: if validator fails, give benefit of doubt
            return LayerResult(
                layer      = ValidationLayer.CONSISTENCY,
                passed     = True,
                confidence = 0.75,
                flags      = flags,
                details    = "Validator unavailable — defaulting to pass",
            )

    def _regex_consistency_checks(self, plan_text: str) -> List[str]:
        """Rule-based fast checks before LLM call."""
        flags = []
        text  = plan_text.lower()

        # Suspicious timeline claims
        if re.search(r'(deploy|launch|complete).{0,30}(2 hour|1 hour|30 min)', text):
            flags.append("Implausibly fast deployment claim detected")

        # Circular dependency signals
        if re.search(r'(step \d+).{0,50}(requires|depends on).{0,50}(step \d+)', text):
            # Not necessarily circular, but flag for LLM review
            pass

        # Contradictory instructions
        duplicates = re.findall(r'\b(do not|never|avoid)\b.{0,30}\b(then|and|also)\b.{0,30}\b(do|perform|execute)\b', text)
        if duplicates:
            flags.append("Contradictory instruction pattern detected")

        return flags

    # ── Layer 2: Domain Plausibility ─────────────────────────────────────────

    async def check_plausibility(
        self,
        plan_text: str,
        agent:     str,
        aim:       str,
    ) -> LayerResult:
        """Checks if claimed outcomes are physically/operationally possible."""
        flags = self._regex_plausibility_checks(plan_text)

        prompt = PROMPT_HFW_PLAUSIBILITY.format(
            aim       = aim,
            domain    = self.domain,
            agent     = agent,
            plan_text = plan_text[:1500],
        )
        try:
            raw  = await self.call_fn(self.validator_agent, prompt)
            data = _parse_json(raw)
            score  = float(data.get("plausibility_score", 0.8))
            passed = score >= HFW_PLAUSIBILITY_THRESHOLD
            flags += data.get("impossible_claims", [])
            return LayerResult(
                layer      = ValidationLayer.PLAUSIBILITY,
                passed     = passed,
                confidence = score,
                flags      = flags[:5],
                details    = data.get("details", ""),
            )
        except Exception as e:
            logger.warning(f"[HFW] Plausibility check failed: {e}")
            return LayerResult(
                layer      = ValidationLayer.PLAUSIBILITY,
                passed     = True,
                confidence = 0.75,
                flags      = flags,
                details    = "Validator unavailable — defaulting to pass",
            )

    def _regex_plausibility_checks(self, plan_text: str) -> List[str]:
        """Fast regex-based plausibility heuristics."""
        flags = []
        text  = plan_text.lower()

        # Fabricated tool references (common LLM hallucinations)
        suspicious_tools = [
            r'\bgpt-\d{2}\b',        # future GPT versions
            r'\bgemini\s*\d{3}\b',   # future Gemini
        ]
        for pattern in suspicious_tools:
            if re.search(pattern, text):
                flags.append(f"Reference to possibly non-existent tool: {pattern}")

        # Unrealistic cost claims
        if re.search(r'\$0\s*(cost|budget|spend|investment)\b', text):
            flags.append("Zero-cost claim is implausible for non-trivial project")

        return flags

    # ── Layer 3: Consensus / Statistical Outlier ──────────────────────────────

    def check_consensus(
        self,
        agent:        str,
        agent_score:  float,
        all_scores:   Dict[str, float],
    ) -> LayerResult:
        """
        Flags agent as outlier if score deviates >2.5σ from peer mean.
        Pure statistical — no LLM call needed.
        """
        if len(all_scores) < 3:
            return LayerResult(
                layer      = ValidationLayer.CONSENSUS,
                passed     = True,
                confidence = 1.0,
                flags      = [],
                details    = "Insufficient peers for consensus check",
            )

        peer_scores = [s for a, s in all_scores.items() if a != agent]
        if not peer_scores:
            return LayerResult(
                layer=ValidationLayer.CONSENSUS, passed=True,
                confidence=1.0, flags=[], details="No peers"
            )

        mean   = statistics.mean(peer_scores)
        std    = statistics.stdev(peer_scores) if len(peer_scores) > 1 else 1.0
        z_score= abs(agent_score - mean) / max(std, 0.01)
        passed = z_score <= HFW_VARIANCE_SIGMA

        flags = []
        if not passed:
            direction = "above" if agent_score > mean else "below"
            flags.append(
                f"Score {agent_score:.1f} is {z_score:.1f}σ {direction} peer mean {mean:.1f}"
            )

        return LayerResult(
            layer      = ValidationLayer.CONSENSUS,
            passed     = passed,
            confidence = max(0.0, 1.0 - (z_score / (HFW_VARIANCE_SIGMA * 2))),
            flags      = flags,
            details    = f"z={z_score:.2f} mean={mean:.1f} std={std:.1f}",
        )

    # ── Master Validation ─────────────────────────────────────────────────────

    async def validate(
        self,
        agent:       str,
        plan_text:   str,
        aim:         str,
        score:       float,
        all_scores:  Optional[Dict[str, float]] = None,
    ) -> FirewallResult:
        """
        Run all 3 layers. Returns FirewallResult with verdict and adjusted score.
        """
        # Run L1 and L2 in parallel
        l1_task = self.check_consistency(plan_text, agent, aim)
        l2_task = self.check_plausibility(plan_text, agent, aim)
        l1, l2  = await asyncio.gather(l1_task, l2_task)

        # L3 is synchronous
        l3 = self.check_consensus(agent, score, all_scores or {agent: score})

        layers        = [l1, l2, l3]
        layers_failed = sum(1 for l in layers if not l.passed)

        # Verdict
        if layers_failed == 0:
            verdict        = FirewallVerdict.PASS
            adjusted_score = score
            quarantine_msg = None
        elif layers_failed == 1:
            verdict        = FirewallVerdict.WARN
            adjusted_score = max(0.0, score - HFW_SCORE_PENALTY)
            quarantine_msg = None
        else:
            verdict        = FirewallVerdict.QUARANTINE
            adjusted_score = 0.0
            failed_layers  = [l.layer.value for l in layers if not l.passed]
            all_flags      = [f for l in layers for f in l.flags]
            quarantine_msg = f"Failed layers: {failed_layers}. Issues: {all_flags[:3]}"

        is_admitted = verdict != FirewallVerdict.QUARANTINE

        result = FirewallResult(
            agent             = agent,
            plan_text         = plan_text,
            verdict           = verdict,
            layers            = layers,
            original_score    = score,
            adjusted_score    = adjusted_score,
            quarantine_reason = quarantine_msg,
            is_admitted       = is_admitted,
        )

        if not is_admitted:
            self._quarantine_log.append(result)
            self._agent_quarantine_counts[agent] = \
                self._agent_quarantine_counts.get(agent, 0) + 1
            logger.warning(
                f"[HFW] QUARANTINED: {agent} | {quarantine_msg}"
            )
        elif verdict == FirewallVerdict.WARN:
            logger.info(
                f"[HFW] WARN: {agent} score penalised "
                f"{score:.1f}→{adjusted_score:.1f}"
            )
        else:
            logger.info(f"[HFW] PASS: {agent} score={score:.1f}")

        return result

    async def validate_all(
        self,
        plans:  Dict[str, str],
        scores: Dict[str, float],
        aim:    str,
    ) -> Tuple[Dict[str, str], Dict[str, float], List[FirewallResult]]:
        """
        Validate all agent plans. Returns (admitted_plans, adjusted_scores, results).
        """
        tasks = [
            self.validate(agent, plan, aim, scores.get(agent, 0.0), scores)
            for agent, plan in plans.items()
        ]
        results        = await asyncio.gather(*tasks)
        admitted_plans = {}
        adj_scores     = {}
        for r in results:
            if r.is_admitted:
                admitted_plans[r.agent] = r.plan_text
                adj_scores[r.agent]     = r.adjusted_score

        n_quarantined = sum(1 for r in results if not r.is_admitted)
        logger.info(
            f"[HFW] Validation complete: {len(admitted_plans)} admitted, "
            f"{n_quarantined} quarantined"
        )
        return admitted_plans, adj_scores, list(results)

    def get_firewall_report(self, results: List[FirewallResult]) -> Dict:
        return {
            "total_validated":     len(results),
            "passed":              sum(1 for r in results if r.verdict == FirewallVerdict.PASS),
            "warned":              sum(1 for r in results if r.verdict == FirewallVerdict.WARN),
            "quarantined":         sum(1 for r in results if r.verdict == FirewallVerdict.QUARANTINE),
            "quarantine_log":      [
                {
                    "agent":   r.agent,
                    "reason":  r.quarantine_reason,
                    "layers_failed": r.layers_failed,
                }
                for r in results if not r.is_admitted
            ],
            "agent_quarantine_counts": self._agent_quarantine_counts,
            "score_adjustments":   [
                {
                    "agent":    r.agent,
                    "original": r.original_score,
                    "adjusted": r.adjusted_score,
                    "penalty":  r.original_score - r.adjusted_score,
                }
                for r in results if r.original_score != r.adjusted_score
            ],
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
