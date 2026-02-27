# ═══════════════════════════════════════════════════════════════════════════════
# cross_plan_contradiction_detector.py — Feature 22: CPCD
# After all agents submit plans, detects semantic contradictions between them.
# Forces reconciliation before consensus — no conflicting instructions reach user.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

CPCD_MAX_CONTRADICTIONS_TO_RESOLVE = 8
CPCD_MIN_CONFLICT_CONFIDENCE       = 0.65    # below this → ignore as noise
CPCD_MAX_AGENTS_PER_CLUSTER        = 5       # max agents per side in a conflict


class ConflictType(Enum):
    APPROACH      = "approach"         # fundamentally different methodologies
    SEQUENCING    = "sequencing"       # step order contradictions
    RESOURCE      = "resource"         # incompatible resource allocations
    TECHNOLOGY    = "technology"       # conflicting tech stack choices
    TIMELINE      = "timeline"         # irreconcilable timeline claims
    SCOPE         = "scope"            # agents disagree on what's in scope
    ASSUMPTION    = "assumption"       # conflicting foundational assumptions


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Contradiction:
    contradiction_id:  str
    conflict_type:     ConflictType
    position_a:        str          # what some agents claim
    position_b:        str          # what other agents claim
    agents_for_a:      List[str]
    agents_for_b:      List[str]
    specific_steps:    List[str]    # which steps contain the conflict
    confidence:        float        # 0–1 certainty this is a real conflict
    severity:          str          # critical | major | minor
    reconcilable:      bool
    reconciliation_hint: str


@dataclass
class ReconciliationResult:
    contradiction_id:  str
    chosen_position:   str
    rationale:         str
    evidence_weight:   str          # "4 agents vs 2" or "higher-scored agent won"
    merged_step:       Optional[str]  # if steps were merged
    dropped_agents:    List[str]      # agents whose position was overruled


@dataclass
class CPCDResult:
    contradictions:      List[Contradiction]
    reconciliations:     List[ReconciliationResult]
    unresolvable:        List[Contradiction]
    reconciled_plans:    Dict[str, str]   # agent → de-conflicted plan
    conflict_summary:    str
    pre_conflict_count:  int
    post_conflict_count: int


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_CPCD_DETECTOR = """You are a SEMANTIC CONTRADICTION ANALYST.

AIM: {aim}

MULTI-AGENT PLAN SUBMISSIONS:
{plans_json}

Detect REAL semantic contradictions between agents — not style differences.
A contradiction exists when:
  • Agent A says "use microservices" and Agent B says "monolith is essential"
  • Agent A sequences market research BEFORE product design; Agent B says the opposite
  • Agent A allocates 2 developers; Agent B allocates 8 for the same task
  • Agent A assumes external funding; Agent B assumes bootstrap

For each contradiction:
  1. Identify which AGENTS hold each position
  2. Classify conflict type
  3. Rate severity and reconcilability

Ignore: word choice differences, emphasis differences, minor sequencing variations.

Respond ONLY with valid JSON:
{{
  "contradictions": [
    {{
      "conflict_type": "approach|sequencing|resource|technology|timeline|scope|assumption",
      "position_a": "...",
      "position_b": "...",
      "agents_for_a": ["gemini", "openai"],
      "agents_for_b": ["cohere"],
      "specific_steps": ["step_3", "step_5"],
      "confidence": 0.0,
      "severity": "critical|major|minor",
      "reconcilable": true,
      "reconciliation_hint": "..."
    }}
  ]
}}"""

PROMPT_CPCD_RECONCILER = """You are a CONFLICT RECONCILIATION SPECIALIST.

AIM: {aim}
CONTRADICTION TYPE: {conflict_type}

POSITION A (held by: {agents_a}, combined score: {score_a:.1f}):
"{position_a}"

POSITION B (held by: {agents_b}, combined score: {score_b:.1f}):
"{position_b}"

RECONCILIATION HINT: {hint}

Choose the BETTER position or synthesise a compromise.
Justify your choice with domain reasoning — not just majority vote.

Rules:
  1. Higher-scored agents get more weight but aren't automatically correct
  2. Prefer the more specific, evidence-based position
  3. If a compromise is possible and superior to both, propose it
  4. If irreconcilable, choose the safer/lower-risk position

Respond ONLY with valid JSON:
{{
  "chosen_position": "...",
  "position_chosen": "a|b|compromise",
  "rationale": "...",
  "evidence_weight": "...",
  "merged_step": null
}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class CrossPlanContradictionDetector:
    """
    Surfaces hidden conflicts between agent plans BEFORE consensus.

    Without CPCD: BFTCE might pick a "winner" plan that contradicts
    the knowledge from 8 other agents, causing downstream confusion.

    With CPCD: contradictions are surfaced → reconciled → consensus
    operates on a coherent, non-contradictory population.
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        agent:    str = "gemini",
    ):
        self.call_fn = call_fn
        self.agent   = agent

    # ── Detection ─────────────────────────────────────────────────────────────

    async def detect_contradictions(
        self,
        plans:  Dict[str, str],
        aim:    str,
    ) -> List[Contradiction]:
        """LLM compares all agent plans and identifies semantic conflicts."""
        # Condense plans for prompt (first 200 chars each)
        plans_summary = {
            agent: plan[:400]
            for agent, plan in list(plans.items())[:8]  # cap at 8 for prompt size
        }
        prompt = PROMPT_CPCD_DETECTOR.format(
            aim        = aim,
            plans_json = json.dumps(plans_summary, indent=2)[:3000],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            contradictions = []
            for i, c in enumerate(data.get("contradictions", [])):
                conf = float(c.get("confidence", 0.5))
                if conf < CPCD_MIN_CONFLICT_CONFIDENCE:
                    continue
                ctype = c.get("conflict_type", "approach").lower()
                try:
                    ct = ConflictType(ctype)
                except ValueError:
                    ct = ConflictType.APPROACH

                contradictions.append(Contradiction(
                    contradiction_id = f"conflict_{i+1}",
                    conflict_type    = ct,
                    position_a       = c.get("position_a", ""),
                    position_b       = c.get("position_b", ""),
                    agents_for_a     = c.get("agents_for_a", [])[:CPCD_MAX_AGENTS_PER_CLUSTER],
                    agents_for_b     = c.get("agents_for_b", [])[:CPCD_MAX_AGENTS_PER_CLUSTER],
                    specific_steps   = c.get("specific_steps", []),
                    confidence       = conf,
                    severity         = c.get("severity", "minor"),
                    reconcilable     = bool(c.get("reconcilable", True)),
                    reconciliation_hint = c.get("reconciliation_hint", ""),
                ))

            # Sort by severity
            sev_order = {"critical": 0, "major": 1, "minor": 2}
            contradictions.sort(key=lambda x: sev_order.get(x.severity, 3))
            logger.info(f"[CPCD] Detected {len(contradictions)} contradictions")
            return contradictions

        except Exception as e:
            logger.warning(f"[CPCD] Detection failed: {e}")
            return []

    # ── Reconciliation ────────────────────────────────────────────────────────

    async def reconcile(
        self,
        contradiction: Contradiction,
        all_scores:    Dict[str, float],
        aim:           str,
    ) -> ReconciliationResult:
        """Resolve one contradiction by choosing or synthesising a position."""
        score_a = sum(all_scores.get(a, 50.0) for a in contradiction.agents_for_a)
        score_b = sum(all_scores.get(a, 50.0) for a in contradiction.agents_for_b)

        prompt = PROMPT_CPCD_RECONCILER.format(
            aim          = aim,
            conflict_type= contradiction.conflict_type.value,
            agents_a     = ", ".join(contradiction.agents_for_a),
            score_a      = score_a,
            position_a   = contradiction.position_a[:300],
            agents_b     = ", ".join(contradiction.agents_for_b),
            score_b      = score_b,
            position_b   = contradiction.position_b[:300],
            hint         = contradiction.reconciliation_hint[:200],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            chosen = data.get("position_chosen", "a")
            return ReconciliationResult(
                contradiction_id = contradiction.contradiction_id,
                chosen_position  = data.get("chosen_position", contradiction.position_a),
                rationale        = data.get("rationale", ""),
                evidence_weight  = data.get("evidence_weight",
                    f"{len(contradiction.agents_for_a)} vs {len(contradiction.agents_for_b)} agents"),
                merged_step      = data.get("merged_step"),
                dropped_agents   = contradiction.agents_for_b if chosen == "a"
                                   else contradiction.agents_for_a,
            )
        except Exception as e:
            logger.warning(f"[CPCD] Reconciliation of {contradiction.contradiction_id} failed: {e}")
            # Heuristic: score-weighted choice
            if score_a >= score_b:
                chosen_pos    = contradiction.position_a
                dropped       = contradiction.agents_for_b
            else:
                chosen_pos    = contradiction.position_b
                dropped       = contradiction.agents_for_a
            return ReconciliationResult(
                contradiction_id = contradiction.contradiction_id,
                chosen_position  = chosen_pos,
                rationale        = f"Score-weighted: A={score_a:.0f} vs B={score_b:.0f}",
                evidence_weight  = f"Score-based: {score_a:.0f} vs {score_b:.0f}",
                merged_step      = None,
                dropped_agents   = dropped,
            )

    # ── Plan De-confliction ───────────────────────────────────────────────────

    def apply_reconciliations(
        self,
        plans:           Dict[str, str],
        reconciliations: List[ReconciliationResult],
        contradictions:  List[Contradiction],
    ) -> Dict[str, str]:
        """
        Apply reconciliation decisions to agent plans.
        Agents whose position was overruled get a note appended.
        """
        reconciled = dict(plans)
        conflict_map = {r.contradiction_id: r for r in reconciliations}

        for contradiction in contradictions:
            rec = conflict_map.get(contradiction.contradiction_id)
            if not rec:
                continue

            # Agents whose position was dropped get a correction note
            for agent in rec.dropped_agents:
                if agent in reconciled:
                    note = (
                        f"\n\n[CONTRADICTION RESOLVED — {contradiction.conflict_type.value.upper()}]\n"
                        f"Position overruled: {contradiction.position_b if agent in contradiction.agents_for_b else contradiction.position_a}\n"
                        f"Adopted position:   {rec.chosen_position}\n"
                        f"Rationale: {rec.rationale[:100]}"
                    )
                    reconciled[agent] = reconciled[agent] + note

        return reconciled

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def detect_and_reconcile(
        self,
        plans:      Dict[str, str],
        all_scores: Dict[str, float],
        aim:        str,
    ) -> CPCDResult:
        """Full CPCD pipeline."""
        pre_count = len(plans)

        contradictions = await self.detect_contradictions(plans, aim)
        if not contradictions:
            return CPCDResult(
                contradictions     = [],
                reconciliations    = [],
                unresolvable       = [],
                reconciled_plans   = plans,
                conflict_summary   = "No contradictions detected — population is coherent",
                pre_conflict_count = pre_count,
                post_conflict_count= pre_count,
            )

        # Reconcile critical and major only (limit total)
        to_reconcile  = [c for c in contradictions
                         if c.reconcilable and c.severity in ("critical", "major")]
        to_reconcile  = to_reconcile[:CPCD_MAX_CONTRADICTIONS_TO_RESOLVE]
        unresolvable  = [c for c in contradictions if not c.reconcilable]

        tasks = [
            self.reconcile(c, all_scores, aim)
            for c in to_reconcile
        ]
        reconciliations = await asyncio.gather(*tasks)

        reconciled_plans = self.apply_reconciliations(
            plans, list(reconciliations), to_reconcile
        )

        critical_count = sum(1 for c in contradictions if c.severity == "critical")
        summary = (
            f"Found {len(contradictions)} contradictions "
            f"({critical_count} critical). "
            f"Reconciled {len(reconciliations)}. "
            f"Unresolvable: {len(unresolvable)}."
        )

        logger.info(f"[CPCD] {summary}")
        return CPCDResult(
            contradictions     = contradictions,
            reconciliations    = list(reconciliations),
            unresolvable       = unresolvable,
            reconciled_plans   = reconciled_plans,
            conflict_summary   = summary,
            pre_conflict_count = pre_count,
            post_conflict_count= len(reconciled_plans),
        )

    def get_cpcd_report(self, result: CPCDResult) -> Dict:
        return {
            "contradictions_found":   len(result.contradictions),
            "reconciled":             len(result.reconciliations),
            "unresolvable":           len(result.unresolvable),
            "conflict_summary":       result.conflict_summary,
            "by_type": {
                ct.value: sum(1 for c in result.contradictions if c.conflict_type == ct)
                for ct in ConflictType
                if any(c.conflict_type == ct for c in result.contradictions)
            },
            "by_severity": {
                s: sum(1 for c in result.contradictions if c.severity == s)
                for s in ("critical", "major", "minor")
            },
            "reconciliation_details": [
                {
                    "id":       r.contradiction_id,
                    "position": r.chosen_position[:80],
                    "rationale":r.rationale[:100],
                    "overruled":r.dropped_agents,
                }
                for r in result.reconciliations
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
