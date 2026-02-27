# ═══════════════════════════════════════════════════════════════════════════════
# multi_agent_debate_protocol.py — Feature 7: MADP
# Adversarial 3-round debate that pressure-tests every plan.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

from system_config import (
    MADP_DEBATE_ROUNDS, MADP_TOKENS_PER_ROUND,
    MADP_MIN_CONCESSIONS_REQUIRED, MADP_SCORE_IMPROVEMENT_MIN
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class DebateRound:
    round_num:                int
    round_type:               str     # opening | cross_examination | rebuttal
    proponent_agent:          str
    skeptic_agent:            str
    proponent_argument:       str
    proponent_evidence:       List[str]     = field(default_factory=list)
    proponent_concessions:    List[str]     = field(default_factory=list)
    skeptic_argument:         str           = ""
    skeptic_attack_vectors:   List[str]     = field(default_factory=list)
    skeptic_concessions:      List[str]     = field(default_factory=list)
    specific_flaws:           List[Dict]    = field(default_factory=list)
    logical_fallacies:        List[str]     = field(default_factory=list)
    round_winner:             str           = "draw"   # proponent|skeptic|draw


@dataclass
class DebateResult:
    original_plan:        str
    revised_plan:         str
    debate_verdict:       str           # proponent_wins | skeptic_wins | draw
    plan_improvement_score: float
    rounds:               List[DebateRound]
    total_concessions:    int
    changes_made:         List[Dict]
    proponent_agent:      str
    skeptic_agent:        str
    synthesiser_agent:    str
    accepted:             bool          # False if improvement < MADP_SCORE_IMPROVEMENT_MIN


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_PROPONENT = """You are the PROPONENT in a structured plan debate.
Defend this plan VIGOROUSLY against all criticism.

AIM: {aim}
PLAN TO DEFEND:
{plan_text}

ROUND {round_num} — {round_type}
{prior_skeptic_argument}

Your DEFENCE must:
  1. Address every specific attack point (if any)
  2. Provide concrete evidence or analogies
  3. Pre-empt likely follow-up attacks
  4. Quantify claimed benefits where possible
  5. Concede ONLY points that are factually indefensible

Respond ONLY with valid JSON (no markdown):
{{"argument_text":"...","evidence_cited":["..."],"concessions":["..."],"counter_attacks":["..."],"confidence_score":0.0}}"""

PROMPT_SKEPTIC = """You are the SKEPTIC in a structured plan debate.
Find and expose every flaw, gap, and dangerous assumption.

AIM: {aim}
PLAN UNDER ATTACK:
{plan_text}

ROUND {round_num} — {round_type}
{prior_proponent_argument}

Your ATTACK must target substance only — not style:
  • Steps that assume unavailable resources
  • Timeline estimates that ignore dependencies
  • Missing risk mitigation for high-cascade steps
  • Technology assumptions unverified
  • Market/regulatory assumptions left open

Respond ONLY with valid JSON (no markdown):
{{"argument_text":"...","attack_vectors":["..."],"specific_flaws":[{{"step":"step_N","flaw":"...","severity":"critical|major|minor"}}],"concessions":["..."],"strongest_attack":"..."}}"""

PROMPT_SYNTHESISER = """You are the NEUTRAL SYNTHESISER extracting truth from both sides of a debate.

AIM: {aim}
ORIGINAL PLAN:
{original_plan}

FULL DEBATE TRANSCRIPT:
{transcript_json}

Build a SUPERIOR revised plan that:
  1. Incorporates ALL concessions from both sides
  2. Addresses every "critical" and "major" flaw raised by the Skeptic
  3. Retains all Proponent-defended strengths
  4. Adds missing steps if the debate revealed structural gaps
  5. Is decisive — adopt the better position, do not average

Respond ONLY with valid JSON (no markdown):
{{"revised_plan":"Step 1: ...\\nStep 2: ...","changes_made":[{{"type":"added|modified|removed","step":"step_N","reason":"..."}}],"debate_verdict":"proponent_wins|skeptic_wins|draw","plan_improvement_score":0.0,"synthesis_rationale":"..."}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class MultiAgentDebateProtocol:
    """
    3-round adversarial debate protocol.

    Round 1 — Opening:        each makes their strongest argument
    Round 2 — Cross-examination: directly challenges Round 1 claims
    Round 3 — Rebuttal:       final position incorporating cross-exam
    Synthesis — Neutral agent builds revised plan from all concessions
    """

    def __init__(
        self,
        call_fn:   Callable[[str, str], Awaitable[str]],
        score_fn:  Callable[[str, str], Awaitable[float]],
    ):
        """
        Args:
            call_fn:  async callable(agent, prompt) → str
            score_fn: async callable(plan_text, aim) → float
        """
        self.call_fn  = call_fn
        self.score_fn = score_fn
        self._transcript: List[DebateRound] = []

    # ── Role Assignment ───────────────────────────────────────────────────────

    def assign_debate_roles(
        self,
        agents:              List[str],
        credibility_scores:  Optional[Dict[str, float]] = None,
    ) -> Tuple[str, str, str]:
        """
        Proponent  = highest-credibility agent (best defender)
        Skeptic    = second-highest (strongest challenger)
        Synthesiser= third-highest or a fresh agent (neutral)

        Falls back to first three agents if no credibility data.
        """
        if credibility_scores and len(credibility_scores) >= 3:
            ranked = sorted(credibility_scores, key=credibility_scores.get, reverse=True)
            return ranked[0], ranked[1], ranked[2]

        # Default assignment
        if len(agents) >= 3:
            return agents[0], agents[1], agents[2]
        elif len(agents) == 2:
            return agents[0], agents[1], agents[0]
        else:
            return agents[0], agents[0], agents[0]

    # ── Individual Rounds ─────────────────────────────────────────────────────

    async def _run_proponent(
        self,
        agent:               str,
        plan_text:           str,
        aim:                 str,
        round_num:           int,
        round_type:          str,
        prior_skeptic_arg:   str = "",
    ) -> Tuple[str, List[str], List[str]]:
        """Returns (argument, evidence, concessions)."""
        prompt = PROMPT_PROPONENT.format(
            aim                   = aim,
            plan_text             = plan_text[:1800],
            round_num             = round_num,
            round_type            = round_type,
            prior_skeptic_argument= (
                f"SKEPTIC'S PREVIOUS ARGUMENT:\n{prior_skeptic_arg}"
                if prior_skeptic_arg else "(First round — no prior argument)"
            ),
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)
            return (
                data.get("argument_text", raw[:500]),
                data.get("evidence_cited", []),
                data.get("concessions", []),
            )
        except Exception as e:
            logger.warning(f"[MADP] Proponent {agent} R{round_num} failed: {e}")
            return ("I maintain the plan is sound as structured.", [], [])

    async def _run_skeptic(
        self,
        agent:              str,
        plan_text:          str,
        aim:                str,
        round_num:          int,
        round_type:         str,
        prior_proponent_arg:str = "",
    ) -> Tuple[str, List[str], List[str], List[Dict]]:
        """Returns (argument, attack_vectors, concessions, specific_flaws)."""
        prompt = PROMPT_SKEPTIC.format(
            aim                    = aim,
            plan_text              = plan_text[:1800],
            round_num              = round_num,
            round_type             = round_type,
            prior_proponent_argument= (
                f"PROPONENT'S PREVIOUS ARGUMENT:\n{prior_proponent_arg}"
                if prior_proponent_arg else "(First round — no prior argument)"
            ),
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)
            return (
                data.get("argument_text", raw[:500]),
                data.get("attack_vectors", []),
                data.get("concessions", []),
                data.get("specific_flaws", []),
            )
        except Exception as e:
            logger.warning(f"[MADP] Skeptic {agent} R{round_num} failed: {e}")
            return ("I find significant gaps in the dependency handling.", [], [], [])

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        synthesiser:   str,
        original_plan: str,
        aim:           str,
    ) -> Tuple[str, List[Dict], str, float]:
        """Returns (revised_plan, changes, verdict, improvement_score)."""
        # Build compact transcript for prompt
        transcript_data = []
        for r in self._transcript:
            transcript_data.append({
                "round":                 r.round_num,
                "type":                  r.round_type,
                "proponent_argument":    r.proponent_argument[:400],
                "proponent_concessions": r.proponent_concessions,
                "skeptic_argument":      r.skeptic_argument[:400],
                "skeptic_concessions":   r.skeptic_concessions,
                "specific_flaws":        r.specific_flaws[:5],
            })

        prompt = PROMPT_SYNTHESISER.format(
            aim            = aim,
            original_plan  = original_plan[:1500],
            transcript_json= json.dumps(transcript_data, indent=2)[:2000],
        )
        try:
            raw  = await self.call_fn(synthesiser, prompt)
            data = _parse_json(raw)
            return (
                data.get("revised_plan", original_plan),
                data.get("changes_made", []),
                data.get("debate_verdict", "draw"),
                float(data.get("plan_improvement_score", 0.0)),
            )
        except Exception as e:
            logger.warning(f"[MADP] Synthesis failed: {e}")
            return (original_plan, [], "draw", 0.0)

    # ── Master Orchestrator ───────────────────────────────────────────────────

    async def run_full_debate(
        self,
        agents:              List[str],
        best_plan:           str,
        aim:                 str,
        credibility_scores:  Optional[Dict[str, float]] = None,
    ) -> DebateResult:
        """
        Runs full 3-round debate + synthesis.
        Returns DebateResult with revised plan and improvement metrics.
        """
        self._transcript = []
        proponent, skeptic, synthesiser = self.assign_debate_roles(
            agents, credibility_scores
        )
        logger.info(f"[MADP] Debate: {proponent}(P) vs {skeptic}(S), {synthesiser}(N)")

        round_types = ["opening", "cross_examination", "rebuttal"]
        p_arg, s_arg = "", ""

        for rnum in range(1, MADP_DEBATE_ROUNDS + 1):
            rtype = round_types[min(rnum - 1, 2)]
            logger.info(f"[MADP] Round {rnum} — {rtype}")

            # Run both sides in parallel
            p_task = self._run_proponent(proponent, best_plan, aim, rnum, rtype, s_arg)
            s_task = self._run_skeptic(skeptic, best_plan, aim, rnum, rtype, p_arg)

            (p_arg, p_evidence, p_concessions), \
            (s_arg, s_vectors, s_concessions, s_flaws) = await asyncio.gather(p_task, s_task)

            dr = DebateRound(
                round_num             = rnum,
                round_type            = rtype,
                proponent_agent       = proponent,
                skeptic_agent         = skeptic,
                proponent_argument    = p_arg,
                proponent_evidence    = p_evidence,
                proponent_concessions = p_concessions,
                skeptic_argument      = s_arg,
                skeptic_attack_vectors= s_vectors,
                skeptic_concessions   = s_concessions,
                specific_flaws        = s_flaws,
                round_winner          = self._judge_round(p_arg, s_arg, s_flaws),
            )
            self._transcript.append(dr)

        # Synthesise
        revised_plan, changes, verdict, improvement = await self._synthesise(
            synthesiser, best_plan, aim
        )

        total_concessions = sum(
            len(r.proponent_concessions) + len(r.skeptic_concessions)
            for r in self._transcript
        )

        # Validate improvement with score_fn
        accepted = (
            improvement >= MADP_SCORE_IMPROVEMENT_MIN
            and total_concessions >= MADP_MIN_CONCESSIONS_REQUIRED
        )

        if not accepted:
            logger.info(
                f"[MADP] Debate result rejected: improvement={improvement:.1f} "
                f"concessions={total_concessions}. Keeping original plan."
            )
            revised_plan = best_plan

        result = DebateResult(
            original_plan         = best_plan,
            revised_plan          = revised_plan,
            debate_verdict        = verdict,
            plan_improvement_score= improvement,
            rounds                = self._transcript,
            total_concessions     = total_concessions,
            changes_made          = changes,
            proponent_agent       = proponent,
            skeptic_agent         = skeptic,
            synthesiser_agent     = synthesiser,
            accepted              = accepted,
        )
        logger.info(
            f"[MADP] Debate complete. Verdict={verdict} "
            f"improvement={improvement:.1f} accepted={accepted}"
        )
        return result

    def _judge_round(
        self,
        p_arg:   str,
        s_arg:   str,
        s_flaws: List[Dict],
    ) -> str:
        """
        Heuristic round judgement.
        Skeptic wins if they found critical flaws; proponent wins if they presented evidence.
        """
        critical_flaws = [f for f in s_flaws if f.get("severity") == "critical"]
        if critical_flaws:
            return "skeptic"
        if len(p_arg) > len(s_arg) * 1.3:
            return "proponent"
        return "draw"

    def get_transcript(self) -> List[Dict]:
        """Serialise transcript for logging/UI display."""
        return [
            {
                "round":                 r.round_num,
                "type":                  r.round_type,
                "proponent":             r.proponent_agent,
                "skeptic":               r.skeptic_agent,
                "proponent_argument":    r.proponent_argument[:300],
                "proponent_concessions": r.proponent_concessions,
                "skeptic_argument":      r.skeptic_argument[:300],
                "skeptic_attack_vectors":r.skeptic_attack_vectors,
                "specific_flaws":        r.specific_flaws,
                "round_winner":          r.round_winner,
            }
            for r in self._transcript
        ]

    def get_debate_report(self, result: DebateResult) -> Dict:
        """Structured report for Excel output."""
        rounds_won = {"proponent": 0, "skeptic": 0, "draw": 0}
        all_flaws  = []
        for r in result.rounds:
            rounds_won[r.round_winner] = rounds_won.get(r.round_winner, 0) + 1
            all_flaws.extend(r.specific_flaws)

        critical = [f for f in all_flaws if f.get("severity") == "critical"]
        return {
            "accepted":            result.accepted,
            "verdict":             result.debate_verdict,
            "improvement_score":   result.plan_improvement_score,
            "total_concessions":   result.total_concessions,
            "total_changes":       len(result.changes_made),
            "rounds_won":          rounds_won,
            "critical_flaws_found":len(critical),
            "proponent":           result.proponent_agent,
            "skeptic":             result.skeptic_agent,
            "synthesiser":         result.synthesiser_agent,
            "changes":             result.changes_made[:10],
        }


def _parse_json(raw: str) -> Dict:
    raw = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
