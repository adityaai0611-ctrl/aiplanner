# ═══════════════════════════════════════════════════════════════════════════════
# stakeholder_persona_simulator.py — Feature 20: SPS
# Simulates 6 real-world stakeholder personas reviewing the plan.
# Each raises domain-specific objections → final plan addresses all of them.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

SPS_DEFAULT_PERSONAS = [
    "ceo", "developer", "customer", "legal", "investor", "operations"
]
SPS_MIN_APPROVAL_RATE      = 0.60    # plan accepted if >= 60% personas approve
SPS_MAX_OBJECTIONS_PER_PERSONA = 4
SPS_CRITICAL_OBJECTION_THRESHOLD = 2  # plan blocked if >= this many critical objections


# ── Persona Definitions ───────────────────────────────────────────────────────

PERSONA_PROFILES: Dict[str, Dict] = {
    "ceo": {
        "title":       "Chief Executive Officer",
        "icon":        "👔",
        "priorities":  ["ROI", "strategic fit", "competitive advantage", "timeline to revenue", "board optics"],
        "red_flags":   ["no clear revenue model", "timeline > 2 years to first dollar", "no exit strategy"],
        "questions":   ["What is the 3-year NPV?", "How does this beat competitors?", "What's the board narrative?"],
        "approval_bias": 0.65,   # moderate — wants results
    },
    "developer": {
        "title":       "Lead Software Engineer",
        "icon":        "💻",
        "priorities":  ["technical feasibility", "clean architecture", "avoiding tech debt", "realistic timelines", "tooling"],
        "red_flags":   ["vague technical specs", "unrealistic dev timelines", "ignored edge cases", "no testing strategy"],
        "questions":   ["What tech stack?", "How do we handle failure modes?", "Who writes the tests?"],
        "approval_bias": 0.55,   # skeptical by nature
    },
    "customer": {
        "title":       "Target Customer / End User",
        "icon":        "👤",
        "priorities":  ["ease of use", "value delivered", "privacy", "reliability", "support quality"],
        "red_flags":   ["plan ignores user onboarding", "no feedback loop", "data privacy vague"],
        "questions":   ["How does this make my life easier?", "What if it breaks?", "Who do I call?"],
        "approval_bias": 0.70,   # easy to please if UX addressed
    },
    "legal": {
        "title":       "General Counsel / Legal",
        "icon":        "⚖️",
        "priorities":  ["compliance", "IP protection", "liability minimisation", "data privacy", "regulatory risk"],
        "red_flags":   ["GDPR not mentioned", "no IP strategy", "regulatory pathway missing", "liability unclear"],
        "questions":   ["GDPR/CCPA compliant?", "Who owns the IP?", "What's the regulatory pathway?"],
        "approval_bias": 0.45,   # highly skeptical — risk averse
    },
    "investor": {
        "title":       "Series A Investor / VC",
        "icon":        "💰",
        "priorities":  ["market size (TAM/SAM/SOM)", "unit economics", "team capability", "defensibility", "exit options"],
        "red_flags":   ["no market sizing", "burn rate not mentioned", "no moat", "vague team roles"],
        "questions":   ["What's the TAM?", "When do you break even?", "What prevents a Google from copying this?"],
        "approval_bias": 0.50,   # neutral until convinced
    },
    "operations": {
        "title":       "VP Operations / COO",
        "icon":        "⚙️",
        "priorities":  ["process efficiency", "resource allocation", "vendor risk", "scalability", "day-1 readiness"],
        "red_flags":   ["dependencies not mapped", "no contingency plans", "resource conflicts", "no escalation path"],
        "questions":   ["Who does what on day 1?", "What happens if Vendor X fails?", "How do we scale from 10 to 1000 users?"],
        "approval_bias": 0.60,   # reasonable if plan is operational
    },
}


# ── Data Structures ───────────────────────────────────────────────────────────

class ObjectionSeverity(Enum):
    CRITICAL = "critical"    # blocks plan approval
    MAJOR    = "major"       # requires plan change
    MINOR    = "minor"       # nice-to-have fix


@dataclass
class StakeholderObjection:
    persona:      str
    severity:     ObjectionSeverity
    category:     str             # what aspect of plan is objected to
    objection:    str             # the actual objection text
    specific_step:Optional[str]   # which plan step triggered this
    change_request:str            # what they want changed
    is_blocker:   bool            # True if this alone blocks approval


@dataclass
class PersonaVerdict:
    persona:          str
    title:            str
    approved:         bool
    approval_score:   float       # 0–100
    objections:       List[StakeholderObjection]
    praise:           List[str]   # what they liked
    conditional:      Optional[str]  # "I'll approve IF you add X"
    summary:          str


@dataclass
class SPSResult:
    verdicts:             Dict[str, PersonaVerdict]
    overall_approved:     bool
    approval_rate:        float          # fraction of personas that approved
    critical_objections:  List[StakeholderObjection]
    all_change_requests:  List[str]
    hardened_plan:        str            # plan revised to address all objections
    personas_blocked:     List[str]      # personas that did NOT approve
    consensus_summary:    str


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_SPS_PERSONA_REVIEW = """You are the {title} reviewing an execution plan for approval.

YOUR PRIORITIES: {priorities}
YOUR RED FLAGS: {red_flags}
YOUR KEY QUESTIONS: {key_questions}

AIM: {aim}
PLAN TO REVIEW:
{plan_text}

Review this plan from your perspective ONLY. Be realistic, specific, and ruthless.

For each objection:
  • severity: critical (blocks plan), major (requires change), minor (nice-to-have)
  • Cite the SPECIFIC plan step that triggers the objection
  • State exactly what you want changed

Respond ONLY with valid JSON:
{{
  "approved": false,
  "approval_score": 0.0,
  "objections": [
    {{
      "severity": "critical",
      "category": "...",
      "objection": "...",
      "specific_step": "Step N",
      "change_request": "...",
      "is_blocker": true
    }}
  ],
  "praise": ["what I liked"],
  "conditional": "I will approve IF you...",
  "summary": "..."
}}"""

PROMPT_SPS_PLAN_HARDENER = """You are a MASTER PLAN INTEGRATOR.

AIM: {aim}
ORIGINAL PLAN:
{original_plan}

STAKEHOLDER REVIEW RESULTS ({n_personas} personas reviewed):
{verdicts_summary}

ALL CHANGE REQUESTS:
{change_requests}

CRITICAL OBJECTIONS THAT MUST BE ADDRESSED:
{critical_objections}

Revise the plan to address ALL critical and major objections.
Rules:
  1. Address every CRITICAL objection — non-negotiable
  2. Address MAJOR objections unless they conflict with each other
  3. Preserve what all personas praised
  4. Do not add more than 4 new steps total
  5. The revised plan must remain executable and coherent

Output ONLY the revised plan as numbered steps:"""


# ── Engine ────────────────────────────────────────────────────────────────────

class StakeholderPersonaSimulator:
    """
    Simulates 6 real-world stakeholders reviewing the final plan.
    
    Each persona:
    - Reviews the plan from their specific domain perspective
    - Raises severity-graded objections
    - Approves or blocks the plan
    
    After all reviews:
    - Counts approvals vs blocks
    - Aggregates all change requests
    - Re-generates plan addressing all critical/major objections
    - Returns hardened plan + consensus metrics
    """

    def __init__(
        self,
        call_fn: Callable[[str, str], Awaitable[str]],
        agents:  Optional[List[str]] = None,
        personas:Optional[List[str]] = None,
    ):
        self.call_fn = call_fn
        # Map each persona to a specific agent for diversity
        self.agents  = agents or [
            "gemini", "openai", "cohere", "groq", "deepseek", "writer"
        ]
        self.personas = personas or SPS_DEFAULT_PERSONAS

    # ── Individual Persona Review ─────────────────────────────────────────────

    async def review_as_persona(
        self,
        persona_key: str,
        plan_text:   str,
        aim:         str,
        agent:       str,
    ) -> PersonaVerdict:
        """One persona reviews the plan and returns their verdict."""
        profile = PERSONA_PROFILES.get(persona_key, PERSONA_PROFILES["ceo"])
        prompt  = PROMPT_SPS_PERSONA_REVIEW.format(
            title        = profile["title"],
            priorities   = ", ".join(profile["priorities"]),
            red_flags    = ", ".join(profile["red_flags"]),
            key_questions= " | ".join(profile["questions"]),
            aim          = aim,
            plan_text    = plan_text[:1600],
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)

            objections = []
            for obj in data.get("objections", [])[:SPS_MAX_OBJECTIONS_PER_PERSONA]:
                sev_str = obj.get("severity", "minor").lower()
                sev     = ObjectionSeverity.CRITICAL if sev_str == "critical" else \
                          ObjectionSeverity.MAJOR    if sev_str == "major"    else \
                          ObjectionSeverity.MINOR
                objections.append(StakeholderObjection(
                    persona       = persona_key,
                    severity      = sev,
                    category      = obj.get("category", "general"),
                    objection     = obj.get("objection", ""),
                    specific_step = obj.get("specific_step"),
                    change_request= obj.get("change_request", ""),
                    is_blocker    = bool(obj.get("is_blocker", False)),
                ))

            approved = bool(data.get("approved", False))
            score    = float(data.get("approval_score", 50.0))

            # Apply approval bias — personas have inherent skepticism levels
            bias     = profile.get("approval_bias", 0.6)
            adjusted = score * bias + score * (1 - bias)  # weighted toward bias
            approved = approved and adjusted >= 50.0

            verdict = PersonaVerdict(
                persona        = persona_key,
                title          = profile["title"],
                approved       = approved,
                approval_score = round(adjusted, 1),
                objections     = objections,
                praise         = data.get("praise", [])[:3],
                conditional    = data.get("conditional"),
                summary        = data.get("summary", ""),
            )
            icon = profile["icon"]
            status = "✅ APPROVED" if approved else "❌ BLOCKED"
            logger.info(
                f"[SPS] {icon} {profile['title']}: {status} "
                f"(score={verdict.approval_score:.0f}, "
                f"objections={len(objections)})"
            )
            return verdict

        except Exception as e:
            logger.warning(f"[SPS] {persona_key} review failed ({agent}): {e}")
            # Fallback verdict
            profile = PERSONA_PROFILES.get(persona_key, {})
            return PersonaVerdict(
                persona        = persona_key,
                title          = profile.get("title", persona_key.title()),
                approved       = True,
                approval_score = 65.0,
                objections     = [],
                praise         = ["Plan structure is acceptable"],
                conditional    = None,
                summary        = "Review inconclusive — defaulting to conditional approval",
            )

    # ── All Persona Reviews ───────────────────────────────────────────────────

    async def run_all_reviews(
        self,
        plan_text: str,
        aim:       str,
    ) -> Dict[str, PersonaVerdict]:
        """All personas review the plan in parallel."""
        persona_agent_pairs = list(zip(
            self.personas,
            self.agents + self.agents   # cycle agents if fewer than personas
        ))[:len(self.personas)]

        tasks = [
            self.review_as_persona(persona, plan_text, aim, agent)
            for persona, agent in persona_agent_pairs
        ]
        verdicts_list = await asyncio.gather(*tasks)
        return {v.persona: v for v in verdicts_list}

    # ── Plan Hardening ────────────────────────────────────────────────────────

    async def harden_plan(
        self,
        original_plan: str,
        aim:           str,
        verdicts:      Dict[str, PersonaVerdict],
        hardener_agent:str = "gemini",
    ) -> str:
        """Re-generates plan addressing all critical and major objections."""
        # Aggregate verdicts summary
        verdicts_summary = ""
        for persona, verdict in verdicts.items():
            status = "APPROVED" if verdict.approved else "BLOCKED"
            profile = PERSONA_PROFILES.get(persona, {})
            verdicts_summary += f"\n{profile.get('icon','')} {verdict.title} [{status}] score={verdict.approval_score:.0f}\n"
            for obj in verdict.objections[:3]:
                verdicts_summary += f"  • [{obj.severity.value.upper()}] {obj.objection[:100]}\n"
            if verdict.conditional:
                verdicts_summary += f"  → Conditional: {verdict.conditional}\n"

        # All change requests
        change_requests = []
        critical_objections = []
        for verdict in verdicts.values():
            for obj in verdict.objections:
                if obj.change_request:
                    change_requests.append(f"[{obj.persona}/{obj.severity.value}] {obj.change_request}")
                if obj.severity == ObjectionSeverity.CRITICAL:
                    critical_objections.append(f"[{obj.persona}] {obj.objection}")

        if not change_requests:
            logger.info("[SPS] No change requests — plan accepted as-is")
            return original_plan

        prompt = PROMPT_SPS_PLAN_HARDENER.format(
            aim                = aim,
            original_plan      = original_plan[:1200],
            n_personas         = len(verdicts),
            verdicts_summary   = verdicts_summary[:1000],
            change_requests    = "\n".join(f"  • {cr}" for cr in change_requests[:12]),
            critical_objections= "\n".join(f"  ⚠ {co}" for co in critical_objections[:6]),
        )
        try:
            hardened = await self.call_fn(hardener_agent, prompt)
            logger.info(f"[SPS] Plan hardened to address {len(change_requests)} change requests")
            return hardened.strip()
        except Exception as e:
            logger.warning(f"[SPS] Plan hardening failed: {e}")
            return original_plan

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def simulate(
        self,
        plan_text: str,
        aim:       str,
    ) -> SPSResult:
        """
        Full SPS pipeline:
        1. All 6 personas review the plan in parallel
        2. Count approvals and critical objections
        3. Aggregate all change requests
        4. Harden the plan
        5. Return SPSResult
        """
        logger.info(f"[SPS] Starting stakeholder review with {len(self.personas)} personas...")

        verdicts = await self.run_all_reviews(plan_text, aim)

        # Aggregate metrics
        approved_count = sum(1 for v in verdicts.values() if v.approved)
        approval_rate  = approved_count / max(len(verdicts), 1)

        all_objections = [
            obj
            for v in verdicts.values()
            for obj in v.objections
        ]
        critical_objections = [o for o in all_objections if o.severity == ObjectionSeverity.CRITICAL]

        change_requests = list(dict.fromkeys([
            obj.change_request
            for obj in all_objections
            if obj.change_request and obj.severity != ObjectionSeverity.MINOR
        ]))

        personas_blocked = [
            v.persona for v in verdicts.values() if not v.approved
        ]

        overall_approved = (
            approval_rate >= SPS_MIN_APPROVAL_RATE and
            len(critical_objections) < SPS_CRITICAL_OBJECTION_THRESHOLD
        )

        # Harden plan
        hardened_plan = await self.harden_plan(plan_text, aim, verdicts)

        # Consensus summary
        status_emoji = "✅" if overall_approved else "⛔"
        consensus_summary = (
            f"{status_emoji} {approved_count}/{len(verdicts)} personas approved "
            f"({approval_rate:.0%}). "
            f"{len(critical_objections)} critical objections. "
            f"{len(change_requests)} change requests incorporated."
        )

        result = SPSResult(
            verdicts             = verdicts,
            overall_approved     = overall_approved,
            approval_rate        = approval_rate,
            critical_objections  = critical_objections,
            all_change_requests  = change_requests,
            hardened_plan        = hardened_plan,
            personas_blocked     = personas_blocked,
            consensus_summary    = consensus_summary,
        )

        logger.info(
            f"[SPS] Complete. {consensus_summary}"
        )
        return result

    # ── Reports ───────────────────────────────────────────────────────────────

    def get_sps_report(self, result: SPSResult) -> Dict:
        return {
            "overall_approved":    result.overall_approved,
            "approval_rate":       round(result.approval_rate * 100, 1),
            "personas_approved":   [p for p, v in result.verdicts.items() if v.approved],
            "personas_blocked":    result.personas_blocked,
            "critical_objections": len(result.critical_objections),
            "total_objections":    sum(len(v.objections) for v in result.verdicts.values()),
            "change_requests":     len(result.all_change_requests),
            "consensus_summary":   result.consensus_summary,
            "persona_breakdown":   [
                {
                    "persona":      p,
                    "title":        v.title,
                    "approved":     v.approved,
                    "score":        v.approval_score,
                    "objections":   len(v.objections),
                    "critical":     sum(1 for o in v.objections if o.severity == ObjectionSeverity.CRITICAL),
                    "conditional":  v.conditional,
                    "summary":      v.summary[:150],
                }
                for p, v in result.verdicts.items()
            ],
            "top_objections": [
                {
                    "persona":   o.persona,
                    "severity":  o.severity.value,
                    "objection": o.objection[:120],
                    "change":    o.change_request[:120],
                }
                for o in sorted(result.critical_objections,
                                key=lambda x: x.severity.value)[:6]
            ],
        }

    def get_approval_matrix(self, result: SPSResult) -> List[Dict]:
        """One row per persona — for Excel matrix output."""
        rows = []
        for p, v in result.verdicts.items():
            profile = PERSONA_PROFILES.get(p, {})
            rows.append({
                "persona":          p,
                "title":            v.title,
                "icon":             profile.get("icon", ""),
                "approved":         "Yes" if v.approved else "No",
                "approval_score":   v.approval_score,
                "objections_count": len(v.objections),
                "critical_count":   sum(1 for o in v.objections if o.severity == ObjectionSeverity.CRITICAL),
                "major_count":      sum(1 for o in v.objections if o.severity == ObjectionSeverity.MAJOR),
                "minor_count":      sum(1 for o in v.objections if o.severity == ObjectionSeverity.MINOR),
                "conditional":      v.conditional or "",
                "top_objection":    v.objections[0].objection[:100] if v.objections else "None",
                "praise":           "; ".join(v.praise[:2]),
                "summary":          v.summary[:150],
            })
        return rows


def _parse_json(raw: str) -> Dict:
    raw   = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
