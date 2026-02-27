# ═══════════════════════════════════════════════════════════════════════════════
# adversarial_red_team_simulator.py — Feature 13: ARTS
# Simulates external adversaries attacking the plan to find vulnerabilities
# before delivery. Generates hardened counter-measures for each attack vector.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

ARTS_ADVERSARY_TYPES   = ["competitor", "regulator", "market_force", "technical_failure", "black_swan"]
ARTS_MIN_SEVERITY      = "minor"   # ignore attacks below this
ARTS_MAX_ATTACKS_PER_TYPE = 3


# ── Data Structures ───────────────────────────────────────────────────────────

class AdversaryType(Enum):
    COMPETITOR       = "competitor"
    REGULATOR        = "regulator"
    MARKET_FORCE     = "market_force"
    TECHNICAL_FAILURE= "technical_failure"
    BLACK_SWAN       = "black_swan"


@dataclass
class AttackVector:
    attack_id:         str
    adversary_type:    AdversaryType
    attack_name:       str
    attack_description:str
    target_steps:      List[str]       # plan steps most vulnerable to this attack
    probability:       float           # 0–1 likelihood
    impact_severity:   str             # critical | major | minor
    impact_description:str
    earliest_trigger:  str             # When in plan execution could this occur
    detection_signals: List[str]       # early warning indicators


@dataclass
class CounterMeasure:
    attack_id:          str
    countermeasure_id:  str
    action:             str
    implementation_step:str   # WHERE in plan to add this countermeasure
    cost_hrs:           float
    effectiveness:      float   # 0–1 probability attack fails if measure applied
    residual_risk:      float   # 0–1 risk remaining after countermeasure


@dataclass
class RedTeamReport:
    total_attacks_simulated:  int
    critical_vulnerabilities: List[AttackVector]
    major_vulnerabilities:    List[AttackVector]
    minor_vulnerabilities:    List[AttackVector]
    countermeasures:          Dict[str, CounterMeasure]   # attack_id → countermeasure
    hardened_plan_additions:  List[str]   # new steps to add to plan
    residual_risk_score:      float       # 0–100 (lower = more residual risk)
    plan_survivability_score: float       # 0–100 probability plan succeeds under attack


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_ARTS_ADVERSARY = """You are a {adversary_type_upper} adversary. Your goal is to DESTROY or BLOCK this plan.

AIM: {aim}
PLAN TO ATTACK:
{plan_text}

ADVERSARY PROFILE: {adversary_profile}

Identify {max_attacks} specific, realistic attack vectors this adversary would deploy.
Be concrete — no vague threats. Each attack must target specific steps.

Attack severities:
  critical = stops plan entirely
  major    = causes 40%+ delay or cost overrun
  minor    = causes <20% friction

Respond ONLY with valid JSON (no markdown):
{{"attacks": [
  {{
    "attack_name": "...",
    "attack_description": "...",
    "target_steps": ["step_N"],
    "probability": 0.0,
    "impact_severity": "critical|major|minor",
    "impact_description": "...",
    "earliest_trigger": "After step_2 is completed",
    "detection_signals": ["Signal A", "Signal B"]
  }}
]}}"""

PROMPT_ARTS_COUNTERMEASURE = """You are a DEFENSIVE STRATEGY ARCHITECT.

AIM: {aim}
ATTACK VECTOR:
  Name: {attack_name}
  Adversary: {adversary_type}
  Description: {attack_description}
  Target Steps: {target_steps}
  Probability: {probability}
  Severity: {severity}

Design ONE highly effective countermeasure that:
  1. Directly neutralises this specific attack
  2. Can be embedded into the existing plan
  3. Is proportional to attack probability × severity
  4. Specifies WHERE in plan execution to implement

Respond ONLY with valid JSON (no markdown):
{{"action": "...", "implementation_step": "Add before step_N or after step_M", "cost_hrs": 0.0, "effectiveness": 0.0, "residual_risk": 0.0, "rationale": "..."}}"""

ADVERSARY_PROFILES = {
    "competitor":        "A well-funded direct competitor with 3x your resources, insider market knowledge, and willingness to engage in legal, pricing, and talent-poaching warfare.",
    "regulator":         "A regulatory body that can issue compliance requirements, demand audits, impose fines, and delay launches through bureaucratic processes.",
    "market_force":      "Unpredictable macroeconomic shifts: inflation, recession, supply chain disruption, currency fluctuation, or sudden demand collapse.",
    "technical_failure": "Systematic technical failures: infrastructure outages, security breaches, data corruption, third-party API failures, scaling bottlenecks.",
    "black_swan":        "An unprecedented, high-impact, low-probability event: pandemic, natural disaster, geopolitical crisis, or major technology disruption.",
}


# ── Engine ────────────────────────────────────────────────────────────────────

class AdversarialRedTeamSimulator:
    """
    Simulates 5 adversary types attacking the plan simultaneously,
    then generates targeted countermeasures and a hardened plan.
    """

    def __init__(
        self,
        call_fn: Callable[[str, str], Awaitable[str]],
        agents:  List[str] = None,
    ):
        self.call_fn      = call_fn
        self.agents       = agents or ["gemini", "openai", "together", "groq", "cohere"]
        self._all_attacks: List[AttackVector] = []
        self._countermeasures: Dict[str, CounterMeasure] = {}

    # ── Attack Simulation ─────────────────────────────────────────────────────

    async def simulate_adversary(
        self,
        adversary_type: AdversaryType,
        plan_text:      str,
        aim:            str,
        agent:          str,
    ) -> List[AttackVector]:
        """One adversary generates their attack vectors."""
        prompt = PROMPT_ARTS_ADVERSARY.format(
            adversary_type_upper = adversary_type.value.upper().replace("_", " "),
            aim                  = aim,
            plan_text            = plan_text[:1500],
            adversary_profile    = ADVERSARY_PROFILES.get(adversary_type.value, "Unknown adversary"),
            max_attacks          = ARTS_MAX_ATTACKS_PER_TYPE,
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)
            attacks = []
            for i, atk in enumerate(data.get("attacks", [])[:ARTS_MAX_ATTACKS_PER_TYPE]):
                attack_id = f"{adversary_type.value}_{i+1}"
                attacks.append(AttackVector(
                    attack_id          = attack_id,
                    adversary_type     = adversary_type,
                    attack_name        = atk.get("attack_name", "Unknown attack"),
                    attack_description = atk.get("attack_description", ""),
                    target_steps       = atk.get("target_steps", []),
                    probability        = float(atk.get("probability", 0.3)),
                    impact_severity    = atk.get("impact_severity", "minor"),
                    impact_description = atk.get("impact_description", ""),
                    earliest_trigger   = atk.get("earliest_trigger", "Any time"),
                    detection_signals  = atk.get("detection_signals", []),
                ))
            logger.info(f"[ARTS] {adversary_type.value}: {len(attacks)} attacks generated")
            return attacks
        except Exception as e:
            logger.warning(f"[ARTS] {adversary_type.value} simulation failed: {e}")
            return []

    async def run_all_adversaries(
        self,
        plan_text: str,
        aim:       str,
    ) -> List[AttackVector]:
        """All 5 adversaries attack simultaneously."""
        adversary_agent_pairs = [
            (AdversaryType.COMPETITOR,        self.agents[0] if len(self.agents) > 0 else "gemini"),
            (AdversaryType.REGULATOR,         self.agents[1] if len(self.agents) > 1 else "gemini"),
            (AdversaryType.MARKET_FORCE,      self.agents[2] if len(self.agents) > 2 else "gemini"),
            (AdversaryType.TECHNICAL_FAILURE, self.agents[3] if len(self.agents) > 3 else "gemini"),
            (AdversaryType.BLACK_SWAN,        self.agents[4] if len(self.agents) > 4 else "gemini"),
        ]
        tasks = [
            self.simulate_adversary(adv_type, plan_text, aim, agent)
            for adv_type, agent in adversary_agent_pairs
        ]
        results = await asyncio.gather(*tasks)
        all_attacks = []
        for attack_list in results:
            all_attacks.extend(attack_list)

        # Sort by risk score = probability × severity_weight
        severity_weights = {"critical": 3.0, "major": 2.0, "minor": 1.0}
        all_attacks.sort(
            key=lambda a: a.probability * severity_weights.get(a.impact_severity, 1.0),
            reverse=True
        )
        self._all_attacks = all_attacks
        logger.info(f"[ARTS] Total attacks simulated: {len(all_attacks)}")
        return all_attacks

    # ── Countermeasure Generation ─────────────────────────────────────────────

    async def generate_countermeasure(
        self,
        attack: AttackVector,
        aim:    str,
        agent:  str,
    ) -> CounterMeasure:
        """Design one countermeasure for one attack vector."""
        prompt = PROMPT_ARTS_COUNTERMEASURE.format(
            aim             = aim,
            attack_name     = attack.attack_name,
            adversary_type  = attack.adversary_type.value,
            attack_description = attack.attack_description[:300],
            target_steps    = ", ".join(attack.target_steps),
            probability     = f"{attack.probability:.0%}",
            severity        = attack.impact_severity,
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)
            return CounterMeasure(
                attack_id           = attack.attack_id,
                countermeasure_id   = f"cm_{attack.attack_id}",
                action              = data.get("action", "Monitor and respond"),
                implementation_step = data.get("implementation_step", "Before Step 1"),
                cost_hrs            = float(data.get("cost_hrs", 4.0)),
                effectiveness       = float(data.get("effectiveness", 0.7)),
                residual_risk       = float(data.get("residual_risk", 0.3)),
            )
        except Exception as e:
            logger.warning(f"[ARTS] Countermeasure for {attack.attack_id} failed: {e}")
            return CounterMeasure(
                attack_id           = attack.attack_id,
                countermeasure_id   = f"cm_{attack.attack_id}",
                action              = f"Monitor for: {attack.detection_signals[:1]}",
                implementation_step = "Ongoing throughout plan",
                cost_hrs            = 2.0,
                effectiveness       = 0.5,
                residual_risk       = 0.5,
            )

    async def generate_all_countermeasures(
        self,
        attacks: Optional[List[AttackVector]] = None,
        aim:     str = "",
    ) -> Dict[str, CounterMeasure]:
        """Generate countermeasures for critical and major attacks only."""
        attacks = attacks or self._all_attacks
        priority_attacks = [
            a for a in attacks
            if a.impact_severity in ("critical", "major")
        ][:10]  # cap at 10 countermeasures

        tasks = []
        for i, attack in enumerate(priority_attacks):
            agent = self.agents[i % len(self.agents)]
            tasks.append(self.generate_countermeasure(attack, aim, agent))

        cms = await asyncio.gather(*tasks)
        self._countermeasures = {cm.attack_id: cm for cm in cms}
        logger.info(f"[ARTS] Generated {len(cms)} countermeasures")
        return self._countermeasures

    # ── Plan Hardening ────────────────────────────────────────────────────────

    def generate_hardened_plan_additions(
        self,
        countermeasures: Optional[Dict[str, CounterMeasure]] = None,
    ) -> List[str]:
        """
        Convert countermeasures into concrete plan steps to add.
        Groups by implementation_step location.
        """
        cms = countermeasures or self._countermeasures
        additions = []
        for cm in cms.values():
            if cm.effectiveness >= 0.6:
                step = f"[RISK MITIGATION] {cm.action} ({cm.implementation_step})"
                if step not in additions:
                    additions.append(step)
        return additions

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def run_full_red_team(
        self,
        plan_text: str,
        aim:       str,
    ) -> RedTeamReport:
        """Complete red team pipeline."""
        attacks = await self.run_all_adversaries(plan_text, aim)
        cms     = await self.generate_all_countermeasures(attacks, aim)
        additions = self.generate_hardened_plan_additions(cms)

        critical = [a for a in attacks if a.impact_severity == "critical"]
        major    = [a for a in attacks if a.impact_severity == "major"]
        minor    = [a for a in attacks if a.impact_severity == "minor"]

        # Residual risk = average residual risk of countermeasures (0=worst, 100=best)
        if cms:
            avg_residual = sum(cm.residual_risk for cm in cms.values()) / len(cms)
            residual_score = (1 - avg_residual) * 100
        else:
            residual_score = 50.0

        # Survivability: probability plan succeeds despite attacks
        attack_kill_prob = 1.0
        for atk in critical:
            cm = cms.get(atk.attack_id)
            if cm:
                effective_prob = atk.probability * (1 - cm.effectiveness)
            else:
                effective_prob = atk.probability
            attack_kill_prob *= (1 - effective_prob)

        survivability = attack_kill_prob * 100

        report = RedTeamReport(
            total_attacks_simulated  = len(attacks),
            critical_vulnerabilities = critical,
            major_vulnerabilities    = major,
            minor_vulnerabilities    = minor,
            countermeasures          = cms,
            hardened_plan_additions  = additions,
            residual_risk_score      = residual_score,
            plan_survivability_score = survivability,
        )
        logger.info(
            f"[ARTS] Red team complete. Attacks={len(attacks)} "
            f"Critical={len(critical)} Survivability={survivability:.1f}%"
        )
        return report

    def get_arts_report(self, report: RedTeamReport) -> Dict:
        """Serialise for Excel/frontend output."""
        def atk_dict(a: AttackVector) -> Dict:
            return {
                "attack_id":    a.attack_id,
                "adversary":    a.adversary_type.value,
                "name":         a.attack_name,
                "severity":     a.impact_severity,
                "probability":  round(a.probability, 2),
                "target_steps": a.target_steps,
                "signals":      a.detection_signals[:3],
                "has_countermeasure": a.attack_id in report.countermeasures,
            }

        return {
            "total_attacks":          report.total_attacks_simulated,
            "critical_count":         len(report.critical_vulnerabilities),
            "major_count":            len(report.major_vulnerabilities),
            "minor_count":            len(report.minor_vulnerabilities),
            "countermeasures_count":  len(report.countermeasures),
            "residual_risk_score":    round(report.residual_risk_score, 1),
            "survivability_score":    round(report.plan_survivability_score, 1),
            "hardened_steps_added":   len(report.hardened_plan_additions),
            "critical_attacks":       [atk_dict(a) for a in report.critical_vulnerabilities],
            "major_attacks":          [atk_dict(a) for a in report.major_vulnerabilities],
            "countermeasures":        [
                {
                    "attack_id": cm.attack_id,
                    "action":    cm.action[:100],
                    "effectiveness": round(cm.effectiveness, 2),
                    "cost_hrs":  cm.cost_hrs,
                }
                for cm in report.countermeasures.values()
            ],
            "hardened_additions":     report.hardened_plan_additions,
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
