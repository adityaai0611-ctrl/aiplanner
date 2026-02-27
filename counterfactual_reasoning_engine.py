# ═══════════════════════════════════════════════════════════════════════════════
# counterfactual_reasoning_engine.py — Feature 16: CFE
# Simulates alternative plan histories using Structural Causal Models.
# "What if Step 3 came first?" → scores each counterfactual → injects best into next session.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

CFE_DB_PATH                = "counterfactuals.db"
CFE_MAX_COUNTERFACTUALS    = 8      # max alternative histories to evaluate
CFE_MIN_SCORE_IMPROVEMENT  = 5.0   # only store if counterfactual beats original by this
CFE_MAX_INTERVENTIONS      = 3      # max steps to swap/remove per counterfactual
CFE_TOP_K_INJECT           = 2      # top counterfactuals to inject into next session


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Intervention:
    """One causal intervention on the plan."""
    intervention_id:   str
    intervention_type: str       # swap | remove | reorder | replace | add
    target_step_idx:   int
    source_step_idx:   Optional[int]    # for swap/reorder
    replacement_text:  Optional[str]    # for replace/add
    rationale:         str


@dataclass
class Counterfactual:
    """An alternative plan history produced by applying interventions."""
    cf_id:              str
    original_plan:      str
    modified_plan:      str
    interventions:      List[Intervention]
    original_score:     float
    counterfactual_score: float
    score_delta:        float
    do_calculus_query:  str     # "P(outcome | do(step_3=X))" natural language
    causal_explanation: str
    is_improvement:     bool
    session_id:         str
    timestamp:          str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CFEReport:
    original_score:         float
    best_counterfactual:    Optional[Counterfactual]
    all_counterfactuals:    List[Counterfactual]
    improvement_found:      bool
    max_score_delta:        float
    injection_context:      str     # priming text for next session
    structural_insights:    List[str]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_CFE_INTERVENTION_GENERATOR = """You are a CAUSAL INTERVENTION DESIGNER using Structural Causal Models.

AIM: {aim}
ORIGINAL PLAN:
{numbered_steps}
ORIGINAL SCORE: {original_score}/100

Design {n_interventions} counterfactual interventions — alternative plan histories.
Each intervention changes HOW the plan is structured, not the goal.

Intervention types:
  swap    — swap two steps' positions
  remove  — remove a step entirely
  reorder — move a step earlier/later
  replace — replace a step with a better alternative
  add     — add a missing step at a specific position

For each intervention, write the do-calculus query as natural language:
  "What if step_N happened BEFORE step_M?"
  "What if we SKIPPED step_K entirely?"
  "What if step_J was replaced with [better action]?"

Respond ONLY with valid JSON:
{{
  "interventions": [
    {{
      "intervention_type": "swap",
      "target_step_idx": 2,
      "source_step_idx": 4,
      "replacement_text": null,
      "rationale": "Doing market research before product design prevents costly pivots",
      "do_calculus_query": "What if market research happened before product design?"
    }}
  ]
}}"""

PROMPT_CFE_MODIFIED_PLAN = """You are a COUNTERFACTUAL PLAN GENERATOR.

AIM: {aim}
ORIGINAL PLAN:
{numbered_steps}

INTERVENTION: {intervention_description}
DO-CALCULUS QUERY: "{do_calculus_query}"

Generate the MODIFIED plan after applying this intervention.
The plan must still achieve the AIM but with the structural change applied.
Fix any logical inconsistencies caused by the reordering.

Output ONLY the modified plan as numbered steps (no JSON, no explanation):"""

PROMPT_CFE_CAUSAL_EXPLAINER = """You are a CAUSAL EXPLANATION SPECIALIST.

AIM: {aim}
ORIGINAL PLAN (score={original_score}):
{original_plan}

COUNTERFACTUAL PLAN (score={cf_score}):
{cf_plan}

INTERVENTION APPLIED: {intervention_desc}

Explain WHY the counterfactual {'outperformed' if cf_score > original_score else 'underperformed'} the original:
  1. What causal mechanism explains the score difference?
  2. Which dependencies changed when the structure changed?
  3. What does this reveal about the original plan's weak point?

Respond ONLY with valid JSON:
{{"causal_explanation": "...", "structural_insight": "...", "lesson_for_future_plans": "..."}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class CounterfactualReasoningEngine:
    """
    Applies Structural Causal Model logic to plan evaluation.

    After a session delivers a plan:
    1. Generate N intervention designs (step swaps, removals, replacements)
    2. Apply each intervention → produce modified plan
    3. Score each counterfactual
    4. Extract causal explanations for score differences
    5. Persist best counterfactuals for priming future sessions
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        score_fn: Callable[[str, str], Awaitable[float]],
        db_path:  str = CFE_DB_PATH,
        agent:    str = "gemini",
    ):
        self.call_fn  = call_fn
        self.score_fn = score_fn
        self.db_path  = db_path
        self.agent    = agent
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS counterfactuals (
                    cf_id               TEXT PRIMARY KEY,
                    session_id          TEXT,
                    aim_hash            TEXT,
                    original_score      REAL,
                    cf_score            REAL,
                    score_delta         REAL,
                    do_calculus_query   TEXT,
                    causal_explanation  TEXT,
                    structural_insight  TEXT,
                    cf_plan             TEXT,
                    timestamp           TEXT
                )
            """)
            conn.commit()

    # ── Intervention Generation ───────────────────────────────────────────────

    async def generate_interventions(
        self,
        steps:          List[str],
        aim:            str,
        original_score: float,
        n:              int = CFE_MAX_COUNTERFACTUALS,
    ) -> List[Intervention]:
        """LLM designs N structural interventions on the plan."""
        numbered = "\n".join(f"  step_{i+1}: {s}" for i, s in enumerate(steps))
        prompt   = PROMPT_CFE_INTERVENTION_GENERATOR.format(
            aim              = aim,
            numbered_steps   = numbered,
            original_score   = original_score,
            n_interventions  = min(n, CFE_MAX_COUNTERFACTUALS),
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            interventions = []
            for i, iv in enumerate(data.get("interventions", [])[:n]):
                interventions.append(Intervention(
                    intervention_id   = f"iv_{i+1}",
                    intervention_type = iv.get("intervention_type", "swap"),
                    target_step_idx   = int(iv.get("target_step_idx", 0)),
                    source_step_idx   = iv.get("source_step_idx"),
                    replacement_text  = iv.get("replacement_text"),
                    rationale         = iv.get("rationale", ""),
                ))
                # Attach do_calculus_query to rationale for retrieval
                interventions[-1].rationale += f" | QUERY: {iv.get('do_calculus_query','')}"
            return interventions
        except Exception as e:
            logger.warning(f"[CFE] Intervention generation failed: {e}")
            return self._heuristic_interventions(steps)

    def _heuristic_interventions(self, steps: List[str]) -> List[Intervention]:
        """Fallback: generate swap interventions without LLM."""
        interventions = []
        pairs = list(combinations(range(len(steps)), 2))[:CFE_MAX_COUNTERFACTUALS]
        for i, (a, b) in enumerate(pairs):
            interventions.append(Intervention(
                intervention_id   = f"iv_h{i+1}",
                intervention_type = "swap",
                target_step_idx   = a,
                source_step_idx   = b,
                replacement_text  = None,
                rationale         = f"Heuristic: swap step_{a+1} and step_{b+1}",
            ))
        return interventions

    # ── Apply Intervention ────────────────────────────────────────────────────

    def apply_intervention(
        self,
        steps:        List[str],
        intervention: Intervention,
    ) -> List[str]:
        """
        Deterministically applies an intervention to produce modified steps.
        Falls back to LLM for 'replace' and 'add' types.
        """
        s = list(steps)
        t = intervention.target_step_idx
        r = intervention.source_step_idx

        try:
            if intervention.intervention_type == "swap" and r is not None:
                if 0 <= t < len(s) and 0 <= r < len(s):
                    s[t], s[r] = s[r], s[t]

            elif intervention.intervention_type == "remove":
                if 0 <= t < len(s):
                    s.pop(t)

            elif intervention.intervention_type == "reorder" and r is not None:
                if 0 <= t < len(s):
                    step = s.pop(t)
                    insert_at = max(0, min(r, len(s)))
                    s.insert(insert_at, step)

            elif intervention.intervention_type == "replace":
                if 0 <= t < len(s) and intervention.replacement_text:
                    s[t] = intervention.replacement_text

            elif intervention.intervention_type == "add":
                if intervention.replacement_text:
                    insert_at = max(0, min(t, len(s)))
                    s.insert(insert_at, intervention.replacement_text)
        except Exception as e:
            logger.warning(f"[CFE] Intervention application error: {e}")

        return s

    async def generate_modified_plan(
        self,
        original_steps: List[str],
        aim:            str,
        intervention:   Intervention,
    ) -> str:
        """
        For complex interventions, ask LLM to generate a coherent modified plan.
        For simple swaps/removals, use deterministic application.
        """
        if intervention.intervention_type in ("swap", "remove", "reorder"):
            modified = self.apply_intervention(original_steps, intervention)
            return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(modified))

        # For replace/add: LLM generates coherent plan
        numbered = "\n".join(f"  step_{i+1}: {s}" for i, s in enumerate(original_steps))
        do_query = intervention.rationale.split("| QUERY:")[-1].strip() \
                   if "| QUERY:" in intervention.rationale else intervention.rationale

        prompt = PROMPT_CFE_MODIFIED_PLAN.format(
            aim                      = aim,
            numbered_steps           = numbered,
            intervention_description = f"{intervention.intervention_type}: {intervention.rationale[:200]}",
            do_calculus_query        = do_query[:200],
        )
        try:
            return await self.call_fn(self.agent, prompt)
        except Exception:
            modified = self.apply_intervention(original_steps, intervention)
            return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(modified))

    # ── Causal Explanation ────────────────────────────────────────────────────

    async def explain_counterfactual(
        self,
        original_plan:  str,
        cf_plan:        str,
        original_score: float,
        cf_score:       float,
        intervention:   Intervention,
        aim:            str,
    ) -> Tuple[str, str]:
        """Returns (causal_explanation, structural_insight)."""
        prompt = PROMPT_CFE_CAUSAL_EXPLAINER.format(
            aim              = aim,
            original_score   = original_score,
            original_plan    = original_plan[:600],
            cf_score         = cf_score,
            cf_plan          = cf_plan[:600],
            intervention_desc= intervention.rationale[:200],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return (
                data.get("causal_explanation", "Score difference attributed to structural change"),
                data.get("structural_insight", data.get("lesson_for_future_plans", "")),
            )
        except Exception:
            delta = cf_score - original_score
            return (
                f"Counterfactual {'improved' if delta > 0 else 'reduced'} score by {abs(delta):.1f} pts.",
                f"Intervention: {intervention.rationale[:100]}"
            )

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def analyse(
        self,
        steps:          List[str],
        plan_text:      str,
        aim:            str,
        original_score: float,
        session_id:     str,
    ) -> CFEReport:
        """
        Full CFE pipeline:
        1. Generate interventions
        2. Apply each → modified plan
        3. Score each counterfactual
        4. Explain top counterfactuals
        5. Persist improvements
        6. Build injection context
        """
        logger.info(f"[CFE] Analysing {len(steps)} steps, original_score={original_score:.1f}")

        interventions = await self.generate_interventions(steps, aim, original_score)

        # Score all counterfactuals in parallel
        async def evaluate_one(iv: Intervention) -> Optional[Counterfactual]:
            try:
                cf_plan = await self.generate_modified_plan(steps, aim, iv)
                cf_score = await self.score_fn(cf_plan, aim)
                delta    = cf_score - original_score
                do_query = iv.rationale.split("| QUERY:")[-1].strip() \
                           if "| QUERY:" in iv.rationale else iv.rationale

                return Counterfactual(
                    cf_id                = f"cf_{iv.intervention_id}_{session_id[:8]}",
                    original_plan        = plan_text,
                    modified_plan        = cf_plan,
                    interventions        = [iv],
                    original_score       = original_score,
                    counterfactual_score = cf_score,
                    score_delta          = delta,
                    do_calculus_query    = do_query,
                    causal_explanation   = "",
                    is_improvement       = delta >= CFE_MIN_SCORE_IMPROVEMENT,
                    session_id           = session_id,
                )
            except Exception as e:
                logger.warning(f"[CFE] Evaluation failed for {iv.intervention_id}: {e}")
                return None

        results = await asyncio.gather(*[evaluate_one(iv) for iv in interventions])
        counterfactuals = [r for r in results if r is not None]
        counterfactuals.sort(key=lambda c: c.score_delta, reverse=True)

        # Explain top 2
        for cf in counterfactuals[:2]:
            expl, insight = await self.explain_counterfactual(
                cf.original_plan, cf.modified_plan,
                cf.original_score, cf.counterfactual_score,
                cf.interventions[0], aim,
            )
            cf.causal_explanation = expl
            cf.interventions[0].rationale = insight or cf.interventions[0].rationale

        # Persist improvements
        improvements = [c for c in counterfactuals if c.is_improvement]
        self._persist(improvements[:CFE_TOP_K_INJECT], aim)

        best = counterfactuals[0] if counterfactuals else None
        max_delta = best.score_delta if best else 0.0
        injection = self._build_injection_context(improvements[:CFE_TOP_K_INJECT])

        insights = [
            cf.interventions[0].rationale
            for cf in improvements[:3]
            if cf.interventions
        ]

        report = CFEReport(
            original_score       = original_score,
            best_counterfactual  = best,
            all_counterfactuals  = counterfactuals,
            improvement_found    = bool(improvements),
            max_score_delta      = max_delta,
            injection_context    = injection,
            structural_insights  = insights,
        )
        logger.info(
            f"[CFE] Complete. {len(counterfactuals)} CFs evaluated, "
            f"{len(improvements)} improvements, best_delta={max_delta:.1f}"
        )
        return report

    # ── Injection Context ─────────────────────────────────────────────────────

    def _build_injection_context(self, improvements: List[Counterfactual]) -> str:
        if not improvements:
            return ""
        lines = ["╔══ COUNTERFACTUAL INSIGHTS (from prior sessions) ══╗"]
        for cf in improvements:
            lines.append(f"  ✓ {cf.do_calculus_query}")
            lines.append(f"    → Score improvement: +{cf.score_delta:.1f} pts")
            if cf.causal_explanation:
                lines.append(f"    → Why: {cf.causal_explanation[:120]}")
        lines.append("╚════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def get_historical_injections(self, aim_hash: str) -> str:
        """Retrieve best past counterfactuals for this aim type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT do_calculus_query, score_delta, causal_explanation
                   FROM counterfactuals
                   WHERE aim_hash=? AND score_delta >= ?
                   ORDER BY score_delta DESC LIMIT ?""",
                (aim_hash, CFE_MIN_SCORE_IMPROVEMENT, CFE_TOP_K_INJECT)
            ).fetchall()
        if not rows:
            return ""
        lines = ["╔══ PROVEN STRUCTURAL IMPROVEMENTS (from past sessions) ══╗"]
        for query, delta, expl in rows:
            lines.append(f"  ✓ {query}")
            lines.append(f"    → +{delta:.1f} pts | {(expl or '')[:100]}")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def _persist(self, cfs: List[Counterfactual], aim: str) -> None:
        import hashlib
        aim_hash = hashlib.md5(aim.encode()).hexdigest()[:12]
        with sqlite3.connect(self.db_path) as conn:
            for cf in cfs:
                expl   = cf.causal_explanation
                insight= cf.interventions[0].rationale if cf.interventions else ""
                conn.execute(
                    """INSERT OR REPLACE INTO counterfactuals VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (cf.cf_id, cf.session_id, aim_hash,
                     cf.original_score, cf.counterfactual_score, cf.score_delta,
                     cf.do_calculus_query, expl, insight,
                     cf.modified_plan[:500], cf.timestamp)
                )
            conn.commit()

    def get_cfe_report(self, report: CFEReport) -> Dict:
        return {
            "original_score":       report.original_score,
            "counterfactuals_run":  len(report.all_counterfactuals),
            "improvements_found":   len([c for c in report.all_counterfactuals if c.is_improvement]),
            "max_score_delta":      round(report.max_score_delta, 2),
            "improvement_found":    report.improvement_found,
            "structural_insights":  report.structural_insights[:3],
            "best_cf": {
                "do_query":    report.best_counterfactual.do_calculus_query,
                "score_delta": round(report.best_counterfactual.score_delta, 2),
                "explanation": report.best_counterfactual.causal_explanation[:200],
            } if report.best_counterfactual else None,
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
