# ═══════════════════════════════════════════════════════════════════════════════
# confidence_calibration_engine.py — Feature 25: CCE
# Each agent assigns confidence scores to its own claims.
# Calibrates those scores against historical accuracy.
# Overconfident agents are penalised. Underconfident agents are boosted.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

CCE_DB_PATH                  = "confidence_calibration.db"
CCE_MIN_HISTORY_FOR_CALIB    = 5      # need this many sessions before calibrating
CCE_OVERCONFIDENCE_PENALTY   = 0.15   # score multiplier penalty
CCE_UNDERCONFIDENCE_BOOST    = 0.08   # score multiplier boost
CCE_BRIER_THRESHOLD_GOOD     = 0.20   # Brier score below this = well-calibrated
CCE_BRIER_THRESHOLD_POOR     = 0.35   # above this = poorly calibrated
CCE_EMA_ALPHA                = 0.25


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class StepConfidence:
    """An agent's self-reported confidence for one plan step."""
    step_id:        str
    step_text:      str
    confidence:     float        # 0–1: agent's stated confidence this step is correct
    uncertainty_reason: str      # why the agent is uncertain (if conf < 0.7)
    estimated_hrs:  Optional[float]
    alternatives:   List[str]    # what the agent would do differently if unsure


@dataclass
class AgentCalibration:
    """Historical calibration stats for one agent."""
    agent:                str
    sessions_tracked:     int
    mean_stated_conf:     float      # what the agent says on average
    mean_actual_accuracy: float      # what actually scored well
    brier_score:          float      # calibration quality (lower = better)
    calibration_bias:     str        # overconfident | underconfident | well_calibrated
    calibration_factor:   float      # multiply stated conf by this to get true conf
    last_updated:         str


@dataclass
class CCEResult:
    agent_confidences:    Dict[str, List[StepConfidence]]   # agent → steps
    calibrations:         Dict[str, AgentCalibration]
    adjusted_scores:      Dict[str, float]     # score after calibration adjustment
    low_confidence_steps: List[Tuple[str, str, float]]  # (agent, step_text, conf)
    consensus_confidence: float   # aggregate confidence across all agents
    reliability_ranking:  List[str]   # agents ranked by calibration quality


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_CCE_SELF_ASSESS = """You are performing CONFIDENCE SELF-ASSESSMENT on your own plan.

AIM: {aim}
YOUR PLAN:
{plan_text}

For each step, assess your OWN confidence that:
  (a) This step will actually work as described
  (b) Your time estimate is accurate
  (c) The dependencies are correct

Confidence scale: 0.0 = completely guessing, 1.0 = certain based on strong evidence

Be HONEST and CALIBRATED:
  - If you have domain expertise for this step → 0.85–0.95
  - If you're reasoning by analogy → 0.60–0.75
  - If you're uncertain but plausible → 0.40–0.60
  - If you're speculating → 0.20–0.40

Respond ONLY with valid JSON:
{{
  "step_confidences": [
    {{
      "step_id": "step_1",
      "step_text": "...",
      "confidence": 0.0,
      "uncertainty_reason": "...",
      "estimated_hrs": 0.0,
      "alternatives": ["alternative approach if wrong"]
    }}
  ],
  "overall_plan_confidence": 0.0,
  "weakest_step_id": "step_N",
  "strongest_step_id": "step_M"
}}"""

PROMPT_CCE_CALIBRATION_ANALYSER = """You are a PROBABILISTIC CALIBRATION ANALYST.

AGENT: {agent}
HISTORICAL DATA ({n_sessions} sessions):
  Mean stated confidence: {mean_stated:.2f}
  Mean actual accuracy (measured by plan scores): {mean_actual:.2f}
  Brier score: {brier:.3f}

CALIBRATION DEFINITION:
  Well-calibrated: when agent says 0.7, plans score ~70/100
  Overconfident:   agent says 0.9 but plans score 55/100
  Underconfident:  agent says 0.4 but plans score 75/100

Brier score: 0.0 = perfect, 0.25 = random, >0.35 = poorly calibrated

Determine:
  1. Is this agent overconfident, underconfident, or well-calibrated?
  2. What calibration factor should be applied? (multiply stated confidence by this)
  3. What's the likely source of miscalibration?

Respond ONLY with valid JSON:
{{
  "calibration_bias": "overconfident|underconfident|well_calibrated",
  "calibration_factor": 1.0,
  "source_of_bias": "...",
  "recommended_action": "penalise|boost|trust|monitor"
}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class ConfidenceCalibrationEngine:
    """
    Tracks whether agents are well-calibrated in their confidence estimates.

    Pipeline per session:
    1. Each agent self-assesses confidence on its own plan steps
    2. Calibration factors are applied (from historical data)
    3. Adjusted confidence scores influence final scoring
    4. Outcome stored for future calibration updates

    Over time: well-calibrated agents gain credibility weight.
               Overconfident agents are progressively penalised.
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        db_path:  str = CCE_DB_PATH,
    ):
        self.call_fn = call_fn
        self.db_path = db_path
        self._calibrations: Dict[str, AgentCalibration] = {}
        self._init_db()
        self._calibrations = self._load_calibrations()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_calibrations (
                    agent                TEXT PRIMARY KEY,
                    sessions_tracked     INTEGER DEFAULT 0,
                    mean_stated_conf     REAL DEFAULT 0.7,
                    mean_actual_accuracy REAL DEFAULT 0.7,
                    brier_score          REAL DEFAULT 0.25,
                    calibration_bias     TEXT DEFAULT 'well_calibrated',
                    calibration_factor   REAL DEFAULT 1.0,
                    last_updated         TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence_sessions (
                    session_id      TEXT,
                    agent           TEXT,
                    stated_conf     REAL,
                    actual_score    REAL,
                    brier_contrib   REAL,
                    timestamp       TEXT
                )
            """)
            conn.commit()

    def _load_calibrations(self) -> Dict[str, AgentCalibration]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM agent_calibrations"
            ).fetchall()
        result = {}
        for row in rows:
            result[row[0]] = AgentCalibration(
                agent=row[0], sessions_tracked=row[1],
                mean_stated_conf=row[2], mean_actual_accuracy=row[3],
                brier_score=row[4], calibration_bias=row[5],
                calibration_factor=row[6], last_updated=row[7] or "",
            )
        return result

    # ── Self-Assessment ───────────────────────────────────────────────────────

    async def elicit_confidence(
        self,
        agent:     str,
        plan_text: str,
        aim:       str,
    ) -> List[StepConfidence]:
        """Ask agent to self-assess confidence on each of its plan steps."""
        prompt = PROMPT_CCE_SELF_ASSESS.format(
            aim       = aim,
            plan_text = plan_text[:1500],
        )
        try:
            raw  = await self.call_fn(agent, prompt)
            data = _parse_json(raw)
            confidences = []
            for sc in data.get("step_confidences", []):
                confidences.append(StepConfidence(
                    step_id     = sc.get("step_id", ""),
                    step_text   = sc.get("step_text", "")[:200],
                    confidence  = max(0.0, min(1.0, float(sc.get("confidence", 0.7)))),
                    uncertainty_reason = sc.get("uncertainty_reason", ""),
                    estimated_hrs      = sc.get("estimated_hrs"),
                    alternatives       = sc.get("alternatives", [])[:2],
                ))
            return confidences
        except Exception as e:
            logger.warning(f"[CCE] Self-assessment failed for {agent}: {e}")
            # Heuristic: assign 0.7 to all steps
            lines = [l.strip() for l in plan_text.split('\n') if l.strip()]
            return [
                StepConfidence(
                    step_id    = f"step_{i+1}",
                    step_text  = line[:100],
                    confidence = 0.70,
                    uncertainty_reason = "default",
                    estimated_hrs = None,
                    alternatives  = [],
                )
                for i, line in enumerate(lines[:10])
            ]

    async def elicit_all(
        self,
        plans:  Dict[str, str],
        aim:    str,
    ) -> Dict[str, List[StepConfidence]]:
        """Parallel self-assessment from all agents."""
        tasks = {
            agent: self.elicit_confidence(agent, plan, aim)
            for agent, plan in plans.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    # ── Calibration Factor Application ────────────────────────────────────────

    def apply_calibration(
        self,
        agent:            str,
        raw_score:        float,
        step_confidences: List[StepConfidence],
    ) -> float:
        """
        Adjust agent score based on:
        1. Mean stated confidence vs calibration factor
        2. Brier score quality
        """
        calib = self._calibrations.get(agent)
        if not calib or calib.sessions_tracked < CCE_MIN_HISTORY_FOR_CALIB:
            return raw_score  # not enough history → no adjustment

        mean_conf = statistics.mean(sc.confidence for sc in step_confidences) \
                    if step_confidences else 0.7

        adjusted_conf = mean_conf * calib.calibration_factor

        if calib.calibration_bias == "overconfident":
            penalty   = CCE_OVERCONFIDENCE_PENALTY * (mean_conf - adjusted_conf)
            new_score = raw_score * (1 - penalty)
            logger.info(f"[CCE] {agent}: overconfident penalty → {raw_score:.1f}→{new_score:.1f}")
        elif calib.calibration_bias == "underconfident":
            boost     = CCE_UNDERCONFIDENCE_BOOST * (adjusted_conf - mean_conf)
            new_score = raw_score * (1 + boost)
            logger.info(f"[CCE] {agent}: underconfident boost → {raw_score:.1f}→{new_score:.1f}")
        else:
            new_score = raw_score

        # Additionally penalise for poor Brier score
        if calib.brier_score > CCE_BRIER_THRESHOLD_POOR:
            new_score *= 0.95

        return round(max(0.0, min(100.0, new_score)), 2)

    # ── Calibration Update ─────────────────────────────────────────────────────

    def record_outcome(
        self,
        session_id:  str,
        agent:       str,
        stated_conf: float,
        actual_score:float,
    ) -> None:
        """
        After session: record what agent said vs what score it got.
        Updates Brier score and running calibration stats.
        """
        # Normalise actual_score to 0–1
        actual_norm = actual_score / 100.0
        # Brier score contribution: (confidence - outcome)^2
        brier_contrib = (stated_conf - actual_norm) ** 2

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO confidence_sessions VALUES (?,?,?,?,?,?)",
                (session_id, agent, stated_conf, actual_norm,
                 brier_contrib, datetime.utcnow().isoformat())
            )
            # Update running stats
            rows = conn.execute(
                "SELECT stated_conf, actual_score, brier_contrib FROM confidence_sessions WHERE agent=?",
                (agent,)
            ).fetchall()

            if rows:
                mean_stated  = statistics.mean(r[0] for r in rows)
                mean_actual  = statistics.mean(r[1] for r in rows)
                mean_brier   = statistics.mean(r[2] for r in rows)

                # Determine bias
                diff = mean_stated - mean_actual
                if diff > 0.10:
                    bias   = "overconfident"
                    factor = mean_actual / max(mean_stated, 0.01)
                elif diff < -0.10:
                    bias   = "underconfident"
                    factor = mean_actual / max(mean_stated, 0.01)
                else:
                    bias   = "well_calibrated"
                    factor = 1.0

                conn.execute(
                    """INSERT OR REPLACE INTO agent_calibrations VALUES (?,?,?,?,?,?,?,?)""",
                    (agent, len(rows), mean_stated, mean_actual,
                     mean_brier, bias, round(factor, 3), datetime.utcnow().isoformat())
                )
            conn.commit()

        # Refresh in-memory
        self._calibrations = self._load_calibrations()

    async def update_calibrations_llm(self) -> None:
        """Periodically re-run LLM analysis on calibration data."""
        for agent, calib in self._calibrations.items():
            if calib.sessions_tracked < CCE_MIN_HISTORY_FOR_CALIB:
                continue
            prompt = PROMPT_CCE_CALIBRATION_ANALYSER.format(
                agent        = agent,
                n_sessions   = calib.sessions_tracked,
                mean_stated  = calib.mean_stated_conf,
                mean_actual  = calib.mean_actual_accuracy,
                brier        = calib.brier_score,
            )
            try:
                raw  = await self.call_fn(agent, prompt)
                data = _parse_json(raw)
                new_bias   = data.get("calibration_bias", calib.calibration_bias)
                new_factor = float(data.get("calibration_factor", calib.calibration_factor))
                # EMA blend
                calib.calibration_factor = (
                    CCE_EMA_ALPHA * new_factor + (1 - CCE_EMA_ALPHA) * calib.calibration_factor
                )
                calib.calibration_bias   = new_bias
            except Exception:
                pass

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def calibrate(
        self,
        plans:      Dict[str, str],
        scores:     Dict[str, float],
        aim:        str,
        session_id: str,
    ) -> CCEResult:
        """Full CCE pipeline."""
        # Elicit confidences from all agents
        agent_confidences = await self.elicit_all(plans, aim)

        # Apply calibration adjustments
        adjusted_scores: Dict[str, float] = {}
        for agent, raw_score in scores.items():
            confs = agent_confidences.get(agent, [])
            adjusted_scores[agent] = self.apply_calibration(agent, raw_score, confs)

        # Find low-confidence steps
        low_conf_steps = []
        for agent, confs in agent_confidences.items():
            for sc in confs:
                if sc.confidence < 0.55:
                    low_conf_steps.append((agent, sc.step_text, sc.confidence))
        low_conf_steps.sort(key=lambda x: x[2])

        # Consensus confidence
        all_confs = [
            sc.confidence
            for confs in agent_confidences.values()
            for sc in confs
        ]
        consensus_conf = statistics.mean(all_confs) if all_confs else 0.7

        # Reliability ranking by Brier score
        def calib_score(agent: str) -> float:
            c = self._calibrations.get(agent)
            if not c or c.sessions_tracked < 2:
                return 0.25  # default neutral
            return c.brier_score
        reliability = sorted(scores.keys(), key=calib_score)  # lowest Brier = best

        # Record outcomes
        for agent, score in scores.items():
            confs = agent_confidences.get(agent, [])
            stated = statistics.mean(sc.confidence for sc in confs) if confs else 0.7
            self.record_outcome(session_id, agent, stated, score)

        return CCEResult(
            agent_confidences    = agent_confidences,
            calibrations         = self._calibrations,
            adjusted_scores      = adjusted_scores,
            low_confidence_steps = low_conf_steps[:10],
            consensus_confidence = round(consensus_conf, 3),
            reliability_ranking  = reliability,
        )

    def get_cce_report(self, result: CCEResult) -> Dict:
        calib_data = {}
        for agent, calib in result.calibrations.items():
            calib_data[agent] = {
                "sessions":         calib.sessions_tracked,
                "brier_score":      round(calib.brier_score, 3),
                "bias":             calib.calibration_bias,
                "calib_factor":     round(calib.calibration_factor, 3),
                "mean_stated_conf": round(calib.mean_stated_conf, 2),
                "mean_actual_acc":  round(calib.mean_actual_accuracy, 2),
            }
        return {
            "consensus_confidence":  result.consensus_confidence,
            "reliability_ranking":   result.reliability_ranking,
            "score_adjustments":     {
                agent: {
                    "original": round(score, 1),
                    "adjusted": round(result.adjusted_scores.get(agent, score), 1),
                    "delta":    round(result.adjusted_scores.get(agent, score) - score, 1),
                }
                for agent, score in {
                    a: sum(sc.confidence for sc in confs) / max(len(confs), 1) * 100
                    for a, confs in result.agent_confidences.items()
                }.items()
            },
            "low_confidence_steps":  [
                {"agent": a, "step": s[:60], "confidence": round(c, 2)}
                for a, s, c in result.low_confidence_steps[:5]
            ],
            "agent_calibrations":    calib_data,
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
