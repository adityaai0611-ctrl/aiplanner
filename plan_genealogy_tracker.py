# ═══════════════════════════════════════════════════════════════════════════════
# plan_genealogy_tracker.py — Feature 24: PGT
# Full audit trail of plan evolution: who contributed what, when, and why.
# Every step is traceable to its origin agent, mutation, debate, or compression.
# ═══════════════════════════════════════════════════════════════════════════════

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)

PGT_DB_PATH = "plan_genealogy.db"


class OriginType(Enum):
    AGENT_ORIGINAL   = "agent_original"    # step written by an agent from scratch
    GPM_CROSSOVER    = "gpm_crossover"     # step produced by genetic crossover
    GPM_MUTATION     = "gpm_mutation"      # step from genetic mutation
    MADP_DEBATE      = "madp_debate"       # step emerged from adversarial debate
    SPCE_MERGE       = "spce_merge"        # step is a merge of two original steps
    SPS_HARDENING    = "sps_hardening"     # step added to address stakeholder objection
    ARTS_MITIGATION  = "arts_mitigation"   # step added as red-team countermeasure
    CFE_COUNTERFACT  = "cfe_counterfact"   # step from counterfactual intervention
    HUMAN_EDIT       = "human_edit"        # manually edited by user
    ESS_FIX          = "ess_fix"           # added by execution sandbox fix


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class StepLineage:
    """Full lineage record for one step in the final plan."""
    step_hash:          str           # SHA256[:12] of step text — unique ID
    step_text:          str
    step_index:         int           # position in final plan
    origin_type:        OriginType
    origin_agent:       Optional[str]
    origin_session:     str
    parent_hashes:      List[str]     # hashes of steps this was derived from
    generation:         int           # 0=original, 1=1st mutation/crossover, etc
    score_at_origin:    float         # score of plan when this step was created
    score_contribution: float         # estimated contribution to final score
    transformation_log: List[str]     # ordered list of transformations applied
    timestamp:          str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GenealogyReport:
    session_id:         str
    final_plan:         str
    step_lineages:      List[StepLineage]
    origin_distribution:Dict[str, int]    # OriginType → count
    top_contributor:    Optional[str]      # agent with most surviving steps
    generation_map:     Dict[int, int]     # generation → step count
    total_transformations: int
    audit_trail_text:   str


# ── Engine ────────────────────────────────────────────────────────────────────

class PlanGenealogyTracker:
    """
    Maintains a full audit trail of every step in the final plan.
    Tracks: who wrote it, which feature transformed it, at what score.

    Enables:
    - Attribution: "Gemini contributed 4 steps, MADP debate contributed 2"
    - Quality analysis: "GPM mutations improved score by +12 pts on average"
    - Trust audit: "Every step is traceable to a verified origin"
    - Learning: "Cohere's steps survive to final plan 73% of the time"
    """

    def __init__(self, db_path: str = PGT_DB_PATH):
        self.db_path = db_path
        self._lineages: Dict[str, StepLineage] = {}  # hash → lineage
        self._current_session: Optional[str]   = None
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS step_lineages (
                    step_hash           TEXT PRIMARY KEY,
                    step_text           TEXT,
                    step_index          INTEGER,
                    origin_type         TEXT,
                    origin_agent        TEXT,
                    origin_session      TEXT,
                    parent_hashes       TEXT,
                    generation          INTEGER,
                    score_at_origin     REAL,
                    score_contribution  REAL,
                    transformation_log  TEXT,
                    timestamp           TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_snapshots (
                    snapshot_id   TEXT PRIMARY KEY,
                    session_id    TEXT,
                    snapshot_type TEXT,
                    plan_text     TEXT,
                    score         REAL,
                    timestamp     TEXT
                )
            """)
            conn.commit()

    # ── Registration ──────────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> None:
        self._current_session = session_id
        self._lineages = {}
        logger.info(f"[PGT] Session started: {session_id}")

    def register_agent_plans(
        self,
        plans:  Dict[str, str],
        scores: Dict[str, float],
    ) -> None:
        """Register the initial agent-generated steps."""
        for agent, plan in plans.items():
            steps = _parse_steps(plan)
            for i, step in enumerate(steps):
                h = _hash(step)
                if h not in self._lineages:
                    self._lineages[h] = StepLineage(
                        step_hash          = h,
                        step_text          = step,
                        step_index         = i,
                        origin_type        = OriginType.AGENT_ORIGINAL,
                        origin_agent       = agent,
                        origin_session     = self._current_session or "unknown",
                        parent_hashes      = [],
                        generation         = 0,
                        score_at_origin    = scores.get(agent, 0.0),
                        score_contribution = 0.0,
                        transformation_log = [f"written by {agent}"],
                    )

    def record_transformation(
        self,
        original_step:    str,
        transformed_step: str,
        origin_type:      OriginType,
        agent:            Optional[str],
        score:            float,
        transformation:   str,
    ) -> str:
        """Record a step transformation and return new step hash."""
        parent_h   = _hash(original_step)
        child_h    = _hash(transformed_step)
        parent_gen = self._lineages.get(parent_h, StepLineage(
            step_hash="", step_text="", step_index=0,
            origin_type=OriginType.AGENT_ORIGINAL, origin_agent=None,
            origin_session="", parent_hashes=[], generation=0,
            score_at_origin=0, score_contribution=0, transformation_log=[]
        )).generation

        parent_log = self._lineages.get(parent_h, None)
        prev_log   = parent_log.transformation_log if parent_log else []

        self._lineages[child_h] = StepLineage(
            step_hash          = child_h,
            step_text          = transformed_step,
            step_index         = self._lineages.get(parent_h, StepLineage("","",0,OriginType.AGENT_ORIGINAL,None,"","",0,0,0,[])).step_index,
            origin_type        = origin_type,
            origin_agent       = agent,
            origin_session     = self._current_session or "unknown",
            parent_hashes      = [parent_h],
            generation         = parent_gen + 1,
            score_at_origin    = score,
            score_contribution = 0.0,
            transformation_log = prev_log + [transformation],
        )
        return child_h

    def record_merge(
        self,
        step_a:      str,
        step_b:      str,
        merged_step: str,
        score:       float,
    ) -> str:
        """Record SPCE merge of two steps into one."""
        ha, hb = _hash(step_a), _hash(step_b)
        hm     = _hash(merged_step)
        gen_a  = self._lineages.get(ha, StepLineage("","",0,OriginType.AGENT_ORIGINAL,None,"","",0,0,0,[])).generation
        gen_b  = self._lineages.get(hb, StepLineage("","",0,OriginType.AGENT_ORIGINAL,None,"","",0,0,0,[])).generation

        self._lineages[hm] = StepLineage(
            step_hash          = hm,
            step_text          = merged_step,
            step_index         = 0,
            origin_type        = OriginType.SPCE_MERGE,
            origin_agent       = None,
            origin_session     = self._current_session or "unknown",
            parent_hashes      = [ha, hb],
            generation         = max(gen_a, gen_b) + 1,
            score_at_origin    = score,
            score_contribution = 0.0,
            transformation_log = [
                f"merged from [{step_a[:40]}] + [{step_b[:40]}]"
            ],
        )
        return hm

    def snapshot_plan(
        self,
        plan_text:     str,
        score:         float,
        snapshot_type: str,
    ) -> None:
        """Save a plan snapshot at a pipeline phase."""
        snap_id = f"{self._current_session}_{snapshot_type}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO plan_snapshots VALUES (?,?,?,?,?,?)",
                (snap_id, self._current_session, snapshot_type,
                 plan_text[:2000], score, datetime.utcnow().isoformat())
            )
            conn.commit()

    # ── Attribution ───────────────────────────────────────────────────────────

    def compute_score_contributions(
        self,
        final_plan:   str,
        agent_scores: Dict[str, float],
    ) -> None:
        """
        Estimate each step's score contribution based on origin agent performance.
        """
        steps      = _parse_steps(final_plan)
        n          = max(len(steps), 1)
        for i, step in enumerate(steps):
            h = _hash(step)
            if h in self._lineages:
                lin  = self._lineages[h]
                agent= lin.origin_agent
                base_score = agent_scores.get(agent, 50.0) if agent else 50.0
                lin.score_contribution = base_score / n
                lin.step_index = i

    def build_genealogy_report(
        self,
        final_plan:   str,
        session_id:   str,
        agent_scores: Dict[str, float],
    ) -> GenealogyReport:
        """Build the full genealogy report for the final plan."""
        self.compute_score_contributions(final_plan, agent_scores)
        steps = _parse_steps(final_plan)

        # Resolve lineage for each step in final plan
        lineages = []
        for i, step in enumerate(steps):
            h = _hash(step)
            if h in self._lineages:
                lin = self._lineages[h]
                lin.step_index = i
                lineages.append(lin)
            else:
                # Untracked step (added externally)
                lineages.append(StepLineage(
                    step_hash          = h,
                    step_text          = step,
                    step_index         = i,
                    origin_type        = OriginType.HUMAN_EDIT,
                    origin_agent       = None,
                    origin_session     = session_id,
                    parent_hashes      = [],
                    generation         = 0,
                    score_at_origin    = 0.0,
                    score_contribution = 0.0,
                    transformation_log = ["untracked origin"],
                ))

        # Origin distribution
        origin_dist = {}
        for lin in lineages:
            k = lin.origin_type.value
            origin_dist[k] = origin_dist.get(k, 0) + 1

        # Top contributor
        agent_counts: Dict[str, int] = {}
        for lin in lineages:
            if lin.origin_agent:
                agent_counts[lin.origin_agent] = agent_counts.get(lin.origin_agent, 0) + 1
        top_contributor = max(agent_counts, key=agent_counts.get) if agent_counts else None

        # Generation map
        gen_map: Dict[int, int] = {}
        for lin in lineages:
            gen_map[lin.generation] = gen_map.get(lin.generation, 0) + 1

        total_transforms = sum(len(lin.transformation_log) for lin in lineages)

        # Audit trail text
        trail_lines = [
            f"═══ PLAN GENEALOGY — Session {session_id} ═══",
            f"Final plan: {len(steps)} steps",
            f"Total transformations applied: {total_transforms}",
            "",
        ]
        for lin in lineages:
            agent_str = f"@{lin.origin_agent}" if lin.origin_agent else "system"
            trail_lines.append(
                f"Step {lin.step_index+1} | {lin.origin_type.value} | {agent_str} | "
                f"gen={lin.generation} | score_contrib={lin.score_contribution:.1f}"
            )
            for t in lin.transformation_log[-2:]:
                trail_lines.append(f"  → {t}")

        self._persist_lineages(lineages)

        return GenealogyReport(
            session_id           = session_id,
            final_plan           = final_plan,
            step_lineages        = lineages,
            origin_distribution  = origin_dist,
            top_contributor      = top_contributor,
            generation_map       = gen_map,
            total_transformations= total_transforms,
            audit_trail_text     = "\n".join(trail_lines),
        )

    def _persist_lineages(self, lineages: List[StepLineage]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for lin in lineages:
                conn.execute(
                    "INSERT OR REPLACE INTO step_lineages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (lin.step_hash, lin.step_text[:500], lin.step_index,
                     lin.origin_type.value, lin.origin_agent or "",
                     lin.origin_session, json.dumps(lin.parent_hashes),
                     lin.generation, lin.score_at_origin, lin.score_contribution,
                     json.dumps(lin.transformation_log), lin.timestamp)
                )
            conn.commit()

    # ── Historical Lookup ─────────────────────────────────────────────────────

    def get_agent_contribution_stats(self) -> Dict[str, Dict]:
        """Cross-session: how often does each agent's output survive to final plan?"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT origin_agent, COUNT(*), AVG(score_contribution)
                   FROM step_lineages
                   WHERE origin_type = 'agent_original' AND origin_agent != ''
                   GROUP BY origin_agent
                   ORDER BY COUNT(*) DESC"""
            ).fetchall()
        return {
            row[0]: {"total_steps_survived": row[1], "avg_contribution": round(row[2], 2)}
            for row in rows
        }

    def get_pgt_report(self, report: GenealogyReport) -> Dict:
        return {
            "session_id":          report.session_id,
            "steps_in_final_plan": len(report.step_lineages),
            "origin_distribution": report.origin_distribution,
            "top_contributor":     report.top_contributor,
            "generation_distribution": report.generation_map,
            "total_transformations":   report.total_transformations,
            "agent_contributions": {
                lin.origin_agent: lin.step_contribution
                for lin in report.step_lineages
                if lin.origin_agent
            } if False else self.get_agent_contribution_stats(),
            "lineage_summary": [
                {
                    "step":   lin.step_index + 1,
                    "origin": lin.origin_type.value,
                    "agent":  lin.origin_agent or "system",
                    "gen":    lin.generation,
                    "transforms": len(lin.transformation_log),
                }
                for lin in report.step_lineages
            ],
        }


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:12]


def _parse_steps(plan_text: str) -> List[str]:
    lines = plan_text.strip().split('\n')
    steps = []
    for line in lines:
        clean = re.sub(r'^\s*(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '',
                       line, flags=re.IGNORECASE).strip()
        if clean and len(clean) > 5:
            steps.append(clean)
    return steps or [plan_text]
