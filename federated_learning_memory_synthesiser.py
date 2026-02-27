# ═══════════════════════════════════════════════════════════════════════════════
# federated_learning_memory_synthesiser.py — Feature 15: FLMS
# Aggregates learning across user sessions using privacy-preserving federated
# averaging to improve agent selection and strategy extraction over time.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import hashlib
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

FLMS_DB_PATH              = "federated_learning.db"
FLMS_MIN_SESSIONS_TO_LEARN= 5      # need this many sessions before aggregating
FLMS_AGGREGATION_INTERVAL = 10     # aggregate every N new sessions
FLMS_EMA_ALPHA            = 0.25   # exponential moving average weight
FLMS_NOISE_EPSILON        = 0.02   # differential privacy noise level (σ)
FLMS_MAX_STRATEGY_PATTERNS= 50     # max stored strategy patterns
FLMS_AGENT_PROFILE_DECAY  = 0.95   # per-session decay for agent scores


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class LocalUpdate:
    """
    Privacy-safe gradient from one session.
    Contains only aggregated statistics, no raw plan text.
    """
    session_hash:      str           # SHA256(session_id) — not reversible
    agent_deltas:      Dict[str, float]  # agent → score delta from baseline
    step_count_delta:  float
    domain_tag:        str
    feature_flags:     Dict[str, bool]   # which features were active
    timestamp:         str


@dataclass
class GlobalModel:
    """Aggregated model across all sessions."""
    agent_reliability_scores: Dict[str, float]   # 0–100, higher = more reliable
    agent_domain_strengths:   Dict[str, Dict[str, float]]  # agent → domain → score
    optimal_step_counts:      Dict[str, float]    # domain → optimal step count
    strategy_patterns:        List[Dict]          # high-frequency winning strategies
    session_count:            int
    last_aggregated:          str
    version:                  int


@dataclass
class SynthesisReport:
    global_model:           GlobalModel
    top_agents_by_domain:   Dict[str, List[str]]    # domain → ranked agents
    recommended_agents:     List[str]                # for current session
    strategy_injections:    List[str]                # patterns to inject into prompts
    confidence_level:       float                    # 0–1 based on session count
    privacy_budget_spent:   float                    # differential privacy epsilon used


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_FLMS_PATTERN_EXTRACTOR = """You are a STRATEGY PATTERN EXTRACTOR for federated learning.

You have access to ANONYMISED summaries of {n_sessions} planning sessions.
Extract recurring high-level strategy patterns (NOT specific plan text).

SESSION STATISTICS:
{session_stats_json}

Extract patterns that appear in HIGH-SCORING sessions (score > {score_threshold}).
Focus on:
  • Structural patterns (how plans are organised)
  • Sequencing patterns (what always comes before what)
  • Resource patterns (what resources high-scoring plans allocate first)
  • Risk patterns (how successful plans handle uncertainty)

DO NOT include any plan text or PII. Abstract patterns only.

Respond ONLY with valid JSON:
{{"patterns": [
  {{
    "pattern_id": "p_001",
    "pattern_name": "Risk-First Sequencing",
    "pattern_description": "...",
    "applicable_domains": ["technology", "business"],
    "avg_score_lift": 0.0,
    "frequency": 0,
    "abstracted_template": "..."
  }}
]}}"""

PROMPT_FLMS_AGENT_PROFILER = """You are an AGENT PERFORMANCE ANALYST.

ANONYMISED AGENT PERFORMANCE DATA (last {n_sessions} sessions):
{agent_stats_json}

Analyse the data to determine:
  1. Which agents consistently outperform the mean?
  2. Which agents excel in specific domains?
  3. Which agents show improving trends vs plateauing?
  4. Are there agent combinations that produce better consensus?

Respond ONLY with valid JSON:
{{"agent_insights": [
  {{
    "agent": "...",
    "reliability_score": 0.0,
    "trend": "improving|stable|declining",
    "strength_domains": ["domain_1"],
    "weakness_domains": ["domain_2"],
    "recommended_role": "planner|verifier|specialist|generalist"
  }}
]}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class FederatedLearningMemorySynthesiser:
    """
    Simulates federated learning over planning sessions:
    1. Each session produces a privacy-safe LocalUpdate (no raw text)
    2. Every N sessions, FedAvg aggregates updates into GlobalModel
    3. GlobalModel enriches next session's agent selection + prompt injection

    Privacy: Gaussian noise added to score deltas (differential privacy).
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        db_path:  str = FLMS_DB_PATH,
        agent:    str = "gemini",
    ):
        self.call_fn = call_fn
        self.db_path = db_path
        self.agent   = agent
        self._init_db()
        self._global_model: Optional[GlobalModel] = self._load_global_model()

    # ── DB Setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_updates (
                    session_hash   TEXT PRIMARY KEY,
                    agent_deltas   TEXT NOT NULL,
                    step_count_delta REAL,
                    domain_tag     TEXT,
                    feature_flags  TEXT,
                    timestamp      TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_models (
                    version        INTEGER PRIMARY KEY,
                    model_json     TEXT NOT NULL,
                    session_count  INTEGER,
                    timestamp      TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_patterns (
                    pattern_id     TEXT PRIMARY KEY,
                    pattern_json   TEXT NOT NULL,
                    frequency      INTEGER DEFAULT 1,
                    avg_score_lift REAL DEFAULT 0.0,
                    timestamp      TEXT
                )
            """)
            conn.commit()
        logger.info(f"[FLMS] Database initialised: {self.db_path}")

    # ── Local Update (per session) ────────────────────────────────────────────

    def create_local_update(
        self,
        session_id:    str,
        agent_scores:  Dict[str, float],
        baseline_score:float,
        step_count:    int,
        domain:        str,
        feature_flags: Optional[Dict[str, bool]] = None,
    ) -> LocalUpdate:
        """
        Create a privacy-safe local update from one session's results.
        Applies Gaussian noise (differential privacy) to score deltas.
        """
        # Score deltas vs session baseline
        raw_deltas = {
            agent: score - baseline_score
            for agent, score in agent_scores.items()
        }

        # Add Gaussian noise for differential privacy
        noisy_deltas = {
            agent: delta + self._gaussian_noise(FLMS_NOISE_EPSILON)
            for agent, delta in raw_deltas.items()
        }

        update = LocalUpdate(
            session_hash     = hashlib.sha256(session_id.encode()).hexdigest()[:16],
            agent_deltas     = noisy_deltas,
            step_count_delta = float(step_count),
            domain_tag       = domain,
            feature_flags    = feature_flags or {},
            timestamp        = datetime.utcnow().isoformat(),
        )
        self._store_local_update(update)
        return update

    def _store_local_update(self, update: LocalUpdate) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO local_updates VALUES (?,?,?,?,?,?)",
                (update.session_hash, json.dumps(update.agent_deltas),
                 update.step_count_delta, update.domain_tag,
                 json.dumps(update.feature_flags), update.timestamp)
            )
            conn.commit()

    # ── FedAvg Aggregation ────────────────────────────────────────────────────

    def federated_average(
        self,
        updates: List[LocalUpdate],
    ) -> Dict[str, float]:
        """
        FedAvg: weighted average of agent deltas across all updates.
        Weight = 1/n (uniform — no client weighting).
        """
        if not updates:
            return {}

        agent_sums:   Dict[str, float] = {}
        agent_counts: Dict[str, int]   = {}

        for update in updates:
            for agent, delta in update.agent_deltas.items():
                agent_sums[agent]   = agent_sums.get(agent, 0.0) + delta
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            agent: agent_sums[agent] / agent_counts[agent]
            for agent in agent_sums
        }

    def should_aggregate(self) -> bool:
        """True when enough new sessions have accumulated."""
        with sqlite3.connect(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM local_updates").fetchone()[0]
        last_v = self._global_model.session_count if self._global_model else 0
        return (n - last_v) >= FLMS_AGGREGATION_INTERVAL and n >= FLMS_MIN_SESSIONS_TO_LEARN

    # ── Global Model Update ───────────────────────────────────────────────────

    async def aggregate_global_model(self) -> GlobalModel:
        """
        Full FedAvg aggregation pipeline:
        1. Load all local updates
        2. FedAvg → agent reliability scores
        3. Domain-specific aggregation
        4. LLM pattern extraction from anonymised stats
        5. Persist new global model
        """
        updates   = self._load_all_updates()
        if len(updates) < FLMS_MIN_SESSIONS_TO_LEARN:
            logger.info(f"[FLMS] Only {len(updates)} sessions. Skipping aggregation.")
            return self._global_model or self._empty_global_model()

        # FedAvg for global agent scores
        avg_deltas = self.federated_average(updates)

        # Initialise from existing model or baseline
        existing = self._global_model or self._empty_global_model()
        new_scores: Dict[str, float] = {}
        for agent, delta in avg_deltas.items():
            prev = existing.agent_reliability_scores.get(agent, 50.0)
            # EMA update
            new_scores[agent] = FLMS_EMA_ALPHA * (prev + delta) + \
                                 (1 - FLMS_EMA_ALPHA) * prev
            new_scores[agent] = max(0.0, min(100.0, new_scores[agent]))

        # Domain-specific aggregation
        domain_strengths = self._compute_domain_strengths(updates, avg_deltas)

        # Optimal step counts per domain
        optimal_steps = self._compute_optimal_steps(updates)

        # LLM pattern extraction
        patterns = await self._extract_patterns(updates)

        new_model = GlobalModel(
            agent_reliability_scores = new_scores,
            agent_domain_strengths   = domain_strengths,
            optimal_step_counts      = optimal_steps,
            strategy_patterns        = patterns,
            session_count            = len(updates),
            last_aggregated          = datetime.utcnow().isoformat(),
            version                  = (existing.version + 1),
        )
        self._persist_global_model(new_model)
        self._global_model = new_model

        logger.info(
            f"[FLMS] Global model v{new_model.version} aggregated from "
            f"{len(updates)} sessions. Top agents: "
            f"{sorted(new_scores, key=new_scores.get, reverse=True)[:3]}"
        )
        return new_model

    def _compute_domain_strengths(
        self,
        updates:    List[LocalUpdate],
        avg_deltas: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Per-domain agent score aggregation."""
        domain_data: Dict[str, Dict[str, List[float]]] = {}
        for update in updates:
            d = update.domain_tag or "general"
            if d not in domain_data:
                domain_data[d] = {}
            for agent, delta in update.agent_deltas.items():
                domain_data[d].setdefault(agent, []).append(delta)

        result: Dict[str, Dict[str, float]] = {}
        for domain, agents in domain_data.items():
            result[domain] = {
                agent: sum(scores) / len(scores)
                for agent, scores in agents.items()
            }
        return result

    def _compute_optimal_steps(
        self,
        updates: List[LocalUpdate],
    ) -> Dict[str, float]:
        """Domain → median optimal step count from high-scoring sessions."""
        domain_steps: Dict[str, List[float]] = {}
        for update in updates:
            d = update.domain_tag or "general"
            domain_steps.setdefault(d, []).append(update.step_count_delta)
        return {
            domain: statistics.median(counts)
            for domain, counts in domain_steps.items()
        }

    async def _extract_patterns(
        self,
        updates: List[LocalUpdate],
    ) -> List[Dict]:
        """LLM extracts strategy patterns from anonymised stats."""
        # Build anonymised stats (no raw text)
        stats = {
            "total_sessions":  len(updates),
            "domain_breakdown":{},
            "feature_usage":   {},
            "agent_win_rates": {},
        }
        for update in updates:
            d = update.domain_tag or "general"
            stats["domain_breakdown"][d] = stats["domain_breakdown"].get(d, 0) + 1
            for flag, val in update.feature_flags.items():
                if val:
                    stats["feature_usage"][flag] = stats["feature_usage"].get(flag, 0) + 1
            top_agent = max(update.agent_deltas, key=update.agent_deltas.get, default=None)
            if top_agent:
                stats["agent_win_rates"][top_agent] = stats["agent_win_rates"].get(top_agent, 0) + 1

        prompt = PROMPT_FLMS_PATTERN_EXTRACTOR.format(
            n_sessions       = len(updates),
            session_stats_json= json.dumps(stats, indent=2)[:1500],
            score_threshold  = 70,
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return data.get("patterns", [])[:FLMS_MAX_STRATEGY_PATTERNS]
        except Exception as e:
            logger.warning(f"[FLMS] Pattern extraction failed: {e}")
            return []

    # ── Inference (per session) ───────────────────────────────────────────────

    def get_synthesis_for_session(
        self,
        aim:    str,
        domain: str,
        n_top_agents: int = 4,
    ) -> SynthesisReport:
        """
        Returns a SynthesisReport that enriches the current session:
        - Top agents to prioritise
        - Strategy patterns to inject into prompts
        - Confidence level based on session count
        """
        model = self._global_model or self._empty_global_model()

        # Domain-specific agent ranking
        domain_scores = model.agent_domain_strengths.get(
            domain,
            model.agent_reliability_scores
        )
        ranked_agents = sorted(
            domain_scores, key=domain_scores.get, reverse=True
        )

        # Top agents for this domain
        top_agents = ranked_agents[:n_top_agents]

        # Strategy injections — top 3 patterns
        injections = [
            f"Strategy pattern [{p.get('pattern_name', 'Unknown')}]: "
            f"{p.get('abstracted_template', '')}"
            for p in model.strategy_patterns[:3]
            if domain in p.get("applicable_domains", ["general"])
        ]

        # Confidence: sigmoid of session count
        raw_confidence = 1 / (1 + math.exp(-0.1 * (model.session_count - 20)))
        confidence     = max(0.1, min(0.95, raw_confidence))

        all_domain_rankings = {
            dom: sorted(agents, key=agents.get, reverse=True)[:4]
            for dom, agents in model.agent_domain_strengths.items()
        }

        return SynthesisReport(
            global_model         = model,
            top_agents_by_domain = all_domain_rankings,
            recommended_agents   = top_agents,
            strategy_injections  = injections,
            confidence_level     = confidence,
            privacy_budget_spent = FLMS_NOISE_EPSILON * model.session_count,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_noise(sigma: float) -> float:
        """Box-Muller transform for Gaussian noise."""
        import random
        u1 = max(1e-10, random.random())
        u2 = random.random()
        return sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def _load_all_updates(self) -> List[LocalUpdate]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM local_updates ORDER BY timestamp"
            ).fetchall()
        result = []
        for row in rows:
            try:
                result.append(LocalUpdate(
                    session_hash     = row[0],
                    agent_deltas     = json.loads(row[1]),
                    step_count_delta = row[2],
                    domain_tag       = row[3] or "general",
                    feature_flags    = json.loads(row[4]) if row[4] else {},
                    timestamp        = row[5],
                ))
            except Exception:
                continue
        return result

    def _load_global_model(self) -> Optional[GlobalModel]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT model_json, session_count FROM global_models ORDER BY version DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return GlobalModel(
                agent_reliability_scores = data.get("agent_reliability_scores", {}),
                agent_domain_strengths   = data.get("agent_domain_strengths", {}),
                optimal_step_counts      = data.get("optimal_step_counts", {}),
                strategy_patterns        = data.get("strategy_patterns", []),
                session_count            = row[1],
                last_aggregated          = data.get("last_aggregated", ""),
                version                  = data.get("version", 1),
            )
        except Exception:
            return None

    def _persist_global_model(self, model: GlobalModel) -> None:
        model_json = json.dumps({
            "agent_reliability_scores": model.agent_reliability_scores,
            "agent_domain_strengths":   model.agent_domain_strengths,
            "optimal_step_counts":      model.optimal_step_counts,
            "strategy_patterns":        model.strategy_patterns,
            "last_aggregated":          model.last_aggregated,
            "version":                  model.version,
        })
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO global_models VALUES (?,?,?,?)",
                (model.version, model_json, model.session_count,
                 datetime.utcnow().isoformat())
            )
            conn.commit()

    def _empty_global_model(self) -> GlobalModel:
        return GlobalModel(
            agent_reliability_scores = {},
            agent_domain_strengths   = {},
            optimal_step_counts      = {},
            strategy_patterns        = [],
            session_count            = 0,
            last_aggregated          = datetime.utcnow().isoformat(),
            version                  = 0,
        )

    def get_flms_report(self) -> Dict:
        model = self._global_model or self._empty_global_model()
        return {
            "global_model_version":   model.version,
            "total_sessions_learned": model.session_count,
            "last_aggregated":        model.last_aggregated,
            "top_agents_global":      sorted(
                model.agent_reliability_scores,
                key=model.agent_reliability_scores.get,
                reverse=True
            )[:5],
            "domains_tracked":        list(model.agent_domain_strengths.keys()),
            "strategy_patterns_found":len(model.strategy_patterns),
            "privacy_noise_level":    FLMS_NOISE_EPSILON,
            "agent_scores":           {
                k: round(v, 1)
                for k, v in sorted(
                    model.agent_reliability_scores.items(),
                    key=lambda x: x[1], reverse=True
                )
            },
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
