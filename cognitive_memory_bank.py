# ═══════════════════════════════════════════════════════════════════════════════
# cognitive_memory_bank.py — Feature 2: Cognitive Memory Bank (CMB)
# 3-tier persistent memory: working / episodic / semantic
# ═══════════════════════════════════════════════════════════════════════════════

import json
import math
import sqlite3
import logging
import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from system_config import (
    CMB_DB_PATH, CMB_RETRIEVAL_TOP_K, CMB_MIN_SIMILARITY,
    CMB_EMA_ALPHA, CMB_DISTILLATION_INTERVAL, CMB_CACHE_MAX_SIZE
)

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class EpisodicRecord:
    """One complete planning session stored in long-term memory."""
    session_id:      str
    aim:             str
    initial_steps:   List[str]
    best_plan:       str
    winning_agent:   str
    winning_score:   float
    all_scores:      Dict[str, float]
    aim_embedding:   List[float]      # 768-dim float vector
    domain_tags:     List[str]
    feature_count:   int
    timestamp:       str
    param_file_hash: str              # MD5 of parameters used
    step_count:      int              = 0
    improvement_delta: float          = 0.0  # score gain from improvement chain


@dataclass
class AgentProfile:
    """Per-agent performance knowledge accumulated across sessions."""
    agent_name:            str
    total_sessions:        int             = 0
    mean_score:            float           = 0.0
    score_variance:        float           = 0.0
    domain_scores:         Dict[str, float] = field(default_factory=dict)
    best_parameter_combos: List[str]       = field(default_factory=list)
    common_failure_modes:  List[str]       = field(default_factory=list)
    avg_token_cost:        int             = 0
    win_count:             int             = 0
    last_updated:          str             = ""


@dataclass
class StrategyPattern:
    """Distilled generalizable planning insight."""
    pattern_id:           str
    pattern_text:         str
    frequency:            int
    avg_score_with:       float
    avg_score_without:    float
    applicable_domains:   List[str]
    evidence_session_ids: List[str]
    created_at:           str


# ── Main Class ───────────────────────────────────────────────────────────────

class CognitiveMemoryBank:
    """
    3-tier memory system for the AI Planner.

    Tier 1 — Working Memory:   current session (handled by caller)
    Tier 2 — Episodic Memory:  SQLite table of full session records
    Tier 3 — Semantic Memory:  distilled agent profiles + strategy patterns

    Primary entry points:
        get_memory_priming_context(aim)  → inject before step1 prompts
        store_episodic(record)           → persist session after completion
        update_agent_profile(...)        → incremental profile update
    """

    _SCHEMA_EPISODIC = """
    CREATE TABLE IF NOT EXISTS episodic_records (
        session_id      TEXT PRIMARY KEY,
        aim             TEXT NOT NULL,
        initial_steps   TEXT NOT NULL,   -- JSON array
        best_plan       TEXT NOT NULL,
        winning_agent   TEXT NOT NULL,
        winning_score   REAL NOT NULL,
        all_scores      TEXT NOT NULL,   -- JSON object
        aim_embedding   TEXT NOT NULL,   -- JSON array of floats
        domain_tags     TEXT NOT NULL,   -- JSON array
        feature_count   INTEGER DEFAULT 0,
        timestamp       TEXT NOT NULL,
        param_file_hash TEXT NOT NULL,
        step_count      INTEGER DEFAULT 0,
        improvement_delta REAL DEFAULT 0.0
    )"""

    _SCHEMA_PROFILES = """
    CREATE TABLE IF NOT EXISTS agent_profiles (
        agent_name            TEXT PRIMARY KEY,
        total_sessions        INTEGER DEFAULT 0,
        mean_score            REAL DEFAULT 0.0,
        score_variance        REAL DEFAULT 0.0,
        domain_scores         TEXT DEFAULT '{}',
        best_parameter_combos TEXT DEFAULT '[]',
        common_failure_modes  TEXT DEFAULT '[]',
        avg_token_cost        INTEGER DEFAULT 0,
        win_count             INTEGER DEFAULT 0,
        last_updated          TEXT DEFAULT ''
    )"""

    _SCHEMA_PATTERNS = """
    CREATE TABLE IF NOT EXISTS strategy_patterns (
        pattern_id           TEXT PRIMARY KEY,
        pattern_text         TEXT NOT NULL,
        frequency            INTEGER DEFAULT 1,
        avg_score_with       REAL DEFAULT 0.0,
        avg_score_without    REAL DEFAULT 0.0,
        applicable_domains   TEXT DEFAULT '[]',
        evidence_session_ids TEXT DEFAULT '[]',
        created_at           TEXT NOT NULL
    )"""

    def __init__(self, db_path: str = CMB_DB_PATH, embedding_fn: Optional[callable] = None):
        """
        Args:
            db_path:      Path to SQLite database file.
            embedding_fn: Optional async callable(text: str) -> List[float].
                          If None, falls back to a deterministic mock embedding.
        """
        self.db_path      = db_path
        self.embedding_fn = embedding_fn
        self._lock        = threading.Lock()
        self._embed_cache: Dict[str, List[float]] = {}   # LRU manually managed
        self._init_schema()
        logger.info(f"[CMB] Initialized. DB: {db_path}")

    # ── Schema ───────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA_EPISODIC)
            conn.execute(self._SCHEMA_PROFILES)
            conn.execute(self._SCHEMA_PATTERNS)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Embeddings ───────────────────────────────────────────────────────────

    async def get_aim_embedding(self, text: str) -> List[float]:
        """
        Returns a 768-dim embedding for `text`.
        Uses embedding_fn if provided (e.g. Gemini API), otherwise
        falls back to a deterministic TF-IDF-style hash embedding.
        Results are cached in _embed_cache.
        """
        key = hashlib.md5(text.encode()).hexdigest()

        if key in self._embed_cache:
            return self._embed_cache[key]

        if self.embedding_fn:
            try:
                vec = await self.embedding_fn(text)
                embedding = vec if isinstance(vec, list) else list(vec)
            except Exception as e:
                logger.warning(f"[CMB] Embedding API failed: {e}. Using fallback.")
                embedding = self._fallback_embedding(text)
        else:
            embedding = self._fallback_embedding(text)

        # LRU eviction when cache is full
        if len(self._embed_cache) >= CMB_CACHE_MAX_SIZE:
            oldest_key = next(iter(self._embed_cache))
            del self._embed_cache[oldest_key]

        self._embed_cache[key] = embedding
        return embedding

    def _fallback_embedding(self, text: str, dims: int = 768) -> List[float]:
        """
        Deterministic pseudo-embedding using character-level hash seeding.
        Not semantically meaningful but consistent across sessions.
        """
        words = text.lower().split()
        vec = [0.0] * dims
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % dims
            vec[idx] += 1.0 / (i + 1)   # position-weighted
        # L2 normalise
        norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Fast cosine similarity between two equal-length float vectors."""
        dot  = sum(ai * bi for ai, bi in zip(a, b))
        na   = math.sqrt(sum(ai ** 2 for ai in a))
        nb   = math.sqrt(sum(bi ** 2 for bi in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ── Episodic Memory ───────────────────────────────────────────────────────

    def store_episodic(self, record: EpisodicRecord) -> str:
        """Persist a completed session to episodic memory. Returns session_id."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO episodic_records VALUES (
                        :session_id, :aim, :initial_steps, :best_plan,
                        :winning_agent, :winning_score, :all_scores,
                        :aim_embedding, :domain_tags, :feature_count,
                        :timestamp, :param_file_hash, :step_count, :improvement_delta
                    )
                """, {
                    "session_id":       record.session_id,
                    "aim":              record.aim,
                    "initial_steps":    json.dumps(record.initial_steps),
                    "best_plan":        record.best_plan,
                    "winning_agent":    record.winning_agent,
                    "winning_score":    record.winning_score,
                    "all_scores":       json.dumps(record.all_scores),
                    "aim_embedding":    json.dumps(record.aim_embedding),
                    "domain_tags":      json.dumps(record.domain_tags),
                    "feature_count":    record.feature_count,
                    "timestamp":        record.timestamp,
                    "param_file_hash":  record.param_file_hash,
                    "step_count":       record.step_count,
                    "improvement_delta":record.improvement_delta,
                })
                conn.commit()
        logger.info(f"[CMB] Stored episodic record: {record.session_id} (score={record.winning_score:.1f})")
        return record.session_id

    async def retrieve_similar_sessions(
        self,
        aim: str,
        top_k: int       = CMB_RETRIEVAL_TOP_K,
        min_sim: float   = CMB_MIN_SIMILARITY,
    ) -> List[Tuple[EpisodicRecord, float]]:
        """
        Embed the query aim, compute cosine similarity against all stored
        embeddings, return top-k above min_sim threshold.
        """
        query_emb = await self.get_aim_embedding(aim)

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM episodic_records ORDER BY timestamp DESC").fetchall()

        results: List[Tuple[EpisodicRecord, float]] = []
        for row in rows:
            stored_emb = json.loads(row["aim_embedding"])
            sim = self.cosine_similarity(query_emb, stored_emb)
            if sim >= min_sim:
                rec = EpisodicRecord(
                    session_id       = row["session_id"],
                    aim              = row["aim"],
                    initial_steps    = json.loads(row["initial_steps"]),
                    best_plan        = row["best_plan"],
                    winning_agent    = row["winning_agent"],
                    winning_score    = row["winning_score"],
                    all_scores       = json.loads(row["all_scores"]),
                    aim_embedding    = stored_emb,
                    domain_tags      = json.loads(row["domain_tags"]),
                    feature_count    = row["feature_count"],
                    timestamp        = row["timestamp"],
                    param_file_hash  = row["param_file_hash"],
                    step_count       = row["step_count"],
                    improvement_delta= row["improvement_delta"],
                )
                results.append((rec, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]
        logger.info(f"[CMB] Retrieved {len(top)} similar sessions for aim: '{aim[:60]}...'")
        return top

    # ── Semantic Memory — Agent Profiles ─────────────────────────────────────

    def update_agent_profile(
        self,
        agent_name:   str,
        session_score: float,
        domain:       str,
        token_cost:   int,
        is_winner:    bool = False,
        failure_mode: Optional[str] = None,
        top_params:   Optional[List[str]] = None,
    ) -> None:
        """
        Incremental EMA update of AgentProfile.
        alpha = CMB_EMA_ALPHA → recent sessions weighted more than older ones.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM agent_profiles WHERE agent_name = ?", (agent_name,)
                ).fetchone()

                if row:
                    n          = row["total_sessions"]
                    old_mean   = row["mean_score"]
                    old_var    = row["score_variance"]
                    old_cost   = row["avg_token_cost"]
                    domain_sc  = json.loads(row["domain_scores"])
                    best_params= json.loads(row["best_parameter_combos"])
                    fail_modes = json.loads(row["common_failure_modes"])
                    wins       = row["win_count"]

                    # EMA updates
                    new_mean = (1 - CMB_EMA_ALPHA) * old_mean + CMB_EMA_ALPHA * session_score
                    new_var  = (1 - CMB_EMA_ALPHA) * old_var  + CMB_EMA_ALPHA * (session_score - new_mean) ** 2
                    new_cost = int((1 - CMB_EMA_ALPHA) * old_cost + CMB_EMA_ALPHA * token_cost)

                    # Domain-level EMA
                    domain_sc[domain] = (
                        (1 - CMB_EMA_ALPHA) * domain_sc.get(domain, session_score)
                        + CMB_EMA_ALPHA * session_score
                    )

                    # Top params: keep most recent 5
                    if top_params:
                        best_params = (top_params + best_params)[:5]

                    # Failure modes: keep most recent 5
                    if failure_mode and failure_mode not in fail_modes:
                        fail_modes = ([failure_mode] + fail_modes)[:5]

                    conn.execute("""
                        UPDATE agent_profiles SET
                            total_sessions = ?,
                            mean_score     = ?,
                            score_variance = ?,
                            domain_scores  = ?,
                            best_parameter_combos = ?,
                            common_failure_modes  = ?,
                            avg_token_cost = ?,
                            win_count      = ?,
                            last_updated   = ?
                        WHERE agent_name = ?
                    """, (
                        n + 1, new_mean, new_var,
                        json.dumps(domain_sc), json.dumps(best_params), json.dumps(fail_modes),
                        new_cost, wins + (1 if is_winner else 0),
                        _now(), agent_name
                    ))
                else:
                    # First record for this agent
                    ds = {domain: session_score}
                    conn.execute("""
                        INSERT INTO agent_profiles VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        agent_name, 1, session_score, 0.0,
                        json.dumps(ds),
                        json.dumps(top_params or []),
                        json.dumps([failure_mode] if failure_mode else []),
                        token_cost, 1 if is_winner else 0, _now()
                    ))
                conn.commit()

    def get_agent_profile(self, agent_name: str) -> Optional[AgentProfile]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_profiles WHERE agent_name = ?", (agent_name,)
            ).fetchone()
            if not row:
                return None
            return AgentProfile(
                agent_name            = row["agent_name"],
                total_sessions        = row["total_sessions"],
                mean_score            = row["mean_score"],
                score_variance        = row["score_variance"],
                domain_scores         = json.loads(row["domain_scores"]),
                best_parameter_combos = json.loads(row["best_parameter_combos"]),
                common_failure_modes  = json.loads(row["common_failure_modes"]),
                avg_token_cost        = row["avg_token_cost"],
                win_count             = row["win_count"],
                last_updated          = row["last_updated"],
            )

    def get_ranked_agents(self, domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Return agents ranked by mean score (or domain score if domain given).
        Returns [(agent_name, score), ...] sorted descending.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agent_profiles").fetchall()

        ranked = []
        for row in rows:
            if domain:
                ds = json.loads(row["domain_scores"])
                score = ds.get(domain, row["mean_score"])
            else:
                score = row["mean_score"]
            ranked.append((row["agent_name"], score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ── Semantic Memory — Strategy Patterns ──────────────────────────────────

    def store_strategy_patterns(self, patterns: List[StrategyPattern]) -> None:
        with self._lock:
            with self._connect() as conn:
                for p in patterns:
                    conn.execute("""
                        INSERT OR REPLACE INTO strategy_patterns VALUES (?,?,?,?,?,?,?,?)
                    """, (
                        p.pattern_id, p.pattern_text, p.frequency,
                        p.avg_score_with, p.avg_score_without,
                        json.dumps(p.applicable_domains),
                        json.dumps(p.evidence_session_ids),
                        p.created_at
                    ))
                conn.commit()
        logger.info(f"[CMB] Stored {len(patterns)} strategy patterns")

    def get_strategy_patterns(self, domain: Optional[str] = None) -> List[StrategyPattern]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_patterns ORDER BY avg_score_with DESC"
            ).fetchall()

        patterns = []
        for row in rows:
            domains = json.loads(row["applicable_domains"])
            if domain and domain not in domains and "all" not in domains:
                continue
            patterns.append(StrategyPattern(
                pattern_id           = row["pattern_id"],
                pattern_text         = row["pattern_text"],
                frequency            = row["frequency"],
                avg_score_with       = row["avg_score_with"],
                avg_score_without    = row["avg_score_without"],
                applicable_domains   = domains,
                evidence_session_ids = json.loads(row["evidence_session_ids"]),
                created_at           = row["created_at"],
            ))
        return patterns

    def should_distill(self) -> bool:
        """True if enough new sessions have accumulated since last distillation."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodic_records").fetchone()[0]
        return total > 0 and total % CMB_DISTILLATION_INTERVAL == 0

    # ── Core Injection ────────────────────────────────────────────────────────

    async def get_memory_priming_context(
        self,
        aim:     str,
        top_k:   int = CMB_RETRIEVAL_TOP_K,
        domain:  Optional[str] = None,
    ) -> str:
        """
        Build the priming string to prepend to every step1 prompt.
        Aggregates: similar past sessions + agent priority ranking +
        top parameters + strategy patterns + failure flags.
        """
        similar   = await self.retrieve_similar_sessions(aim, top_k)
        ranked    = self.get_ranked_agents(domain)
        patterns  = self.get_strategy_patterns(domain)

        if not similar and not ranked and not patterns:
            return ""   # No memory yet — first session

        lines = ["═" * 60, "COGNITIVE MEMORY CONTEXT (Do NOT reproduce — use as priming)", "═" * 60]

        # Similar sessions
        if similar:
            lines.append(f"\n📚 TOP {len(similar)} SIMILAR PAST SESSIONS:")
            for rec, sim in similar:
                lines.append(
                    f"  • [{sim:.0%} match] AIM: {rec.aim[:80]}\n"
                    f"    Winner: {rec.winning_agent} (score {rec.winning_score:.1f}) "
                    f"| Improvement delta: +{rec.improvement_delta:.1f}\n"
                    f"    Best plan excerpt: {rec.best_plan[:200]}..."
                )

        # Agent priority
        if ranked:
            lines.append(f"\n🤖 AGENT PRIORITY ORDER (for domain '{domain or 'general'}'):")
            for i, (name, score) in enumerate(ranked[:6], 1):
                lines.append(f"  {i}. {name} — avg score {score:.1f}")

        # Top parameters
        all_top_params: Dict[str, int] = {}
        for rec, _ in similar:
            profile = self.get_agent_profile(rec.winning_agent)
            if profile:
                for p in profile.best_parameter_combos:
                    all_top_params[p] = all_top_params.get(p, 0) + 1
        if all_top_params:
            top5 = sorted(all_top_params, key=all_top_params.get, reverse=True)[:5]
            lines.append(f"\n⚡ TOP EFFECTIVE PARAMETERS: {', '.join(top5)}")

        # Strategy patterns
        if patterns:
            lines.append(f"\n💡 DISTILLED STRATEGY PATTERNS (apply these):")
            for p in patterns[:5]:
                delta = p.avg_score_with - p.avg_score_without
                lines.append(f"  • {p.pattern_text} [+{delta:.1f} score delta]")

        # Failure flags
        all_failures: List[str] = []
        for rec, _ in similar:
            profile = self.get_agent_profile(rec.winning_agent)
            if profile:
                all_failures.extend(profile.common_failure_modes)
        if all_failures:
            lines.append(f"\n⚠️  KNOWN FAILURE MODES TO AVOID:")
            for f in list(dict.fromkeys(all_failures))[:4]:
                lines.append(f"  ✗ {f}")

        best_hist = max((rec.winning_score for rec, _ in similar), default=0)
        if best_hist:
            lines.append(f"\n🎯 HISTORICAL BEST SCORE: {best_hist:.1f} — surpass this.")

        lines.append("═" * 60)
        return "\n".join(lines)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_session_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodic_records").fetchone()[0]

    def export_memory_snapshot(self, output_path: str) -> Dict:
        """Export full memory state to JSON for backup/transfer."""
        with self._connect() as conn:
            episodes = [dict(r) for r in conn.execute("SELECT * FROM episodic_records").fetchall()]
            profiles = [dict(r) for r in conn.execute("SELECT * FROM agent_profiles").fetchall()]
            patterns = [dict(r) for r in conn.execute("SELECT * FROM strategy_patterns").fetchall()]

        snapshot = {
            "exported_at":     _now(),
            "session_count":   len(episodes),
            "agent_count":     len(profiles),
            "pattern_count":   len(patterns),
            "episodic_records":episodes,
            "agent_profiles":  profiles,
            "strategy_patterns":patterns,
        }
        with open(output_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"[CMB] Exported snapshot: {output_path} ({len(episodes)} sessions)")
        return snapshot


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
