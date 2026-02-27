# ═══════════════════════════════════════════════════════════════════════════════
# dynamic_knowledge_graph_engine.py — Feature 11: DKGE
# Builds a live knowledge graph of entities, relationships, and domain concepts
# extracted from every plan. Enables semantic cross-session retrieval.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Awaitable

logger = logging.getLogger(__name__)

DKGE_DB_PATH             = "knowledge_graph.db"
DKGE_MAX_ENTITIES_PER_PLAN = 30
DKGE_MIN_EDGE_CONFIDENCE  = 0.60
DKGE_MAX_PATH_DEPTH       = 5
DKGE_DECAY_RATE           = 0.02   # edge weight decays per session since last seen


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class KGNode:
    entity_id:     str
    label:         str          # e.g. "market_research"
    entity_type:   str          # concept | action | resource | risk | outcome
    description:   str
    frequency:     int  = 1
    domains:       List[str] = field(default_factory=list)
    first_seen:    str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen:     str = field(default_factory=lambda: datetime.utcnow().isoformat())
    plan_contexts: List[str] = field(default_factory=list)  # session_ids


@dataclass
class KGEdge:
    edge_id:         str
    source_id:       str
    target_id:       str
    relation_type:   str    # enables | blocks | requires | produces | mitigates
    weight:          float  # 0–1 confidence
    evidence_count:  int    = 1
    last_seen:       str    = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GraphQueryResult:
    query:          str
    matched_nodes:  List[KGNode]
    subgraph_edges: List[KGEdge]
    shortest_paths: List[List[str]]
    domain_clusters: Dict[str, List[str]]
    relevance_scores: Dict[str, float]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_DKGE_EXTRACTOR = """You are a KNOWLEDGE GRAPH EXTRACTION SPECIALIST.

AIM: {aim}
PLAN:
{plan_text}

Extract all entities and relationships to build a knowledge graph.

Entity types:
  concept  — abstract ideas, methodologies, frameworks
  action   — tasks, activities, operations performed
  resource — tools, people, budget, infrastructure
  risk     — threats, blockers, uncertainties
  outcome  — deliverables, results, milestones

Relationship types:
  enables  — entity A makes entity B possible
  blocks   — entity A prevents entity B
  requires — entity A needs entity B to function
  produces — entity A creates entity B as output
  mitigates— entity A reduces risk of entity B

Respond ONLY with valid JSON (no markdown):
{{
  "entities": [
    {{
      "entity_id": "e_market_research",
      "label": "market_research",
      "entity_type": "action",
      "description": "Systematic analysis of target market segments",
      "domains": ["business", "strategy"]
    }}
  ],
  "relationships": [
    {{
      "source_id": "e_market_research",
      "target_id": "e_product_strategy",
      "relation_type": "enables",
      "weight": 0.85,
      "rationale": "Research findings directly shape strategy"
    }}
  ]
}}"""

PROMPT_DKGE_QUERY = """You are a KNOWLEDGE GRAPH QUERY RESOLVER.

AIM / QUERY: {query}
RELEVANT GRAPH NODES:
{nodes_json}

RELEVANT EDGES:
{edges_json}

Analyse the graph to answer this query:
  1. Which entities are most relevant?
  2. What relationships explain the query context?
  3. What domain clusters emerge?
  4. What insights does the graph reveal that the raw plan does not?

Respond ONLY with valid JSON:
{{
  "top_entities": ["entity_id_1", "entity_id_2"],
  "key_relationships": ["source → relation → target"],
  "domain_clusters": {{"cluster_name": ["entity_id_1"]}},
  "graph_insights": ["insight_1", "insight_2"],
  "recommended_steps": ["step_1", "step_2"]
}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class DynamicKnowledgeGraphEngine:
    """
    Builds and queries a persistent property graph from plan entities.

    Each session extracts entities/relationships via LLM and merges them
    into a SQLite-backed graph. Cross-session queries reveal domain patterns
    invisible to per-session analysis.
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        db_path:  str = DKGE_DB_PATH,
        agent:    str = "gemini",
    ):
        self.call_fn = call_fn
        self.db_path = db_path
        self.agent   = agent
        self._init_db()

    # ── DB Setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    entity_id    TEXT PRIMARY KEY,
                    label        TEXT NOT NULL,
                    entity_type  TEXT NOT NULL,
                    description  TEXT,
                    frequency    INTEGER DEFAULT 1,
                    domains      TEXT DEFAULT '[]',
                    first_seen   TEXT,
                    last_seen    TEXT,
                    plan_contexts TEXT DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    edge_id       TEXT PRIMARY KEY,
                    source_id     TEXT NOT NULL,
                    target_id     TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight        REAL DEFAULT 0.75,
                    evidence_count INTEGER DEFAULT 1,
                    last_seen     TEXT,
                    UNIQUE(source_id, target_id, relation_type)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON kg_edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON kg_edges(target_id)")
            conn.commit()
        logger.info(f"[DKGE] Database initialised: {self.db_path}")

    # ── Extraction ────────────────────────────────────────────────────────────

    async def extract_and_merge(
        self,
        plan_text:  str,
        aim:        str,
        session_id: str,
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        Extract entities/relationships from a plan via LLM,
        then merge into the persistent graph.
        """
        prompt = PROMPT_DKGE_EXTRACTOR.format(
            aim       = aim,
            plan_text = plan_text[:2000],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"[DKGE] Extraction failed: {e}")
            return [], []

        entities      = data.get("entities", [])[:DKGE_MAX_ENTITIES_PER_PLAN]
        relationships = data.get("relationships", [])

        nodes = [
            KGNode(
                entity_id   = e["entity_id"],
                label       = e.get("label", e["entity_id"]),
                entity_type = e.get("entity_type", "concept"),
                description = e.get("description", ""),
                domains     = e.get("domains", []),
                plan_contexts= [session_id],
            )
            for e in entities if "entity_id" in e
        ]
        edges = [
            KGEdge(
                edge_id      = f"{r['source_id']}__{r['relation_type']}__{r['target_id']}",
                source_id    = r["source_id"],
                target_id    = r["target_id"],
                relation_type= r.get("relation_type", "enables"),
                weight       = float(r.get("weight", 0.75)),
            )
            for r in relationships
            if "source_id" in r and "target_id" in r
            and float(r.get("weight", 0.75)) >= DKGE_MIN_EDGE_CONFIDENCE
        ]

        self._upsert_nodes(nodes)
        self._upsert_edges(edges)
        logger.info(f"[DKGE] Merged {len(nodes)} nodes, {len(edges)} edges for {session_id}")
        return nodes, edges

    # ── DB Operations ─────────────────────────────────────────────────────────

    def _upsert_nodes(self, nodes: List[KGNode]) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for n in nodes:
                existing = conn.execute(
                    "SELECT frequency, domains, plan_contexts FROM kg_nodes WHERE entity_id=?",
                    (n.entity_id,)
                ).fetchone()
                if existing:
                    freq     = existing[0] + 1
                    domains  = list(set(json.loads(existing[1]) + n.domains))
                    contexts = (json.loads(existing[2]) + n.plan_contexts)[-20:]
                    conn.execute(
                        """UPDATE kg_nodes SET frequency=?, domains=?, plan_contexts=?, last_seen=?
                           WHERE entity_id=?""",
                        (freq, json.dumps(domains), json.dumps(contexts), now, n.entity_id)
                    )
                else:
                    conn.execute(
                        """INSERT INTO kg_nodes VALUES (?,?,?,?,?,?,?,?,?)""",
                        (n.entity_id, n.label, n.entity_type, n.description,
                         n.frequency, json.dumps(n.domains),
                         n.first_seen, n.last_seen, json.dumps(n.plan_contexts))
                    )
            conn.commit()

    def _upsert_edges(self, edges: List[KGEdge]) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for e in edges:
                existing = conn.execute(
                    """SELECT edge_id, weight, evidence_count FROM kg_edges
                       WHERE source_id=? AND target_id=? AND relation_type=?""",
                    (e.source_id, e.target_id, e.relation_type)
                ).fetchone()
                if existing:
                    n       = existing[2] + 1
                    new_wt  = (existing[1] * existing[2] + e.weight) / n  # running avg
                    conn.execute(
                        """UPDATE kg_edges SET weight=?, evidence_count=?, last_seen=?
                           WHERE edge_id=?""",
                        (new_wt, n, now, existing[0])
                    )
                else:
                    conn.execute(
                        "INSERT INTO kg_edges VALUES (?,?,?,?,?,?,?)",
                        (e.edge_id, e.source_id, e.target_id,
                         e.relation_type, e.weight, e.evidence_count, now)
                    )
            conn.commit()

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_neighbours(
        self,
        entity_id:     str,
        relation_types: Optional[List[str]] = None,
        min_weight:    float = DKGE_MIN_EDGE_CONFIDENCE,
    ) -> List[KGEdge]:
        """Return all edges incident to entity_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                rows = conn.execute(
                    f"""SELECT * FROM kg_edges
                        WHERE (source_id=? OR target_id=?) AND relation_type IN ({placeholders})
                        AND weight >= ?""",
                    [entity_id, entity_id] + relation_types + [min_weight]
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM kg_edges
                       WHERE (source_id=? OR target_id=?) AND weight >= ?""",
                    (entity_id, entity_id, min_weight)
                ).fetchall()
            return [KGEdge(**dict(r)) for r in rows]

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = DKGE_MAX_PATH_DEPTH,
    ) -> Optional[List[str]]:
        """BFS shortest path between two entities in the graph."""
        with sqlite3.connect(self.db_path) as conn:
            # Build adjacency from edges
            rows = conn.execute(
                "SELECT source_id, target_id FROM kg_edges WHERE weight >= ?",
                (DKGE_MIN_EDGE_CONFIDENCE,)
            ).fetchall()

        adj: Dict[str, Set[str]] = {}
        for src, tgt in rows:
            adj.setdefault(src, set()).add(tgt)
            adj.setdefault(tgt, set()).add(src)  # undirected traversal

        # BFS
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                return None
            for neighbour in adj.get(node, []):
                if neighbour == target_id:
                    return path + [target_id]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, path + [neighbour]))
        return None

    def get_domain_clusters(self) -> Dict[str, List[str]]:
        """Group nodes by domain tag."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT entity_id, domains FROM kg_nodes"
            ).fetchall()
        clusters: Dict[str, List[str]] = {}
        for eid, domains_json in rows:
            for domain in json.loads(domains_json):
                clusters.setdefault(domain, []).append(eid)
        return clusters

    def get_high_centrality_nodes(self, top_n: int = 10) -> List[KGNode]:
        """
        Degree centrality: nodes with most edges are most influential.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT entity_id, COUNT(*) as degree
                FROM (
                    SELECT source_id as entity_id FROM kg_edges
                    UNION ALL
                    SELECT target_id as entity_id FROM kg_edges
                )
                GROUP BY entity_id
                ORDER BY degree DESC
                LIMIT ?
            """, (top_n,)).fetchall()

            nodes = []
            for row in rows:
                n = conn.execute(
                    "SELECT * FROM kg_nodes WHERE entity_id=?", (row["entity_id"],)
                ).fetchone()
                if n:
                    nodes.append(KGNode(
                        entity_id   = n["entity_id"],
                        label       = n["label"],
                        entity_type = n["entity_type"],
                        description = n["description"],
                        frequency   = n["frequency"],
                        domains     = json.loads(n["domains"]),
                    ))
        return nodes

    async def query_graph(
        self,
        query:   str,
        aim:     str = "",
        top_n:   int = 10,
    ) -> GraphQueryResult:
        """
        LLM-assisted graph query. Retrieves relevant nodes/edges,
        then asks LLM to synthesise insights.
        """
        # Keyword match for candidate nodes
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)]
        candidate_nodes = self._keyword_search_nodes(keywords, limit=top_n * 2)

        # Get edges between candidates
        candidate_ids = {n.entity_id for n in candidate_nodes}
        edges = self._get_edges_between(candidate_ids)

        nodes_json = json.dumps([
            {"id": n.entity_id, "label": n.label, "type": n.entity_type,
             "freq": n.frequency, "domains": n.domains}
            for n in candidate_nodes[:top_n]
        ], indent=2)
        edges_json = json.dumps([
            {"src": e.source_id, "rel": e.relation_type,
             "tgt": e.target_id, "weight": round(e.weight, 2)}
            for e in edges[:20]
        ], indent=2)

        prompt = PROMPT_DKGE_QUERY.format(
            query      = query,
            nodes_json = nodes_json,
            edges_json = edges_json,
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"[DKGE] Query LLM failed: {e}")
            data = {}

        top_ids      = data.get("top_entities", [n.entity_id for n in candidate_nodes[:5]])
        top_nodes    = [n for n in candidate_nodes if n.entity_id in top_ids]
        paths        = []
        if len(top_ids) >= 2:
            p = self.find_shortest_path(top_ids[0], top_ids[-1])
            if p:
                paths.append(p)

        return GraphQueryResult(
            query           = query,
            matched_nodes   = top_nodes,
            subgraph_edges  = edges,
            shortest_paths  = paths,
            domain_clusters = data.get("domain_clusters", {}),
            relevance_scores= {n.entity_id: n.frequency / max(
                max(cn.frequency for cn in candidate_nodes), 1)
                for n in top_nodes},
        )

    def _keyword_search_nodes(self, keywords: List[str], limit: int = 20) -> List[KGNode]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = []
            seen    = set()
            for kw in keywords:
                rows = conn.execute(
                    """SELECT * FROM kg_nodes
                       WHERE label LIKE ? OR description LIKE ?
                       ORDER BY frequency DESC LIMIT ?""",
                    (f"%{kw}%", f"%{kw}%", limit)
                ).fetchall()
                for row in rows:
                    if row["entity_id"] not in seen:
                        seen.add(row["entity_id"])
                        results.append(KGNode(
                            entity_id   = row["entity_id"],
                            label       = row["label"],
                            entity_type = row["entity_type"],
                            description = row["description"],
                            frequency   = row["frequency"],
                            domains     = json.loads(row["domains"]),
                        ))
            return results[:limit]

    def _get_edges_between(self, node_ids: Set[str]) -> List[KGEdge]:
        if not node_ids:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(node_ids))
            rows = conn.execute(
                f"""SELECT * FROM kg_edges
                    WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
                    AND weight >= ?""",
                list(node_ids) + list(node_ids) + [DKGE_MIN_EDGE_CONFIDENCE]
            ).fetchall()
        return [KGEdge(**dict(r)) for r in rows]

    def get_graph_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            n_nodes = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
            n_edges = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
            top_freq= conn.execute(
                "SELECT label, frequency FROM kg_nodes ORDER BY frequency DESC LIMIT 5"
            ).fetchall()
            rel_dist= conn.execute(
                "SELECT relation_type, COUNT(*) FROM kg_edges GROUP BY relation_type"
            ).fetchall()
        return {
            "total_nodes":        n_nodes,
            "total_edges":        n_edges,
            "top_entities":       [{"label": r[0], "freq": r[1]} for r in top_freq],
            "relationship_dist":  {r[0]: r[1] for r in rel_dist},
            "domain_clusters":    {k: len(v) for k, v in self.get_domain_clusters().items()},
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
