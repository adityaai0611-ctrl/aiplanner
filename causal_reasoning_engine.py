# ═══════════════════════════════════════════════════════════════════════════════
# causal_reasoning_engine.py — Feature 6: Causal Reasoning Engine (CRE)
# Models cause-effect dependency chains before a single API call is made.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Awaitable

from system_config import (
    CRE_BOTTLENECK_FAN_THRESHOLD, CRE_CASCADE_RISK_THRESHOLD,
    CRE_INJECT_INTO_PROMPTS, CRE_MAX_MITIGATION_STEPS
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    """One plan step in the causal dependency graph."""
    node_id:                str
    step_text:              str
    prerequisites:          List[str]       = field(default_factory=list)
    blocks:                 List[str]       = field(default_factory=list)
    failure_cascade_risk:   float           = 0.0   # 0–1 probability
    bottleneck_score:       float           = 0.0   # fan_in + fan_out, normalised
    estimated_duration_hrs: float           = 4.0
    critical_path_member:   bool            = False
    mitigation_steps:       List[str]       = field(default_factory=list)
    semantic_topics:        List[str]       = field(default_factory=list)
    outputs_produced:       List[str]       = field(default_factory=list)
    inputs_consumed:        List[str]       = field(default_factory=list)

    @property
    def fan_in(self) -> int:
        return len(self.prerequisites)

    @property
    def fan_out(self) -> int:
        return len(self.blocks)

    @property
    def is_bottleneck(self) -> bool:
        return self.fan_in >= CRE_BOTTLENECK_FAN_THRESHOLD or \
               self.fan_out >= CRE_BOTTLENECK_FAN_THRESHOLD

    @property
    def is_high_cascade_risk(self) -> bool:
        return self.failure_cascade_risk >= CRE_CASCADE_RISK_THRESHOLD


@dataclass
class CascadeSimulation:
    """Result of simulating failure of one node."""
    failed_node_id:              str
    failed_step_text:            str
    cascade_map:                 Dict[str, Dict]   # node_id → impact dict
    plan_survival_probability:   float
    recommended_action:          str    # abort | reroute | continue_with_risk
    recovery_plan:               str
    total_steps_affected:        int


@dataclass
class CausalReport:
    """Full CRE analysis output."""
    dag:                     Dict[str, CausalNode]
    critical_path:           List[str]
    total_estimated_hrs:     float
    bottlenecks:             List[CausalNode]
    single_point_of_failure: Optional[str]
    high_cascade_nodes:      List[str]
    parallelisable_clusters: List[List[str]]
    causal_context_string:   str          # for prompt injection


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_CRE_DAG_BUILDER = """You are a CAUSAL DEPENDENCY ANALYST for an AI Planning System.

AIM: {aim}
PLAN STEPS:
{numbered_steps}

TASK: For each step, identify ALL causal dependencies.

A causal dependency exists when:
  • Step B CANNOT start until Step A completes
  • Step B's success PROBABILITY changes if Step A fails
  • Step A produces data/artifacts/decisions that Step B consumes

OUTPUT — respond with valid JSON only (no markdown, no explanation):
{{
  "causal_nodes": [
    {{
      "node_id": "step_1",
      "step_text": "...",
      "prerequisites": [],
      "blocks": ["step_3", "step_4"],
      "failure_cascade_risk": 0.85,
      "bottleneck_score": 72.0,
      "estimated_duration_hrs": 8.0,
      "critical_path_member": true,
      "mitigation_steps": ["Alternative A if this fails", "Alternative B"],
      "semantic_topics": ["market_research", "competitor_analysis"],
      "outputs_produced": ["market_report", "competitor_list"],
      "inputs_consumed": []
    }}
  ],
  "critical_path": ["step_1", "step_3", "step_7"],
  "total_estimated_hrs": 0.0,
  "top_3_bottlenecks": ["step_1"],
  "single_point_of_failure": "step_3"
}}"""

PROMPT_CRE_CASCADE_SIMULATOR = """You are a FAILURE CASCADE SIMULATOR.

AIM: {aim}
CAUSAL DAG (abbreviated):
{dag_summary}

FAILED STEP: {failed_node_id} — "{failed_step_text}"

Simulate downstream impact. For each potentially affected step estimate:
  1. Impact: impossible | harder | unaffected
  2. Cascade probability (0.0–1.0)
  3. Workaround path (null if none exists)

Respond with valid JSON only:
{{
  "failed_node": "{failed_node_id}",
  "cascade_map": {{
    "step_N": {{"impact": "impossible", "cascade_prob": 0.95, "workaround": null}}
  }},
  "plan_survival_probability": 0.35,
  "recommended_action": "abort",
  "recovery_plan": "..."
}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class CausalReasoningEngine:
    """
    Builds a causal DAG from plan steps before any agent call is made.

    Pipeline:
      build_causal_dag() → compute_critical_path() → identify_bottlenecks()
      → identify_parallel_clusters() → inject_causal_context()

    The context string is prepended to every agent prompt so plan
    generation is causally-aware from generation-0.
    """

    def __init__(
        self,
        call_fn: Callable[[str, str], Awaitable[str]],
        primary_agent: str = "gemini",
    ):
        """
        Args:
            call_fn: async callable(agent_name, prompt) → raw response string
            primary_agent: which agent handles DAG construction
        """
        self.call_fn       = call_fn
        self.primary_agent = primary_agent
        self.dag:          Dict[str, CausalNode] = {}
        self._report:      Optional[CausalReport] = None

    # ── DAG Construction ──────────────────────────────────────────────────────

    async def build_causal_dag(
        self,
        steps: List[str],
        aim:   str,
    ) -> Dict[str, CausalNode]:
        """
        Call LLM to identify causal dependencies between all steps.
        Returns complete CausalNode dict keyed by node_id.
        """
        numbered = "\n".join(f"  step_{i+1}: {s}" for i, s in enumerate(steps))
        prompt   = PROMPT_CRE_DAG_BUILDER.format(aim=aim, numbered_steps=numbered)

        logger.info(f"[CRE] Building causal DAG for {len(steps)} steps...")
        try:
            raw   = await self.call_fn(self.primary_agent, prompt)
            data  = self._parse_json(raw)
            nodes = {}
            for nd in data.get("causal_nodes", []):
                node = CausalNode(
                    node_id              = nd.get("node_id", f"step_{len(nodes)+1}"),
                    step_text            = nd.get("step_text", ""),
                    prerequisites        = nd.get("prerequisites", []),
                    blocks               = nd.get("blocks", []),
                    failure_cascade_risk = float(nd.get("failure_cascade_risk", 0.0)),
                    bottleneck_score     = float(nd.get("bottleneck_score", 0.0)),
                    estimated_duration_hrs = float(nd.get("estimated_duration_hrs", 4.0)),
                    critical_path_member = bool(nd.get("critical_path_member", False)),
                    mitigation_steps     = nd.get("mitigation_steps", [])[:CRE_MAX_MITIGATION_STEPS],
                    semantic_topics      = nd.get("semantic_topics", []),
                    outputs_produced     = nd.get("outputs_produced", []),
                    inputs_consumed      = nd.get("inputs_consumed", []),
                )
                nodes[node.node_id] = node

            # Fallback: if LLM returned no nodes, build a linear chain
            if not nodes:
                nodes = self._build_linear_fallback(steps)

            self.dag = nodes
            logger.info(f"[CRE] DAG built: {len(nodes)} nodes, "
                        f"{sum(len(n.blocks) for n in nodes.values())} edges")
            return nodes

        except Exception as e:
            logger.warning(f"[CRE] DAG construction failed: {e}. Using linear fallback.")
            self.dag = self._build_linear_fallback(steps)
            return self.dag

    def _build_linear_fallback(self, steps: List[str]) -> Dict[str, CausalNode]:
        """Linear sequential chain — no parallelism, each step depends on previous."""
        nodes = {}
        for i, step in enumerate(steps):
            nid = f"step_{i+1}"
            nodes[nid] = CausalNode(
                node_id      = nid,
                step_text    = step,
                prerequisites= [f"step_{i}"] if i > 0 else [],
                blocks       = [f"step_{i+2}"] if i < len(steps) - 1 else [],
                failure_cascade_risk = 0.5,
                estimated_duration_hrs = 4.0,
                critical_path_member   = True,
            )
        return nodes

    # ── Critical Path ─────────────────────────────────────────────────────────

    def compute_critical_path(
        self,
        dag: Optional[Dict[str, CausalNode]] = None,
    ) -> List[str]:
        """
        Longest dependency chain by estimated_duration_hrs.
        Uses dynamic programming on the topological order.
        Returns list of node_ids on the critical path.
        """
        dag = dag or self.dag
        if not dag:
            return []

        # Topological sort (Kahn's)
        in_degree = {nid: len(n.prerequisites) for nid, n in dag.items()}
        queue     = [nid for nid, deg in in_degree.items() if deg == 0]
        topo      = []
        while queue:
            nid = queue.pop(0)
            topo.append(nid)
            for successor in dag[nid].blocks:
                if successor in in_degree:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        # DP: earliest finish time + predecessor tracking
        earliest: Dict[str, float] = {nid: 0.0 for nid in dag}
        pred:     Dict[str, Optional[str]] = {nid: None for nid in dag}

        for nid in topo:
            node = dag[nid]
            for successor_id in node.blocks:
                if successor_id not in dag:
                    continue
                candidate = earliest[nid] + node.estimated_duration_hrs
                if candidate > earliest[successor_id]:
                    earliest[successor_id] = candidate
                    pred[successor_id]     = nid

        # Trace back from node with maximum earliest finish
        end_node = max(earliest, key=earliest.get)
        path = []
        cur  = end_node
        while cur is not None:
            path.append(cur)
            cur = pred.get(cur)
        path.reverse()

        # Mark nodes
        for nid in path:
            if nid in dag:
                dag[nid].critical_path_member = True

        logger.info(f"[CRE] Critical path: {path} "
                    f"({earliest[end_node]:.1f} hrs)")
        return path

    # ── Bottleneck Detection ──────────────────────────────────────────────────

    def identify_bottlenecks(
        self,
        dag:       Optional[Dict[str, CausalNode]] = None,
        threshold: int = CRE_BOTTLENECK_FAN_THRESHOLD,
    ) -> List[CausalNode]:
        """
        Return nodes where fan-in or fan-out exceeds threshold.
        Updates bottleneck_score for each node.
        """
        dag       = dag or self.dag
        max_fan   = max((n.fan_in + n.fan_out for n in dag.values()), default=1) or 1
        bottlenecks = []

        for node in dag.values():
            raw_score = node.fan_in + node.fan_out
            node.bottleneck_score = (raw_score / max_fan) * 100
            if raw_score >= threshold:
                bottlenecks.append(node)

        bottlenecks.sort(key=lambda n: n.bottleneck_score, reverse=True)
        logger.info(f"[CRE] Bottlenecks: {[n.node_id for n in bottlenecks]}")
        return bottlenecks

    # ── Parallel Cluster Detection ────────────────────────────────────────────

    def identify_parallel_clusters(
        self,
        dag: Optional[Dict[str, CausalNode]] = None,
    ) -> List[List[str]]:
        """
        Groups of steps that have no causal dependency between them
        and can therefore run in parallel.
        """
        dag = dag or self.dag

        # Build adjacency set (direct + transitive reachability)
        def reachable(start: str, d: Dict[str, CausalNode]) -> Set[str]:
            visited, stack = set(), [start]
            while stack:
                cur = stack.pop()
                for nxt in d.get(cur, CausalNode("", "")).blocks:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            return visited

        reach = {nid: reachable(nid, dag) for nid in dag}
        nodes = list(dag.keys())
        assigned: Set[str] = set()
        clusters: List[List[str]] = []

        for nid in nodes:
            if nid in assigned:
                continue
            cluster = [nid]
            for other in nodes:
                if other == nid or other in assigned:
                    continue
                # Independent if neither can reach the other
                if other not in reach[nid] and nid not in reach[other]:
                    # Also share the same predecessor set level
                    if dag[nid].prerequisites == dag[other].prerequisites or \
                       (not dag[nid].prerequisites and not dag[other].prerequisites):
                        cluster.append(other)
            if len(cluster) > 1:
                for c in cluster:
                    assigned.add(c)
                clusters.append(cluster)

        logger.info(f"[CRE] Parallel clusters: {clusters}")
        return clusters

    # ── Cascade Simulation ────────────────────────────────────────────────────

    async def simulate_failure_cascade(
        self,
        failed_node_id: str,
        dag:            Optional[Dict[str, CausalNode]] = None,
        aim:            str = "",
    ) -> CascadeSimulation:
        """
        LLM-powered: what happens when a specific node fails?
        Falls back to heuristic propagation if LLM call fails.
        """
        dag  = dag or self.dag
        node = dag.get(failed_node_id)
        if not node:
            return CascadeSimulation(
                failed_node_id            = failed_node_id,
                failed_step_text          = "Unknown",
                cascade_map               = {},
                plan_survival_probability = 0.0,
                recommended_action        = "abort",
                recovery_plan             = "Node not found in DAG",
                total_steps_affected      = 0,
            )

        dag_summary = json.dumps(
            {nid: {"blocks": n.blocks, "cascade_risk": n.failure_cascade_risk}
             for nid, n in dag.items()},
            indent=2
        )[:1500]  # truncate for prompt

        prompt = PROMPT_CRE_CASCADE_SIMULATOR.format(
            aim             = aim,
            dag_summary     = dag_summary,
            failed_node_id  = failed_node_id,
            failed_step_text= node.step_text[:200],
        )

        try:
            raw  = await self.call_fn(self.primary_agent, prompt)
            data = self._parse_json(raw)
            cascade_map = data.get("cascade_map", {})
            return CascadeSimulation(
                failed_node_id            = failed_node_id,
                failed_step_text          = node.step_text,
                cascade_map               = cascade_map,
                plan_survival_probability = float(data.get("plan_survival_probability", 0.5)),
                recommended_action        = data.get("recommended_action", "continue_with_risk"),
                recovery_plan             = data.get("recovery_plan", "No recovery plan generated"),
                total_steps_affected      = len(cascade_map),
            )
        except Exception as e:
            logger.warning(f"[CRE] Cascade simulation failed: {e}. Using heuristic.")
            return self._heuristic_cascade(failed_node_id, dag)

    def _heuristic_cascade(
        self, failed_id: str, dag: Dict[str, CausalNode]
    ) -> CascadeSimulation:
        """Simple reachability-based cascade without LLM."""
        affected = {}
        stack    = dag[failed_id].blocks[:]
        while stack:
            nid = stack.pop()
            if nid not in affected and nid in dag:
                prob = dag[failed_id].failure_cascade_risk * \
                       dag[nid].failure_cascade_risk
                affected[nid] = {"impact": "harder", "cascade_prob": prob, "workaround": None}
                stack.extend(dag[nid].blocks)

        survival = max(0.0, 1.0 - len(affected) / max(len(dag), 1))
        return CascadeSimulation(
            failed_node_id            = failed_id,
            failed_step_text          = dag[failed_id].step_text,
            cascade_map               = affected,
            plan_survival_probability = survival,
            recommended_action        = "reroute" if survival > 0.4 else "abort",
            recovery_plan             = "Review mitigation steps for affected nodes.",
            total_steps_affected      = len(affected),
        )

    # ── Prompt Context Generation ─────────────────────────────────────────────

    def inject_causal_context(
        self,
        dag:            Optional[Dict[str, CausalNode]] = None,
        critical_path:  Optional[List[str]] = None,
        bottlenecks:    Optional[List[CausalNode]] = None,
    ) -> str:
        """
        Build the causal context string to prepend to every agent prompt.
        Keeps it concise — focused on what agents need to generate better plans.
        """
        if not CRE_INJECT_INTO_PROMPTS:
            return ""

        dag        = dag or self.dag
        cp         = critical_path or []
        bnecks     = bottlenecks or []

        lines = [
            "╔══ CAUSAL DEPENDENCY CONTEXT (use this to improve your plan) ══╗"
        ]

        if cp:
            cp_texts = [dag[nid].step_text[:60] for nid in cp if nid in dag]
            lines.append(f"  CRITICAL PATH ({len(cp)} steps, highest failure impact):")
            for t in cp_texts[:4]:
                lines.append(f"    → {t}...")

        if bnecks:
            lines.append(f"  BOTTLENECK STEPS (high dependency — add extra detail here):")
            for n in bnecks[:3]:
                lines.append(f"    ⚡ {n.node_id}: {n.step_text[:70]}... "
                              f"[cascade risk={n.failure_cascade_risk:.0%}]")
                if n.mitigation_steps:
                    lines.append(f"       Mitigations: {'; '.join(n.mitigation_steps[:2])}")

        # High-cascade nodes
        high_risk = [n for n in dag.values() if n.is_high_cascade_risk and n not in bnecks]
        if high_risk:
            lines.append(f"  HIGH CASCADE RISK STEPS (if these fail, plan likely collapses):")
            for n in high_risk[:2]:
                lines.append(f"    ⚠ {n.node_id}: {n.step_text[:70]}...")

        lines.append("╚════════════════════════════════════════════════════════════╝")
        ctx = "\n".join(lines)

        logger.info(f"[CRE] Causal context built: {len(lines)} lines")
        return ctx

    # ── Full Analysis ─────────────────────────────────────────────────────────

    async def analyse(
        self,
        steps: List[str],
        aim:   str,
    ) -> CausalReport:
        """
        Master entry point. Runs full CRE pipeline.
        Returns CausalReport with everything needed by planner_v4.py.
        """
        dag         = await self.build_causal_dag(steps, aim)
        cp          = self.compute_critical_path(dag)
        bottlenecks = self.identify_bottlenecks(dag)
        clusters    = self.identify_parallel_clusters(dag)
        ctx         = self.inject_causal_context(dag, cp, bottlenecks)

        total_hrs    = sum(n.estimated_duration_hrs for n in dag.values())
        spof         = next(
            (n.node_id for n in dag.values()
             if n.fan_out >= CRE_BOTTLENECK_FAN_THRESHOLD and n.failure_cascade_risk > 0.8),
            None
        )
        high_cascade = [n.node_id for n in dag.values() if n.is_high_cascade_risk]

        self._report = CausalReport(
            dag                     = dag,
            critical_path           = cp,
            total_estimated_hrs     = total_hrs,
            bottlenecks             = bottlenecks,
            single_point_of_failure = spof,
            high_cascade_nodes      = high_cascade,
            parallelisable_clusters = clusters,
            causal_context_string   = ctx,
        )
        logger.info(
            f"[CRE] Analysis complete. CP={cp}, bottlenecks={len(bottlenecks)}, "
            f"SPOF={spof}, total_hrs={total_hrs:.1f}"
        )
        return self._report

    def get_report_dict(self) -> Dict:
        """Serialise CausalReport to dict for Excel/frontend output."""
        if not self._report:
            return {}
        r = self._report
        return {
            "critical_path":           r.critical_path,
            "total_estimated_hrs":     r.total_estimated_hrs,
            "bottleneck_count":        len(r.bottlenecks),
            "bottlenecks":             [{"id": n.node_id, "score": n.bottleneck_score,
                                         "cascade_risk": n.failure_cascade_risk}
                                        for n in r.bottlenecks],
            "single_point_of_failure": r.single_point_of_failure,
            "high_cascade_nodes":      r.high_cascade_nodes,
            "parallel_clusters":       r.parallelisable_clusters,
            "dag_node_count":          len(r.dag),
            "dag_edge_count":          sum(len(n.blocks) for n in r.dag.values()),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        """Extract JSON from LLM response, handling markdown fences."""
        raw = raw.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try stripping markdown
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {}
