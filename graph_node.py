# ═══════════════════════════════════════════════════════════════════════════════
# graph_node.py — Feature 1: Adaptive Execution Graph (AEG)
# Replaces sequential for-loops with a dynamic async DAG
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import time
import logging
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Awaitable

from system_config import (
    AEG_CONCURRENCY_LIMIT, AEG_PRUNE_SIGMA_THRESHOLD,
    AEG_CORRELATION_THRESHOLD, AEG_MAX_RETRIES, AEG_RETRY_BACKOFF_SEC
)

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    PRUNED    = "pruned"    # skipped — score too far below mean
    FAILED    = "failed"


@dataclass
class ExecutionNode:
    """Represents a single agent × parameter-group computation unit in the DAG."""
    node_id:        str
    agent_name:     str
    group_index:    int
    parameters:     List[str]
    depends_on:     List[str]       = field(default_factory=list)
    priority:       int             = 0           # higher = schedule first
    status:         NodeStatus      = NodeStatus.PENDING
    result:         Optional[str]   = None
    score:          Optional[float] = None
    token_cost:     Optional[int]   = None
    execution_time: Optional[float] = None
    retry_count:    int             = 0
    error:          Optional[str]   = None

    @property
    def is_ready(self) -> bool:
        return self.status == NodeStatus.PENDING

    @property
    def node_key(self) -> str:
        return f"{self.agent_name}:g{self.group_index}"


class AdaptiveExecutionGraph:
    """
    Builds and executes a dependency-aware DAG of agent calls.

    Key behaviours:
    • Parallel-by-default: independent nodes run under asyncio.gather()
    • Dependency edges added when historical correlation > AEG_CORRELATION_THRESHOLD
    • Nodes pruned at runtime when score < (mean − AEG_PRUNE_SIGMA_THRESHOLD × σ)
    • Priority scheduling: high-ROI agents get semaphore slots first
    • Streaming: every completed node posts to self.results_bus (asyncio.Queue)
    """

    def __init__(self, call_fn: Callable[[str, str, List[str]], Awaitable[Tuple[str, float, int]]]):
        """
        Args:
            call_fn: async callable(agent_name, prompt, parameters) → (result_text, score, token_count)
        """
        self.call_fn     = call_fn
        self.nodes:      Dict[str, ExecutionNode] = {}
        self.adj:        Dict[str, List[str]]     = defaultdict(list)   # node_id → list of successors
        self.results_bus = asyncio.Queue()
        self._completed_scores: List[float] = []

    # ── Graph Construction ──────────────────────────────────────────────────

    def build_graph(
        self,
        agents:           List[str],
        param_groups:     List[List[str]],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        score_history:    Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, ExecutionNode]:
        """
        Construct the DAG.

        • One node per (agent, group) pair.
        • Dependency edge A→B added when Pearson r(A_scores, B_scores) > threshold,
          meaning B benefits from A's context output.
        • Priority set from agent's historical mean score (higher mean = higher prio).
        """
        self.nodes.clear()
        self.adj.clear()

        # Create nodes
        for agent in agents:
            for g_idx, params in enumerate(param_groups):
                nid = f"{agent}:g{g_idx}"
                priority = 0
                if score_history and agent in score_history:
                    hist = score_history[agent]
                    priority = int(statistics.mean(hist)) if hist else 0
                self.nodes[nid] = ExecutionNode(
                    node_id    = nid,
                    agent_name = agent,
                    group_index= g_idx,
                    parameters = params,
                    priority   = priority,
                )

        # Add dependency edges from correlation matrix
        if correlation_matrix:
            for agent_a, corr_row in correlation_matrix.items():
                for agent_b, r in corr_row.items():
                    if agent_a == agent_b:
                        continue
                    if r >= AEG_CORRELATION_THRESHOLD:
                        # For each group, A must complete before B starts
                        for g_idx in range(len(param_groups)):
                            nid_a = f"{agent_a}:g{g_idx}"
                            nid_b = f"{agent_b}:g{g_idx}"
                            if nid_a in self.nodes and nid_b in self.nodes:
                                if nid_a not in self.nodes[nid_b].depends_on:
                                    self.nodes[nid_b].depends_on.append(nid_a)
                                self.adj[nid_a].append(nid_b)

        logger.info(
            f"[AEG] Built graph: {len(self.nodes)} nodes, "
            f"{sum(len(v) for v in self.adj.values())} dependency edges"
        )
        return self.nodes

    def compute_correlation_matrix(
        self,
        historical_scores: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Pearson correlation between agent score vectors.
        Vectors are zero-padded to the same length if needed.
        """
        agents = list(historical_scores.keys())
        matrix: Dict[str, Dict[str, float]] = {a: {} for a in agents}

        max_len = max((len(v) for v in historical_scores.values()), default=0)
        if max_len < 2:
            return matrix   # not enough data

        def pearson(x: List[float], y: List[float]) -> float:
            n = len(x)
            if n < 2:
                return 0.0
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
            den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
            if den_x == 0 or den_y == 0:
                return 0.0
            return num / (den_x * den_y)

        padded = {}
        for a in agents:
            v = historical_scores[a]
            padded[a] = v + [0.0] * (max_len - len(v))

        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i == j:
                    matrix[a][b] = 1.0
                elif j > i:
                    r = pearson(padded[a], padded[b])
                    matrix[a][b] = r
                    matrix[b][a] = r

        return matrix

    def reorder_by_priority(self) -> None:
        """Sort depends_on lists so higher-priority predecessors complete first."""
        for node in self.nodes.values():
            node.depends_on.sort(
                key=lambda nid: self.nodes[nid].priority if nid in self.nodes else 0,
                reverse=True
            )

    # ── Execution ───────────────────────────────────────────────────────────

    async def _execute_node(
        self,
        node: ExecutionNode,
        semaphore: asyncio.Semaphore,
        context_results: Dict[str, str],
    ) -> ExecutionNode:
        """
        Execute one node with:
        • Semaphore-bounded concurrency
        • Context injection from dependency outputs
        • Exponential-backoff retry on failure
        • Score-based pruning check post-completion
        """
        async with semaphore:
            node.status = NodeStatus.RUNNING
            start = time.monotonic()

            # Build context from completed dependencies
            dep_context = ""
            for dep_id in node.depends_on:
                dep_node = self.nodes.get(dep_id)
                if dep_node and dep_node.result:
                    dep_context += f"\n[Context from {dep_node.agent_name}]:\n{dep_node.result[:400]}\n"

            for attempt in range(AEG_MAX_RETRIES + 1):
                try:
                    result_text, score, tokens = await self.call_fn(
                        node.agent_name,
                        dep_context,
                        node.parameters,
                    )
                    node.result         = result_text
                    node.score          = score
                    node.token_cost     = tokens
                    node.execution_time = time.monotonic() - start
                    node.status         = NodeStatus.COMPLETED

                    self._completed_scores.append(score)

                    # Dynamic pruning check
                    if self._should_prune(score):
                        node.status = NodeStatus.PRUNED
                        logger.info(f"[AEG] Pruned {node.node_id} (score={score:.1f} below threshold)")

                    await self.results_bus.put(("node_complete", node))
                    return node

                except Exception as exc:
                    node.retry_count += 1
                    node.error = str(exc)
                    if attempt < AEG_MAX_RETRIES:
                        wait = AEG_RETRY_BACKOFF_SEC * (2 ** attempt)
                        logger.warning(f"[AEG] {node.node_id} attempt {attempt+1} failed: {exc}. Retrying in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        node.status = NodeStatus.FAILED
                        logger.error(f"[AEG] {node.node_id} permanently failed after {AEG_MAX_RETRIES} retries")
                        await self.results_bus.put(("node_failed", node))

        return node

    def _should_prune(self, score: float) -> bool:
        """Prune if score is more than N standard deviations below the running mean."""
        if len(self._completed_scores) < 4:
            return False
        mean = statistics.mean(self._completed_scores)
        try:
            std = statistics.stdev(self._completed_scores)
        except statistics.StatisticsError:
            return False
        if std == 0:
            return False
        return score < (mean - AEG_PRUNE_SIGMA_THRESHOLD * std)

    async def topological_execute(
        self,
        concurrency_limit: int = AEG_CONCURRENCY_LIMIT,
    ) -> Dict[str, ExecutionNode]:
        """
        Kahn's algorithm async execution.

        Maintains a ready-queue of nodes whose all dependencies are satisfied.
        Dispatches ready nodes up to `concurrency_limit` simultaneously.
        Returns the full node map after all nodes complete/fail/prune.
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        in_degree: Dict[str, int] = {nid: len(n.depends_on) for nid, n in self.nodes.items()}
        context_results: Dict[str, str] = {}

        # Initialise queue with zero-dependency nodes, sorted by priority desc
        ready: List[str] = sorted(
            [nid for nid, deg in in_degree.items() if deg == 0],
            key=lambda nid: self.nodes[nid].priority,
            reverse=True,
        )

        active_tasks: Set[asyncio.Task] = set()

        while ready or active_tasks:
            # Launch all currently-ready nodes up to semaphore limit
            while ready:
                nid = ready.pop(0)
                node = self.nodes[nid]
                task = asyncio.create_task(
                    self._execute_node(node, semaphore, context_results),
                    name=nid,
                )
                active_tasks.add(task)

            if not active_tasks:
                break

            # Wait for the first task to finish
            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                completed_node: ExecutionNode = task.result()
                if completed_node.result:
                    context_results[completed_node.node_id] = completed_node.result

                # Reduce in-degree of successors, add newly-ready ones
                for successor_id in self.adj.get(completed_node.node_id, []):
                    in_degree[successor_id] -= 1
                    if in_degree[successor_id] == 0:
                        # Insert respecting priority
                        succ = self.nodes[successor_id]
                        idx = 0
                        for i, r in enumerate(ready):
                            if self.nodes[r].priority < succ.priority:
                                idx = i
                                break
                        else:
                            idx = len(ready)
                        ready.insert(idx, successor_id)

        completed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPLETED)
        pruned    = sum(1 for n in self.nodes.values() if n.status == NodeStatus.PRUNED)
        failed    = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)
        logger.info(f"[AEG] Execution complete: {completed} completed, {pruned} pruned, {failed} failed")

        return self.nodes

    def get_execution_summary(self) -> Dict:
        """Return a structured summary of the graph execution."""
        nodes_by_status = defaultdict(list)
        for n in self.nodes.values():
            nodes_by_status[n.status.value].append({
                "node_id":        n.node_id,
                "agent":          n.agent_name,
                "group":          n.group_index,
                "score":          n.score,
                "token_cost":     n.token_cost,
                "execution_time": n.execution_time,
                "retries":        n.retry_count,
            })

        scores = [n.score for n in self.nodes.values() if n.score is not None]
        return {
            "total_nodes":    len(self.nodes),
            "by_status":      dict(nodes_by_status),
            "score_mean":     statistics.mean(scores) if scores else 0,
            "score_max":      max(scores) if scores else 0,
            "score_min":      min(scores) if scores else 0,
            "total_tokens":   sum(n.token_cost or 0 for n in self.nodes.values()),
            "total_time_sec": sum(n.execution_time or 0 for n in self.nodes.values()),
        }
