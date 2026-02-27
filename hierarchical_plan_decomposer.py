# ═══════════════════════════════════════════════════════════════════════════════
# hierarchical_plan_decomposer.py — Feature 14: HPD
# Decomposes high-level plan steps into granular executable sub-tasks with
# depth-aware dependency tracking, then re-aggregates results upward.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

HPD_MAX_DECOMPOSITION_DEPTH = 3     # L1 → L2 → L3
HPD_MAX_SUBTASKS_PER_STEP   = 5
HPD_MIN_STEP_COMPLEXITY      = 20   # words: steps below this are atomic
HPD_COMPLEXITY_THRESHOLD     = 0.65 # fraction above which step gets decomposed


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PlanNode:
    """A node in the hierarchical plan tree."""
    node_id:       str
    level:         int            # 0=root aim, 1=step, 2=sub-task, 3=micro-task
    text:          str
    parent_id:     Optional[str]
    children:      List["PlanNode"]   = field(default_factory=list)
    dependencies:  List[str]          = field(default_factory=list)  # sibling node_ids
    complexity_score: float           = 0.0   # 0–1 (1 = needs decomposition)
    is_atomic:     bool               = False  # True = cannot be further decomposed
    estimated_hrs: float              = 1.0
    assignable_to: str                = "team"  # role or agent
    acceptance_criteria: List[str]    = field(default_factory=list)
    status:        str                = "pending"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 or self.is_atomic

    @property
    def full_path(self) -> str:
        return f"L{self.level}:{self.node_id}"


@dataclass
class DecompositionResult:
    root_aim:          str
    tree:              PlanNode              # root node
    all_nodes:         Dict[str, PlanNode]  # flat lookup
    leaf_tasks:        List[PlanNode]       # atomic, executable tasks
    total_nodes:       int
    max_depth_reached: int
    critical_path:     List[str]            # node_ids on critical path
    rollup_plan:       str                  # re-aggregated high-level plan


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_HPD_COMPLEXITY_ASSESSOR = """You are a TASK COMPLEXITY ASSESSOR.

STEP: "{step_text}"
AIM: {aim}

Assess whether this step is ATOMIC (cannot be meaningfully broken down further)
or COMPOSITE (should be decomposed into sub-tasks).

Factors that make a step COMPOSITE:
  • It contains multiple distinct phases or handoffs
  • Different teams/skills are required for different parts
  • Sub-tasks have independent parallelisation potential
  • The step spans >48 hours of work

Respond ONLY with valid JSON (no markdown):
{{"is_atomic":false,"complexity_score":0.0,"decomposition_rationale":"...","estimated_hrs":0.0,"recommended_depth":1}}"""

PROMPT_HPD_DECOMPOSER = """You are a HIERARCHICAL TASK DECOMPOSER.

PARENT AIM: {parent_aim}
PARENT STEP: "{parent_step}"
DECOMPOSITION LEVEL: {level} (1=sub-tasks, 2=micro-tasks, 3=atomic actions)
CONTEXT: {context}

Decompose this step into {max_subtasks} specific, executable sub-tasks.

Each sub-task must:
  1. Be independently assignable
  2. Have a clear completion criterion
  3. Be estimated in hours
  4. Identify dependencies on sibling sub-tasks

Respond ONLY with valid JSON (no markdown):
{{"sub_tasks": [
  {{
    "text": "...",
    "estimated_hrs": 0.0,
    "assignable_to": "role_or_team",
    "acceptance_criteria": ["criterion_1"],
    "depends_on_indices": [],
    "is_atomic": false
  }}
]}}"""

PROMPT_HPD_ROLLUP = """You are a PLAN SYNTHESIS SPECIALIST.

AIM: {aim}
DETAILED TASK TREE:
{tree_summary}

Re-synthesise a concise, HIGH-LEVEL plan (8–12 steps) that:
  1. Preserves all critical milestones from the decomposed tree
  2. Groups related leaf tasks into meaningful phases
  3. Reads as an executive summary, not a task list
  4. Each step should be a coherent phase with clear outputs

Output ONLY the plan text (numbered steps, no JSON):"""


# ── Engine ────────────────────────────────────────────────────────────────────

class HierarchicalPlanDecomposer:
    """
    Recursively decomposes high-level plan steps into executable micro-tasks,
    tracks cross-level dependencies, and re-aggregates into a polished plan.

    3-level hierarchy:
      L1: Plan steps (from original plan)
      L2: Sub-tasks (LLM decomposition of complex L1 steps)
      L3: Micro-tasks (LLM decomposition of complex L2 sub-tasks)
    """

    def __init__(
        self,
        call_fn:      Callable[[str, str], Awaitable[str]],
        agent:        str = "gemini",
        max_depth:    int = HPD_MAX_DECOMPOSITION_DEPTH,
        max_subtasks: int = HPD_MAX_SUBTASKS_PER_STEP,
    ):
        self.call_fn     = call_fn
        self.agent       = agent
        self.max_depth   = max_depth
        self.max_subtasks= max_subtasks
        self._all_nodes: Dict[str, PlanNode] = {}

    # ── Complexity Assessment ─────────────────────────────────────────────────

    async def assess_complexity(
        self,
        step_text: str,
        aim:       str,
    ) -> Tuple[bool, float, float]:
        """Returns (is_atomic, complexity_score, estimated_hrs)."""
        # Fast heuristic: short steps are always atomic
        if len(step_text.split()) < HPD_MIN_STEP_COMPLEXITY:
            return True, 0.1, 2.0

        prompt = PROMPT_HPD_COMPLEXITY_ASSESSOR.format(
            step_text = step_text[:300],
            aim       = aim[:200],
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return (
                bool(data.get("is_atomic", False)),
                float(data.get("complexity_score", 0.5)),
                float(data.get("estimated_hrs", 4.0)),
            )
        except Exception as e:
            logger.warning(f"[HPD] Complexity assessment failed: {e}")
            return False, 0.5, 4.0

    # ── Decomposition ─────────────────────────────────────────────────────────

    async def decompose_node(
        self,
        node:     PlanNode,
        aim:      str,
        context:  str = "",
    ) -> List[PlanNode]:
        """LLM decomposes one node into sub-tasks."""
        prompt = PROMPT_HPD_DECOMPOSER.format(
            parent_aim  = aim[:200],
            parent_step = node.text[:300],
            level       = node.level + 1,
            context     = context[:300],
            max_subtasks= self.max_subtasks,
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            children = []
            for i, st in enumerate(data.get("sub_tasks", [])[:self.max_subtasks]):
                child_id = f"{node.node_id}_t{i+1}"
                dep_indices = st.get("depends_on_indices", [])
                deps = [f"{node.node_id}_t{j+1}" for j in dep_indices if j < i]

                child = PlanNode(
                    node_id      = child_id,
                    level        = node.level + 1,
                    text         = st.get("text", f"Sub-task {i+1}"),
                    parent_id    = node.node_id,
                    dependencies = deps,
                    estimated_hrs= float(st.get("estimated_hrs", 2.0)),
                    assignable_to= st.get("assignable_to", "team"),
                    acceptance_criteria = st.get("acceptance_criteria", []),
                    is_atomic    = bool(st.get("is_atomic", True)),
                )
                children.append(child)
                self._all_nodes[child_id] = child

            logger.info(f"[HPD] Decomposed {node.node_id} → {len(children)} sub-tasks")
            return children
        except Exception as e:
            logger.warning(f"[HPD] Decomposition of {node.node_id} failed: {e}")
            return []

    # ── Recursive Tree Builder ────────────────────────────────────────────────

    async def _build_subtree(
        self,
        node:  PlanNode,
        aim:   str,
        depth: int,
    ) -> None:
        """Recursively decomposes a node until max_depth or atomic."""
        if depth >= self.max_depth or node.is_atomic:
            node.is_atomic = True
            return

        is_atomic, complexity, hrs = await self.assess_complexity(node.text, aim)
        node.complexity_score  = complexity
        node.estimated_hrs     = hrs

        if is_atomic or complexity < HPD_COMPLEXITY_THRESHOLD:
            node.is_atomic = True
            return

        children = await self.decompose_node(node, aim)
        node.children = children

        # Recurse on complex children
        tasks = [
            self._build_subtree(child, aim, depth + 1)
            for child in children
            if not child.is_atomic
        ]
        await asyncio.gather(*tasks)

    # ── Critical Path ─────────────────────────────────────────────────────────

    def _compute_critical_path(self, root: PlanNode) -> List[str]:
        """Longest path by estimated_hrs through the tree."""
        def dfs_max(node: PlanNode) -> Tuple[float, List[str]]:
            if not node.children:
                return node.estimated_hrs, [node.node_id]
            best_dur, best_path = 0.0, []
            for child in node.children:
                dur, path = dfs_max(child)
                total = node.estimated_hrs + dur
                if total > best_dur:
                    best_dur  = total
                    best_path = [node.node_id] + path
            return best_dur, best_path

        _, path = dfs_max(root)
        return path

    # ── Rollup ────────────────────────────────────────────────────────────────

    async def rollup_to_plan(
        self,
        tree: PlanNode,
        aim:  str,
    ) -> str:
        """Re-aggregate the detailed tree back into a readable plan."""
        def summarise_node(n: PlanNode, indent: int = 0) -> str:
            prefix = "  " * indent
            line   = f"{prefix}{'→' if indent > 0 else '•'} {n.text[:100]}"
            if n.children:
                children_text = "\n".join(
                    summarise_node(c, indent + 1) for c in n.children[:3]
                )
                return f"{line}\n{children_text}"
            return line

        tree_summary = summarise_node(tree)[:2000]
        prompt       = PROMPT_HPD_ROLLUP.format(
            aim          = aim,
            tree_summary = tree_summary,
        )
        try:
            raw = await self.call_fn(self.agent, prompt)
            return raw.strip()
        except Exception as e:
            logger.warning(f"[HPD] Rollup failed: {e}")
            # Fallback: return L1 steps
            return "\n".join(
                f"Step {i+1}: {child.text}"
                for i, child in enumerate(tree.children)
            )

    # ── Master Pipeline ───────────────────────────────────────────────────────

    async def decompose(
        self,
        plan_steps: List[str],
        aim:        str,
    ) -> DecompositionResult:
        """
        Full HPD pipeline:
        1. Create L1 nodes from plan steps
        2. Recursively decompose complex nodes
        3. Compute critical path
        4. Rollup to executive plan
        """
        self._all_nodes = {}

        # Root node
        root = PlanNode(
            node_id  = "root",
            level    = 0,
            text     = aim,
            parent_id= None,
        )
        self._all_nodes["root"] = root

        # L1 nodes from original plan steps
        l1_nodes = []
        for i, step in enumerate(plan_steps):
            nid  = f"step_{i+1}"
            node = PlanNode(
                node_id   = nid,
                level     = 1,
                text      = step,
                parent_id = "root",
            )
            l1_nodes.append(node)
            self._all_nodes[nid] = node
        root.children = l1_nodes

        # Assess and decompose L1 nodes in parallel
        decompose_tasks = [
            self._build_subtree(node, aim, depth=1)
            for node in l1_nodes
        ]
        await asyncio.gather(*decompose_tasks)

        # Leaf tasks
        leaves = [
            n for n in self._all_nodes.values()
            if n.is_leaf and n.node_id != "root"
        ]

        # Critical path
        cp = self._compute_critical_path(root)

        # Rollup
        rollup = await self.rollup_to_plan(root, aim)

        # Max depth
        max_depth = max((n.level for n in self._all_nodes.values()), default=1)

        result = DecompositionResult(
            root_aim          = aim,
            tree              = root,
            all_nodes         = self._all_nodes,
            leaf_tasks        = leaves,
            total_nodes       = len(self._all_nodes),
            max_depth_reached = max_depth,
            critical_path     = cp,
            rollup_plan       = rollup,
        )
        logger.info(
            f"[HPD] Decomposition complete. "
            f"Nodes={len(self._all_nodes)} Leaves={len(leaves)} Depth={max_depth}"
        )
        return result

    def get_task_matrix(self, result: DecompositionResult) -> List[Dict]:
        """Flat task matrix for Excel output — one row per leaf node."""
        rows = []
        for node in result.leaf_tasks:
            rows.append({
                "task_id":            node.node_id,
                "level":              node.level,
                "task_text":          node.text,
                "parent_id":          node.parent_id,
                "estimated_hrs":      node.estimated_hrs,
                "assignable_to":      node.assignable_to,
                "dependencies":       ", ".join(node.dependencies),
                "acceptance_criteria":"; ".join(node.acceptance_criteria[:2]),
                "is_on_critical_path":node.node_id in result.critical_path,
                "status":             node.status,
            })
        return rows

    def get_hpd_report(self, result: DecompositionResult) -> Dict:
        total_hrs = sum(n.estimated_hrs for n in result.leaf_tasks)
        by_level  = {}
        for n in result.all_nodes.values():
            by_level[n.level] = by_level.get(n.level, 0) + 1
        return {
            "total_nodes":        result.total_nodes,
            "leaf_tasks":         len(result.leaf_tasks),
            "max_depth":          result.max_depth_reached,
            "total_estimated_hrs":round(total_hrs, 1),
            "critical_path_nodes":len(result.critical_path),
            "nodes_by_level":     by_level,
            "rollup_plan":        result.rollup_plan[:500],
            "task_matrix_rows":   len(result.leaf_tasks),
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
