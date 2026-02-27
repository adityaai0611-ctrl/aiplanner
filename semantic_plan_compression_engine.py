# ═══════════════════════════════════════════════════════════════════════════════
# semantic_plan_compression_engine.py — Feature 9: SPCE
# Eliminates redundant steps, merges overlapping ones, 30% compression target.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Awaitable

from system_config import (
    SPCE_REDUNDANCY_THRESHOLD, SPCE_MERGE_THRESHOLD_LO,
    SPCE_MERGE_THRESHOLD_HI, SPCE_TARGET_COMPRESSION_PCT,
    SPCE_ANTI_REGRESSION_MAX_DROP, SPCE_MIN_STEPS_AFTER_COMPRESS,
    CMB_EMBEDDING_DIMS
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class StepEmbedding:
    step_id:             str
    step_text:           str
    embedding:           List[float]         = field(default_factory=list)
    semantic_topics:     List[str]           = field(default_factory=list)
    sub_goals:           List[str]           = field(default_factory=list)
    outputs_produced:    List[str]           = field(default_factory=list)
    action_type:         str                 = "implement"
    is_redundant:        bool                = False
    merge_candidate_ids: List[str]           = field(default_factory=list)
    coverage_score:      float               = 0.0   # fraction of plan coverage

    @property
    def all_coverage(self) -> Set[str]:
        return set(self.semantic_topics + self.sub_goals + self.outputs_produced)


@dataclass
class MergeResult:
    merged_step:       str
    step_a_id:         str
    step_b_id:         str
    coverage_retained: List[str]
    coverage_lost:     List[str]
    compression_ratio: float
    merge_confidence:  str
    merge_strategy:    str


@dataclass
class CompressionResult:
    original_steps:         List[str]
    compressed_steps:       List[str]
    original_count:         int
    compressed_count:       int
    compression_ratio:      float
    coverage_retained_pct:  float
    score_before:           float
    score_after:            float
    score_delta:            float
    accepted:               bool
    removed_step_ids:       List[str]
    merged_pairs:           List[Tuple[str, str]]
    rejection_reason:       Optional[str]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_SPCE_COVERAGE = """You are a SEMANTIC COVERAGE ANALYST.

AIM: {aim}
STEP: "{step_text}"

Extract the semantic topics and sub-goals this step covers.
Be specific — avoid vague categories like "planning" or "implementation".

Respond ONLY with valid JSON (no markdown):
{{"step_id":"{step_id}","semantic_topics":["topic_A","topic_B"],"sub_goals_covered":["sub_goal_1"],"action_type":"research|design|implement|test|deploy|review|communicate","outputs_produced":["artifact_A"],"inputs_consumed":["artifact_C"],"is_purely_administrative":false,"estimated_uniqueness":0.0}}"""

PROMPT_SPCE_MERGER = """You are a PLAN COMPRESSION SPECIALIST.
Merge two semantically-overlapping steps into ONE superior step.

AIM: {aim}

STEP A (id={step_a_id}): "{step_a}"
  Topics: {coverage_a}

STEP B (id={step_b_id}): "{step_b}"
  Topics: {coverage_b}

MERGE RULES:
  1. The merged step MUST cover ALL topics from both steps
  2. It must be ONE actionable sentence (no "and then" compound sentences)
  3. Start with an action verb
  4. The merged step must be SHORTER than A + B combined
  5. If B is a sub-action of A, absorb B into A

Respond ONLY with valid JSON (no markdown):
{{"merged_step":"...","coverage_retained":["topic_1","topic_2"],"coverage_lost":[],"compression_ratio":0.0,"merge_confidence":"high|medium|low","merge_strategy":"absorb_b_into_a|blend_equal|extract_essence"}}"""


# ── Engine ────────────────────────────────────────────────────────────────────

class SemanticPlanCompressionEngine:
    """
    Pipeline:
      embed_all_steps() → build_similarity_matrix() → detect_redundant()
      → detect_merge_candidates() → merge() → remove() → anti_regression_check()
    """

    def __init__(
        self,
        call_fn:    Callable[[str, str], Awaitable[str]],
        score_fn:   Callable[[str, str], Awaitable[float]],
        embed_fn:   Optional[Callable[[str], Awaitable[List[float]]]] = None,
        agent:      str = "gemini",
    ):
        """
        Args:
            call_fn:  async callable(agent, prompt) → str
            score_fn: async callable(plan_text, aim) → float
            embed_fn: async callable(text) → List[float] — if None, uses TF-IDF hash
            agent:    which agent runs coverage extraction and merging
        """
        self.call_fn  = call_fn
        self.score_fn = score_fn
        self.embed_fn = embed_fn
        self.agent    = agent
        self._embeddings: List[StepEmbedding] = []

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def embed_all_steps(
        self,
        steps: List[str],
        aim:   str,
    ) -> List[StepEmbedding]:
        """
        Embed all steps in parallel.
        Also calls LLM to extract semantic coverage per step.
        """
        embed_tasks   = [self._embed_step(f"step_{i+1}", s) for i, s in enumerate(steps)]
        coverage_tasks= [self._extract_coverage(f"step_{i+1}", s, aim) for i, s in enumerate(steps)]

        embeddings_raw, coverages = await asyncio.gather(
            asyncio.gather(*embed_tasks),
            asyncio.gather(*coverage_tasks),
        )

        result = []
        for i, (emb, cov) in enumerate(zip(embeddings_raw, coverages)):
            se = StepEmbedding(
                step_id          = f"step_{i+1}",
                step_text        = steps[i],
                embedding        = emb,
                semantic_topics  = cov.get("semantic_topics", []),
                sub_goals        = cov.get("sub_goals_covered", []),
                outputs_produced = cov.get("outputs_produced", []),
                action_type      = cov.get("action_type", "implement"),
            )
            result.append(se)

        # Compute coverage scores
        all_topics: Set[str] = set()
        for se in result:
            all_topics.update(se.all_coverage)

        for se in result:
            se.coverage_score = (
                len(se.all_coverage & all_topics) / max(len(all_topics), 1)
            )

        self._embeddings = result
        logger.info(f"[SPCE] Embedded {len(result)} steps, {len(all_topics)} total topics")
        return result

    async def _embed_step(self, step_id: str, text: str) -> List[float]:
        if self.embed_fn:
            try:
                return await self.embed_fn(text)
            except Exception:
                pass
        return _deterministic_embedding(text, CMB_EMBEDDING_DIMS)

    async def _extract_coverage(
        self, step_id: str, step_text: str, aim: str
    ) -> Dict:
        prompt = PROMPT_SPCE_COVERAGE.format(
            aim=aim, step_text=step_text[:300], step_id=step_id
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            return _parse_json(raw)
        except Exception:
            return {"semantic_topics": step_text.lower().split()[:5]}

    # ── Similarity Matrix ─────────────────────────────────────────────────────

    def build_similarity_matrix(
        self,
        embeddings: Optional[List[StepEmbedding]] = None,
    ) -> List[List[float]]:
        """N×N cosine similarity matrix."""
        embs = embeddings or self._embeddings
        n    = len(embs)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif j > i:
                    sim = _cosine_similarity(embs[i].embedding, embs[j].embedding)
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        return matrix

    # ── Redundancy Detection ──────────────────────────────────────────────────

    def detect_redundant_steps(
        self,
        embeddings: Optional[List[StepEmbedding]] = None,
        matrix:     Optional[List[List[float]]] = None,
        threshold:  float = SPCE_REDUNDANCY_THRESHOLD,
    ) -> List[str]:
        """
        A step is redundant if its semantic coverage is a subset of
        the union of coverage from other steps AND cosine sim > threshold.
        """
        embs   = embeddings or self._embeddings
        matrix = matrix or self.build_similarity_matrix(embs)
        n      = len(embs)
        redundant_ids = []

        for i, se in enumerate(embs):
            # Find highly similar steps (excluding self)
            similar_indices = [
                j for j in range(n)
                if j != i and matrix[i][j] >= threshold
            ]
            if not similar_indices:
                continue

            # Check if coverage of se is subsumed by union of similar steps
            union_coverage: Set[str] = set()
            for j in similar_indices:
                union_coverage.update(embs[j].all_coverage)

            my_coverage = se.all_coverage
            if my_coverage and my_coverage.issubset(union_coverage):
                se.is_redundant = True
                redundant_ids.append(se.step_id)
                logger.info(f"[SPCE] {se.step_id} marked redundant (sim>={threshold:.2f})")

        return redundant_ids

    # ── Merge Candidate Detection ─────────────────────────────────────────────

    def detect_merge_candidates(
        self,
        embeddings: Optional[List[StepEmbedding]] = None,
        matrix:     Optional[List[List[float]]] = None,
        lo:         float = SPCE_MERGE_THRESHOLD_LO,
        hi:         float = SPCE_MERGE_THRESHOLD_HI,
    ) -> List[Tuple[str, str]]:
        """
        Pairs where lo ≤ cosine_sim < hi: similar enough to merge,
        but not identical (redundant).
        """
        embs   = embeddings or self._embeddings
        matrix = matrix or self.build_similarity_matrix(embs)
        n      = len(embs)
        pairs: List[Tuple[str, str]] = []
        seen:  Set[Tuple[str, str]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                sim = matrix[i][j]
                if lo <= sim < hi:
                    key = (embs[i].step_id, embs[j].step_id)
                    if key not in seen:
                        pairs.append(key)
                        seen.add(key)
                        embs[i].merge_candidate_ids.append(embs[j].step_id)
                        embs[j].merge_candidate_ids.append(embs[i].step_id)

        logger.info(f"[SPCE] Merge candidates: {pairs}")
        return pairs

    # ── Merging ───────────────────────────────────────────────────────────────

    async def merge_step_pair(
        self,
        step_a: StepEmbedding,
        step_b: StepEmbedding,
        aim:    str,
    ) -> MergeResult:
        """LLM merges two semantically-overlapping steps into one."""
        prompt = PROMPT_SPCE_MERGER.format(
            aim        = aim,
            step_a_id  = step_a.step_id,
            step_a     = step_a.step_text[:250],
            coverage_a = ", ".join(list(step_a.all_coverage)[:8]),
            step_b_id  = step_b.step_id,
            step_b     = step_b.step_text[:250],
            coverage_b = ", ".join(list(step_b.all_coverage)[:8]),
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return MergeResult(
                merged_step       = data.get("merged_step", f"{step_a.step_text} and {step_b.step_text}"),
                step_a_id         = step_a.step_id,
                step_b_id         = step_b.step_id,
                coverage_retained = data.get("coverage_retained", []),
                coverage_lost     = data.get("coverage_lost", []),
                compression_ratio = float(data.get("compression_ratio", 0.5)),
                merge_confidence  = data.get("merge_confidence", "medium"),
                merge_strategy    = data.get("merge_strategy", "blend_equal"),
            )
        except Exception as e:
            logger.warning(f"[SPCE] Merge failed for {step_a.step_id}+{step_b.step_id}: {e}")
            # Heuristic fallback: take the longer (more detailed) step
            fallback = step_a.step_text if len(step_a.step_text) >= len(step_b.step_text) \
                       else step_b.step_text
            return MergeResult(
                merged_step       = fallback,
                step_a_id         = step_a.step_id,
                step_b_id         = step_b.step_id,
                coverage_retained = list(step_a.all_coverage | step_b.all_coverage),
                coverage_lost     = [],
                compression_ratio = 0.5,
                merge_confidence  = "low",
                merge_strategy    = "fallback_keep_longer",
            )

    # ── Master Pipeline ───────────────────────────────────────────────────────

    async def compress(
        self,
        plan_text:    str,
        aim:          str,
        score_before: Optional[float] = None,
    ) -> CompressionResult:
        """
        Full compression pipeline:
        1. Parse steps from plan_text
        2. Embed + extract coverage
        3. Build similarity matrix
        4. Remove redundant steps
        5. Merge candidates
        6. Anti-regression check
        7. Return CompressionResult
        """
        original_steps = _parse_steps(plan_text)
        if len(original_steps) <= SPCE_MIN_STEPS_AFTER_COMPRESS:
            logger.info(f"[SPCE] Plan too short ({len(original_steps)} steps). Skipping.")
            return _no_compression_result(original_steps, score_before or 0.0)

        # Phase 1: Embed
        embeddings = await self.embed_all_steps(original_steps, aim)
        matrix     = self.build_similarity_matrix(embeddings)

        # Phase 2: Detect redundancies
        redundant_ids = self.detect_redundant_steps(embeddings, matrix)

        # Phase 3: Detect merge candidates (among non-redundant steps)
        non_redundant = [se for se in embeddings if not se.is_redundant]
        sub_matrix    = self.build_similarity_matrix(non_redundant)
        merge_pairs   = self.detect_merge_candidates(non_redundant, sub_matrix)

        # Phase 4: Perform merges
        id_to_se    = {se.step_id: se for se in embeddings}
        removed_ids = set(redundant_ids)
        merged_pairs_done: List[Tuple[str, str]] = []
        new_steps   = list(original_steps)  # mutable copy indexed by original position

        # Build a mapping: step_id → index in original_steps
        idx_map = {f"step_{i+1}": i for i in range(len(original_steps))}

        # Merge (greedy — process pairs in order, skip if already consumed)
        merged_into: Dict[str, str] = {}   # step_id → merged text replacing it

        for (aid, bid) in merge_pairs[:5]:  # cap at 5 merges per session
            if aid in removed_ids or bid in removed_ids:
                continue
            if aid not in idx_map or bid not in idx_map:
                continue
            result = await self.merge_step_pair(id_to_se[aid], id_to_se[bid], aim)
            if result.merge_confidence == "low":
                continue
            # Replace step_a with merged, mark step_b removed
            ai = idx_map[aid]
            new_steps[ai] = result.merged_step
            removed_ids.add(bid)
            merged_into[aid] = result.merged_step
            merged_pairs_done.append((aid, bid))

        # Phase 5: Build compressed list
        compressed = []
        for i, step in enumerate(new_steps):
            sid = f"step_{i+1}"
            if sid not in removed_ids:
                compressed.append(step)

        # Enforce minimum
        if len(compressed) < SPCE_MIN_STEPS_AFTER_COMPRESS:
            compressed = original_steps  # revert

        # Phase 6: Compute coverage retained
        all_orig_topics: Set[str] = set()
        all_comp_topics: Set[str] = set()
        for se in embeddings:
            all_orig_topics.update(se.all_coverage)
            if se.step_id not in removed_ids:
                all_comp_topics.update(se.all_coverage)

        coverage_pct = (
            len(all_comp_topics & all_orig_topics) / max(len(all_orig_topics), 1) * 100
        )

        # Phase 7: Score after compression
        compressed_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(compressed))
        try:
            score_after = await self.score_fn(compressed_text, aim)
        except Exception:
            score_after = score_before or 0.0

        score_before = score_before or 0.0
        score_delta  = score_after - score_before

        # Anti-regression guard
        accepted      = True
        reject_reason = None
        if score_delta < -SPCE_ANTI_REGRESSION_MAX_DROP:
            accepted      = False
            reject_reason = (
                f"Score dropped {score_delta:.1f} pts (max allowed: "
                f"-{SPCE_ANTI_REGRESSION_MAX_DROP}). Reverting."
            )
            compressed = original_steps
            logger.warning(f"[SPCE] Anti-regression triggered: {reject_reason}")

        compression_ratio = 1.0 - len(compressed) / max(len(original_steps), 1)

        logger.info(
            f"[SPCE] Compressed {len(original_steps)} → {len(compressed)} steps "
            f"({compression_ratio:.0%} reduction), coverage={coverage_pct:.1f}%, "
            f"score_delta={score_delta:.1f}, accepted={accepted}"
        )

        return CompressionResult(
            original_steps        = original_steps,
            compressed_steps      = compressed,
            original_count        = len(original_steps),
            compressed_count      = len(compressed),
            compression_ratio     = compression_ratio,
            coverage_retained_pct = coverage_pct,
            score_before          = score_before,
            score_after           = score_after,
            score_delta           = score_delta,
            accepted              = accepted,
            removed_step_ids      = list(removed_ids),
            merged_pairs          = merged_pairs_done,
            rejection_reason      = reject_reason,
        )

    def get_compression_report(self, result: CompressionResult) -> Dict:
        return {
            "original_step_count":    result.original_count,
            "compressed_step_count":  result.compressed_count,
            "compression_ratio_pct":  round(result.compression_ratio * 100, 1),
            "coverage_retained_pct":  round(result.coverage_retained_pct, 1),
            "score_before":           result.score_before,
            "score_after":            result.score_after,
            "score_delta":            round(result.score_delta, 2),
            "accepted":               result.accepted,
            "rejection_reason":       result.rejection_reason,
            "steps_removed":          len(result.removed_step_ids),
            "steps_merged":           len(result.merged_pairs),
            "removed_ids":            result.removed_step_ids,
            "merged_pairs":           result.merged_pairs,
        }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    mag_a  = math.sqrt(sum(x * x for x in a))
    mag_b  = math.sqrt(sum(y * y for y in b))
    denom  = mag_a * mag_b
    return dot / denom if denom > 0 else 0.0


def _deterministic_embedding(text: str, dims: int = 768) -> List[float]:
    """Reproducible pseudo-embedding using character trigram hashing."""
    vec = [0.0] * dims
    tokens = text.lower().split()
    for tok in tokens:
        h = hash(tok) % dims
        vec[h] += 1.0
        if len(tok) > 2:
            h2 = hash(tok[:3]) % dims
            vec[h2] += 0.5
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / mag for x in vec]


def _parse_steps(plan_text: str) -> List[str]:
    """Extract numbered steps from plan text."""
    lines = plan_text.strip().split('\n')
    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match "Step N:" or "N." or "N)"
        clean = re.sub(r'^(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '', line, flags=re.IGNORECASE)
        if clean:
            steps.append(clean)
    return steps if steps else [plan_text]


def _no_compression_result(steps: List[str], score: float) -> CompressionResult:
    return CompressionResult(
        original_steps        = steps,
        compressed_steps      = steps,
        original_count        = len(steps),
        compressed_count      = len(steps),
        compression_ratio     = 0.0,
        coverage_retained_pct = 100.0,
        score_before          = score,
        score_after           = score,
        score_delta           = 0.0,
        accepted              = True,
        removed_step_ids      = [],
        merged_pairs          = [],
        rejection_reason      = None,
    )


def _parse_json(raw: str) -> Dict:
    raw   = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
