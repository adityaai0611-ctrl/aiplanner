# ═══════════════════════════════════════════════════════════════════════════════
# plan_entropy_monitor.py — Feature 19: PEM
# Measures Shannon entropy across the plan population.
# When diversity collapses → forces re-diversification to prevent groupthink.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

PEM_ENTROPY_COLLAPSE_THRESHOLD = 1.2    # bits — below this = diversity collapse
PEM_ENTROPY_HEALTHY_MIN        = 2.0    # bits — target minimum
PEM_VOCAB_TOP_N                = 50     # top N tokens for vocab distribution
PEM_STRUCTURAL_SIM_THRESHOLD   = 0.85   # cosine sim above this = structural clone
PEM_DIVERSITY_INJECTION_TYPES  = [
    "constraint_violation",   # "solve WITHOUT technology X"
    "temporal_shift",         # "solve as if it's 1990 / 2040"
    "resource_inversion",     # "solve with 10x budget / 0.1x budget"
    "perspective_flip",       # "solve from the perspective of your customer"
    "domain_reframe",         # "solve as if this is a medical/military problem"
]


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class EntropyMeasurement:
    """Entropy metrics for the current plan population."""
    lexical_entropy:     float    # Shannon H on token distribution
    structural_entropy:  float    # diversity of step-count distributions
    semantic_entropy:    float    # variance in topic coverage
    composite_entropy:   float    # weighted average
    is_collapsed:        bool     # True if below collapse threshold
    clone_pairs:         List[Tuple[str, str]]   # (agent_a, agent_b) too similar
    unique_plans:        int
    total_plans:         int
    diversity_pct:       float    # unique/total as percentage


@dataclass
class DiversityInjection:
    """A forced diversity intervention to re-expand the solution space."""
    injection_id:     str
    injection_type:   str
    constraint:       str        # the actual constraint text injected
    target_agents:    List[str]  # which agents get this injection
    modified_prompts: Dict[str, str]   # agent → modified prompt


@dataclass
class PEMResult:
    before:           EntropyMeasurement
    after:            Optional[EntropyMeasurement]
    injections_used:  List[DiversityInjection]
    entropy_recovered:float
    intervention_was_needed: bool


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_PEM_CONSTRAINT_GENERATOR = """You are a DIVERSITY INJECTION SPECIALIST.

The current agent population has produced near-identical plans (entropy collapse detected).
You must design a constraint that FORCES structural diversity without changing the goal.

AIM: {aim}
INJECTION TYPE: {injection_type}
CLONE PAIRS DETECTED: {clone_pairs}

Design a forcing constraint of type "{injection_type}":
  constraint_violation — "Solve this WITHOUT using [dominant technology/approach]"
  temporal_shift       — "Solve this as if operating in [1985/2040/different era]"
  resource_inversion   — "Solve with [10x resources / 10% of standard budget]"
  perspective_flip     — "Solve from the perspective of [customer/regulator/competitor]"
  domain_reframe       — "Reframe this as a [medical/military/sports] problem"

The constraint must:
  1. Genuinely prevent copy-paste of the dominant plan
  2. Still be answerable for the original AIM
  3. Produce meaningfully different plan structures

Respond ONLY with valid JSON:
{{"constraint_text":"...", "rationale":"...", "expected_diversity_gain":"high|medium|low"}}"""

PROMPT_PEM_DIVERSITY_PROMPT = """You are a creative planner operating under a special constraint.

{constraint_text}

AIM: {aim}

CURRENT STEPS (which you must DIVERGE FROM):
{current_steps}

Generate a STRUCTURALLY DIFFERENT execution plan.
Your plan MUST differ from the current steps in:
  • Step ordering
  • Approach methodology
  • Resource allocation strategy
  • At least 3 completely novel steps not present above

Output only numbered steps (no JSON, no explanation):"""


# ── Engine ────────────────────────────────────────────────────────────────────

class PlanEntropyMonitor:
    """
    Continuously monitors diversity across the agent plan population.

    Mechanism:
    1. Compute lexical, structural, and semantic entropy
    2. If composite entropy < threshold → diversity collapse detected
    3. Identify clone pairs (agents producing near-identical plans)
    4. Generate forcing constraints to re-expand solution space
    5. Re-prompt affected agents with diversity-injected prompts
    6. Measure entropy recovery
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        agent:    str = "gemini",
    ):
        self.call_fn = call_fn
        self.agent   = agent

    # ── Entropy Measurement ───────────────────────────────────────────────────

    def measure_lexical_entropy(self, plans: Dict[str, str]) -> float:
        """
        Shannon entropy H = -Σ p(token) * log2(p(token))
        Measured over the top-N token vocabulary across all plans.
        High H = diverse vocabulary = diverse plans.
        Low H  = all plans use same words = groupthink.
        """
        all_tokens: List[str] = []
        for text in plans.values():
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_tokens.extend(tokens)

        if not all_tokens:
            return 0.0

        counts = Counter(all_tokens)
        top_tokens = [c for _, c in counts.most_common(PEM_VOCAB_TOP_N)]
        total = sum(top_tokens)
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in top_tokens:
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def measure_structural_entropy(self, plans: Dict[str, str]) -> float:
        """
        Entropy over step-count distribution.
        If all agents produce 10-step plans → low structural entropy.
        Diverse step counts → high entropy.
        """
        step_counts = []
        for text in plans.values():
            steps = [l for l in text.split('\n') if re.match(r'^\s*(step\s*\d+|\d+[\.\)])', l, re.IGNORECASE)]
            step_counts.append(len(steps) if steps else len(text.split('\n')))

        if not step_counts:
            return 0.0

        count_dist = Counter(step_counts)
        total = len(step_counts)
        entropy = 0.0
        for c in count_dist.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def measure_semantic_entropy(self, plans: Dict[str, str]) -> float:
        """
        Variance in unique bigram sets across plans.
        Low variance = all plans cover same concepts = groupthink.
        """
        def get_bigrams(text: str):
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(zip(tokens, tokens[1:]))

        bigram_sets = [get_bigrams(text) for text in plans.values()]
        if len(bigram_sets) < 2:
            return 2.0  # single plan = assume max diversity

        # Pairwise Jaccard distances
        distances = []
        for i in range(len(bigram_sets)):
            for j in range(i + 1, len(bigram_sets)):
                a, b = bigram_sets[i], bigram_sets[j]
                union = len(a | b)
                if union > 0:
                    jaccard_sim = len(a & b) / union
                    distances.append(1.0 - jaccard_sim)

        # Map mean Jaccard distance to entropy-like scale (0–4 bits)
        mean_dist = statistics.mean(distances) if distances else 0.5
        return round(mean_dist * 4.0, 4)

    def detect_clone_pairs(
        self,
        plans:     Dict[str, str],
        threshold: float = PEM_STRUCTURAL_SIM_THRESHOLD,
    ) -> List[Tuple[str, str]]:
        """Identify agent pairs producing near-identical plans via bigram overlap."""
        def get_bigrams(text: str):
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(zip(tokens, tokens[1:]))

        agents     = list(plans.keys())
        bigrams    = {a: get_bigrams(plans[a]) for a in agents}
        clone_pairs = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a1, a2 = agents[i], agents[j]
                b1, b2 = bigrams[a1], bigrams[a2]
                union  = len(b1 | b2)
                if union > 0:
                    sim = len(b1 & b2) / union
                    if sim >= threshold:
                        clone_pairs.append((a1, a2))

        return clone_pairs

    def measure_population_entropy(
        self,
        plans: Dict[str, str],
    ) -> EntropyMeasurement:
        """Full entropy measurement — all 3 dimensions."""
        lexical    = self.measure_lexical_entropy(plans)
        structural = self.measure_structural_entropy(plans)
        semantic   = self.measure_semantic_entropy(plans)
        clone_pairs= self.detect_clone_pairs(plans)

        # Weighted composite (semantic weighted highest — most meaningful)
        composite  = 0.30 * lexical + 0.20 * structural + 0.50 * semantic
        is_collapsed = composite < PEM_ENTROPY_COLLAPSE_THRESHOLD

        # Unique plans: not in any clone pair
        cloned_agents = set(a for pair in clone_pairs for a in pair)
        unique_count  = len(plans) - len(cloned_agents) + \
                        (len(cloned_agents) // 2 if cloned_agents else 0)

        m = EntropyMeasurement(
            lexical_entropy    = lexical,
            structural_entropy = structural,
            semantic_entropy   = semantic,
            composite_entropy  = round(composite, 4),
            is_collapsed       = is_collapsed,
            clone_pairs        = clone_pairs,
            unique_plans       = max(0, unique_count),
            total_plans        = len(plans),
            diversity_pct      = round(max(0, unique_count) / max(len(plans), 1) * 100, 1),
        )
        logger.info(
            f"[PEM] Entropy: lexical={lexical:.2f} structural={structural:.2f} "
            f"semantic={semantic:.2f} composite={composite:.2f} "
            f"collapsed={is_collapsed} clones={len(clone_pairs)}"
        )
        return m

    # ── Diversity Injection ───────────────────────────────────────────────────

    async def generate_constraint(
        self,
        injection_type: str,
        aim:            str,
        clone_pairs:    List[Tuple[str, str]],
    ) -> str:
        """LLM designs the forcing constraint."""
        clone_str = ", ".join(f"({a}↔{b})" for a, b in clone_pairs[:3])
        prompt    = PROMPT_PEM_CONSTRAINT_GENERATOR.format(
            aim            = aim,
            injection_type = injection_type,
            clone_pairs    = clone_str or "multiple agents produced identical plans",
        )
        try:
            raw  = await self.call_fn(self.agent, prompt)
            data = _parse_json(raw)
            return data.get("constraint_text", f"Solve this WITHOUT using the most common approach.")
        except Exception:
            return _fallback_constraint(injection_type, aim)

    def build_diversity_prompt(
        self,
        constraint_text: str,
        aim:             str,
        current_steps:   List[str],
    ) -> str:
        """Build the forced-diversity prompt."""
        current = "\n".join(
            f"  Step {i+1}: {s}" for i, s in enumerate(current_steps[:8])
        )
        return PROMPT_PEM_DIVERSITY_PROMPT.format(
            constraint_text = constraint_text,
            aim             = aim,
            current_steps   = current,
        )

    async def inject_diversity(
        self,
        plans:         Dict[str, str],
        aim:           str,
        measurement:   EntropyMeasurement,
        n_injections:  int = 2,
    ) -> List[DiversityInjection]:
        """
        Selects injection types, generates constraints, builds diversity prompts.
        Targets agents in clone pairs first, then generalists.
        """
        injections = []

        # Agents to re-prompt: prefer clone-pair members
        cloned_agents = list(set(a for pair in measurement.clone_pairs for a in pair))
        target_agents = cloned_agents or list(plans.keys())

        # Select injection types (cycle through available types)
        selected_types = PEM_DIVERSITY_INJECTION_TYPES[:n_injections]

        # Current majority plan (mode plan = most common structure)
        majority_steps = _extract_steps(list(plans.values())[0])

        for i, inj_type in enumerate(selected_types):
            constraint = await self.generate_constraint(
                inj_type, aim, measurement.clone_pairs
            )
            # Each injection targets a subset of agents
            batch_size  = max(1, len(target_agents) // n_injections)
            start       = i * batch_size
            batch_agents= target_agents[start:start + batch_size]

            modified_prompts = {}
            for agent in batch_agents:
                modified_prompts[agent] = self.build_diversity_prompt(
                    constraint, aim, majority_steps
                )

            injections.append(DiversityInjection(
                injection_id     = f"inj_{i+1}",
                injection_type   = inj_type,
                constraint       = constraint,
                target_agents    = batch_agents,
                modified_prompts = modified_prompts,
            ))

            logger.info(
                f"[PEM] Injection {i+1}: type={inj_type} "
                f"targets={batch_agents} constraint='{constraint[:60]}...'"
            )

        return injections

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    async def monitor_and_intervene(
        self,
        plans:        Dict[str, str],
        aim:          str,
        call_api_fn:  Optional[Callable[[str, str], Awaitable[str]]] = None,
    ) -> PEMResult:
        """
        Full PEM pipeline:
        1. Measure entropy
        2. If collapsed → inject diversity
        3. Re-call agents with diversity prompts (if call_api_fn provided)
        4. Measure entropy after intervention
        5. Merge new diverse plans back into population
        """
        before = self.measure_population_entropy(plans)
        injections_used: List[DiversityInjection] = []
        after: Optional[EntropyMeasurement] = None
        recovery = 0.0

        if not before.is_collapsed:
            logger.info(
                f"[PEM] Population entropy healthy ({before.composite_entropy:.2f} bits). "
                "No intervention needed."
            )
            return PEMResult(
                before                  = before,
                after                   = None,
                injections_used         = [],
                entropy_recovered       = 0.0,
                intervention_was_needed = False,
            )

        logger.warning(
            f"[PEM] ENTROPY COLLAPSE: {before.composite_entropy:.2f} bits < "
            f"{PEM_ENTROPY_COLLAPSE_THRESHOLD} threshold. "
            f"Clone pairs: {before.clone_pairs}. Injecting diversity..."
        )

        # Generate injections
        injections = await self.inject_diversity(plans, aim, before)
        injections_used = injections

        # Re-call agents if API function provided
        if call_api_fn:
            new_plans = dict(plans)  # start with existing
            for injection in injections:
                tasks = {
                    agent: call_api_fn(agent, prompt)
                    for agent, prompt in injection.modified_prompts.items()
                }
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                for (agent, _), result in zip(tasks.items(), results):
                    if isinstance(result, str) and result.strip():
                        new_plans[agent] = result
                        logger.info(f"[PEM] Re-generated plan for {agent}")

            # Measure entropy after intervention
            after    = self.measure_population_entropy(new_plans)
            recovery = after.composite_entropy - before.composite_entropy
            logger.info(
                f"[PEM] Post-injection entropy: {after.composite_entropy:.2f} bits "
                f"(recovery: +{recovery:.2f})"
            )
        else:
            # Just report the injections without calling agents
            logger.info("[PEM] Injections generated. No call_api_fn provided — skipping re-generation.")

        return PEMResult(
            before                  = before,
            after                   = after,
            injections_used         = injections_used,
            entropy_recovered       = recovery,
            intervention_was_needed = True,
        )

    def get_pem_report(self, result: PEMResult) -> Dict:
        def em_dict(m: EntropyMeasurement) -> Dict:
            return {
                "lexical_entropy":    m.lexical_entropy,
                "structural_entropy": m.structural_entropy,
                "semantic_entropy":   m.semantic_entropy,
                "composite_entropy":  m.composite_entropy,
                "is_collapsed":       m.is_collapsed,
                "clone_pairs":        m.clone_pairs,
                "unique_plans":       m.unique_plans,
                "total_plans":        m.total_plans,
                "diversity_pct":      m.diversity_pct,
            }
        return {
            "intervention_needed":  result.intervention_was_needed,
            "entropy_before":       em_dict(result.before),
            "entropy_after":        em_dict(result.after) if result.after else None,
            "entropy_recovered":    round(result.entropy_recovered, 4),
            "injections_count":     len(result.injections_used),
            "injections": [
                {
                    "type":       inj.injection_type,
                    "constraint": inj.constraint[:100],
                    "targets":    inj.target_agents,
                }
                for inj in result.injections_used
            ],
        }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _extract_steps(plan_text: str) -> List[str]:
    lines = plan_text.strip().split('\n')
    steps = []
    for line in lines:
        clean = re.sub(r'^\s*(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '', line, flags=re.IGNORECASE).strip()
        if clean:
            steps.append(clean)
    return steps or [plan_text]


def _fallback_constraint(injection_type: str, aim: str) -> str:
    fallbacks = {
        "constraint_violation": f"Solve '{aim[:60]}' without using the internet, cloud services, or automated tools.",
        "temporal_shift":       f"Solve '{aim[:60]}' as if operating in 1995 with no modern technology.",
        "resource_inversion":   f"Solve '{aim[:60]}' with exactly 10% of the typical budget.",
        "perspective_flip":     f"Solve '{aim[:60]}' entirely from your end customer's perspective.",
        "domain_reframe":       f"Reframe '{aim[:60]}' as if it were a public health emergency response.",
    }
    return fallbacks.get(injection_type, f"Solve '{aim[:60]}' using a completely unconventional approach.")


def _parse_json(raw: str) -> Dict:
    raw   = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
