# ═══════════════════════════════════════════════════════════════════════════════
# dynamic_agent_specialization_router.py — Feature 18: DASR
# Domain-classifies the AIM, then routes each agent a specialized sub-prompt
# tuned to that agent's historical domain strength.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

DASR_DOMAINS = [
    "technology", "business", "finance", "legal", "medical",
    "engineering", "creative", "marketing", "operations", "research"
]

# Default agent→domain strength mapping (overridden by FLMS global model if available)
DASR_DEFAULT_AGENT_STRENGTHS: Dict[str, Dict[str, float]] = {
    "gemini":     {"technology": 90, "research": 85, "engineering": 80, "business": 75},
    "openai":     {"business": 90, "marketing": 85, "creative": 80, "legal": 75},
    "together":   {"technology": 80, "engineering": 75, "research": 80, "operations": 75},
    "groq":       {"technology": 85, "engineering": 85, "operations": 80, "research": 75},
    "replicate":  {"creative": 85, "technology": 75, "research": 80, "engineering": 70},
    "fireworks":  {"technology": 80, "engineering": 80, "operations": 75, "research": 70},
    "cohere":     {"business": 85, "marketing": 90, "creative": 85, "legal": 80},
    "deepseek":   {"technology": 90, "engineering": 90, "research": 85, "finance": 80},
    "openrouter": {"business": 80, "operations": 75, "finance": 80, "legal": 75},
    "writer":     {"creative": 90, "marketing": 85, "business": 75, "legal": 70},
    "huggingface":{"research": 85, "technology": 80, "engineering": 75, "medical": 75},
    "pawn":       {"operations": 80, "finance": 75, "business": 75, "marketing": 70},
}

# Specialized prompt suffixes per domain
DASR_DOMAIN_PROMPT_SUFFIXES: Dict[str, str] = {
    "technology": """
Focus heavily on:
  • Technical architecture and system design
  • Technology stack selection and rationale
  • Scalability, security, and performance considerations
  • API integrations, data pipelines, and infrastructure
  • Development methodology (agile/sprints/CI-CD)""",

    "business": """
Focus heavily on:
  • Market positioning and competitive differentiation
  • Revenue model and unit economics
  • Go-to-market strategy and channel selection
  • Stakeholder alignment and organisational change
  • KPIs, OKRs, and success metrics""",

    "finance": """
Focus heavily on:
  • Capital allocation and budget phasing
  • Cash flow modelling and burn rate
  • Financial risk management and hedging
  • Regulatory compliance (GAAP/IFRS/SEC)
  • ROI projections with sensitivity analysis""",

    "legal": """
Focus heavily on:
  • Regulatory compliance and jurisdiction analysis
  • Contract structures and IP protection
  • Liability management and risk mitigation
  • Data privacy (GDPR/CCPA) and consent frameworks
  • Dispute resolution and enforcement mechanisms""",

    "medical": """
Focus heavily on:
  • Clinical validation and evidence-based approaches
  • Regulatory pathway (FDA/CE/TGA approval)
  • Patient safety protocols and adverse event monitoring
  • Healthcare data security (HIPAA/HL7/FHIR)
  • Ethical review board requirements""",

    "engineering": """
Focus heavily on:
  • Technical specifications and tolerances
  • Manufacturing processes and quality control
  • Safety standards and certification requirements
  • Supply chain and component sourcing
  • Testing protocols and failure mode analysis""",

    "creative": """
Focus heavily on:
  • Creative brief and brand voice consistency
  • Content production workflows and asset management
  • Audience segmentation and personalisation
  • Multi-channel distribution strategy
  • Creative performance measurement and iteration""",

    "marketing": """
Focus heavily on:
  • Customer acquisition cost and lifetime value
  • Funnel optimisation and conversion strategy
  • Brand positioning and messaging hierarchy
  • Digital/paid/organic channel mix
  • Campaign measurement and attribution""",

    "operations": """
Focus heavily on:
  • Process design and workflow optimisation
  • Resource allocation and capacity planning
  • Vendor management and SLA governance
  • Operational risk and business continuity
  • Performance dashboards and escalation protocols""",

    "research": """
Focus heavily on:
  • Hypothesis formation and experimental design
  • Data collection methodology and sample validity
  • Statistical analysis and confidence intervals
  • Peer review and publication strategy
  • Ethical research protocols and IRB requirements""",
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class DomainClassification:
    primary_domain:   str
    secondary_domain: Optional[str]
    confidence:       float
    domain_scores:    Dict[str, float]   # all domain probabilities
    rationale:        str


@dataclass
class AgentAssignment:
    agent:             str
    role:              str    # specialist | generalist | verifier
    domain_strength:   float
    specialized_prompt:str
    reasoning:         str


@dataclass
class RouterResult:
    classification:    DomainClassification
    assignments:       Dict[str, AgentAssignment]   # agent → assignment
    prompt_variants:   Dict[str, str]               # agent → full specialized prompt
    specialist_agents: List[str]                    # top agents for this domain
    generalist_agents: List[str]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_DASR_CLASSIFIER = """You are a DOMAIN CLASSIFIER for an AI planning system.

AIM: {aim}

Classify this aim across domains. Sum of all scores must equal 1.0.

Domains: technology, business, finance, legal, medical, engineering,
         creative, marketing, operations, research

Respond ONLY with valid JSON:
{{"primary_domain":"...","secondary_domain":"...","confidence":0.0,"domain_scores":{{"technology":0.0,"business":0.0,"finance":0.0,"legal":0.0,"medical":0.0,"engineering":0.0,"creative":0.0,"marketing":0.0,"operations":0.0,"research":0.0}},"rationale":"..."}}"""

PROMPT_DASR_BASE = """You are {agent_persona}.

{domain_context}

AIM: {aim}

CURRENT PLAN STEPS:
{numbered_steps}

{memory_context}

{causal_context}

Generate an IMPROVED, DETAILED execution plan for this aim.
Apply your specialist expertise heavily — generic advice is not enough.
Every step must be specific, actionable, and domain-expert-level.

Output only numbered steps (Step 1: ... Step 2: ... etc):"""


# ── Agent Personas ────────────────────────────────────────────────────────────

AGENT_PERSONAS: Dict[str, str] = {
    "gemini":     "a Senior Systems Architect specialising in large-scale distributed systems and AI infrastructure",
    "openai":     "a Chief Strategy Officer with expertise in business model design and market expansion",
    "together":   "a Technical Lead with deep expertise in open-source AI systems and API architecture",
    "groq":       "a Performance Engineering Expert specialising in low-latency, high-throughput systems",
    "replicate":  "a Machine Learning Research Engineer specialising in model fine-tuning and deployment",
    "fireworks":  "a Cloud Infrastructure Architect specialising in scalable serverless architectures",
    "cohere":     "a Content Strategy Director with expertise in NLP-powered enterprise communication",
    "deepseek":   "a Principal Research Scientist with expertise in mathematical modelling and algorithms",
    "openrouter": "a Business Operations Analyst with expertise in process optimisation and vendor management",
    "writer":     "a Creative Director specialising in brand narrative, content strategy, and audience engagement",
    "huggingface":"a Data Science Lead specialising in ML pipelines, experiment design, and model evaluation",
    "pawn":       "a Project Management Expert with expertise in resource planning and stakeholder coordination",
}


# ── Engine ────────────────────────────────────────────────────────────────────

class DynamicAgentSpecializationRouter:
    """
    Instead of sending all agents the same generic prompt:
    1. Classify AIM domain (primary + secondary)
    2. Rank agents by historical domain strength
    3. Build agent-specific specialized prompts
    4. Route each agent their optimal prompt variant
    """

    def __init__(
        self,
        call_fn:         Callable[[str, str], Awaitable[str]],
        classifier_agent: str = "gemini",
        agent_strengths: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.call_fn          = call_fn
        self.classifier_agent = classifier_agent
        self.agent_strengths  = agent_strengths or DASR_DEFAULT_AGENT_STRENGTHS

    # ── Domain Classification ─────────────────────────────────────────────────

    async def classify_domain(self, aim: str) -> DomainClassification:
        """LLM classifies the AIM into domain probabilities."""
        prompt = PROMPT_DASR_CLASSIFIER.format(aim=aim[:400])
        try:
            raw  = await self.call_fn(self.classifier_agent, prompt)
            data = _parse_json(raw)
            return DomainClassification(
                primary_domain   = data.get("primary_domain", "business"),
                secondary_domain = data.get("secondary_domain"),
                confidence       = float(data.get("confidence", 0.7)),
                domain_scores    = {
                    d: float(data.get("domain_scores", {}).get(d, 0.0))
                    for d in DASR_DOMAINS
                },
                rationale        = data.get("rationale", ""),
            )
        except Exception as e:
            logger.warning(f"[DASR] Classification failed: {e}. Using heuristic.")
            return self._heuristic_classify(aim)

    def _heuristic_classify(self, aim: str) -> DomainClassification:
        """Keyword-based fallback classifier."""
        text   = aim.lower()
        scores = {d: 0.0 for d in DASR_DOMAINS}

        keyword_map = {
            "technology": ["software", "app", "api", "code", "system", "platform", "tech"],
            "business":   ["company", "startup", "business", "market", "revenue", "sales"],
            "finance":    ["investment", "funding", "budget", "cost", "profit", "financial"],
            "legal":      ["compliance", "legal", "regulation", "contract", "law", "ip"],
            "medical":    ["health", "medical", "patient", "clinical", "drug", "therapy"],
            "engineering":["build", "design", "manufacture", "hardware", "infrastructure"],
            "creative":   ["content", "design", "brand", "creative", "media", "art"],
            "marketing":  ["marketing", "campaign", "audience", "advertising", "growth"],
            "operations": ["process", "operations", "workflow", "logistics", "supply"],
            "research":   ["research", "study", "analysis", "data", "experiment"],
        }
        for domain, keywords in keyword_map.items():
            scores[domain] = sum(1.0 for kw in keywords if kw in text)

        total = sum(scores.values()) or 1.0
        scores = {d: v / total for d, v in scores.items()}
        primary = max(scores, key=scores.get)
        sorted_domains = sorted(scores, key=scores.get, reverse=True)
        secondary = sorted_domains[1] if len(sorted_domains) > 1 else None

        return DomainClassification(
            primary_domain   = primary,
            secondary_domain = secondary,
            confidence       = scores[primary],
            domain_scores    = scores,
            rationale        = "Heuristic keyword classification",
        )

    # ── Agent Ranking ─────────────────────────────────────────────────────────

    def rank_agents_for_domain(
        self,
        domain:  str,
        agents:  List[str],
    ) -> List[Tuple[str, float]]:
        """Return agents sorted by strength in given domain."""
        scored = []
        for agent in agents:
            strengths = self.agent_strengths.get(agent, {})
            score     = strengths.get(domain, 50.0)  # default 50 if unknown
            scored.append((agent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Prompt Building ───────────────────────────────────────────────────────

    def build_specialized_prompt(
        self,
        agent:          str,
        domain:         str,
        aim:            str,
        steps:          List[str],
        memory_context: str = "",
        causal_context: str = "",
        domain_strength:float = 75.0,
    ) -> str:
        """Build a domain-specialized prompt for a specific agent."""
        persona        = AGENT_PERSONAS.get(agent, f"an AI planning specialist")
        domain_suffix  = DASR_DOMAIN_PROMPT_SUFFIXES.get(domain, "")
        numbered_steps = "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(steps))

        # Adjust depth instruction based on domain strength
        if domain_strength >= 80:
            depth_instruction = "You are a DOMAIN EXPERT here. Go deep — technical specifics expected."
        elif domain_strength >= 65:
            depth_instruction = "Apply your domain knowledge. Be specific and practical."
        else:
            depth_instruction = "Leverage your general expertise. Prioritise logical structure."

        domain_context = f"""DOMAIN: {domain.upper()}
{depth_instruction}
{domain_suffix}"""

        return PROMPT_DASR_BASE.format(
            agent_persona  = persona,
            domain_context = domain_context,
            aim            = aim,
            numbered_steps = numbered_steps,
            memory_context = f"MEMORY CONTEXT:\n{memory_context}" if memory_context else "",
            causal_context = f"CAUSAL CONTEXT:\n{causal_context}" if causal_context else "",
        )

    # ── Master Router ─────────────────────────────────────────────────────────

    async def route(
        self,
        agents:         List[str],
        aim:            str,
        steps:          List[str],
        memory_context: str = "",
        causal_context: str = "",
    ) -> RouterResult:
        """
        Full routing pipeline:
        1. Classify domain
        2. Rank agents
        3. Build specialized prompts for each agent
        4. Assign roles (specialist/generalist/verifier)
        """
        classification = await self.classify_domain(aim)
        primary        = classification.primary_domain
        secondary      = classification.secondary_domain

        ranked = self.rank_agents_for_domain(primary, agents)

        assignments: Dict[str, AgentAssignment] = {}
        prompt_variants: Dict[str, str] = {}
        specialist_agents = []
        generalist_agents = []

        for i, (agent, strength) in enumerate(ranked):
            # Assign role
            if i < 3 and strength >= 75:
                role = "specialist"
                specialist_agents.append(agent)
                # Specialist gets primary domain prompt
                domain = primary
            elif secondary and strength >= 65:
                role   = "specialist"
                domain = secondary
                specialist_agents.append(agent)
            else:
                role   = "generalist"
                domain = primary
                generalist_agents.append(agent)

            specialized_prompt = self.build_specialized_prompt(
                agent          = agent,
                domain         = domain,
                aim            = aim,
                steps          = steps,
                memory_context = memory_context,
                causal_context = causal_context,
                domain_strength= strength,
            )

            assignments[agent] = AgentAssignment(
                agent             = agent,
                role              = role,
                domain_strength   = strength,
                specialized_prompt= specialized_prompt,
                reasoning         = (
                    f"{agent} assigned as {role} "
                    f"(domain={domain}, strength={strength:.0f})"
                ),
            )
            prompt_variants[agent] = specialized_prompt

        logger.info(
            f"[DASR] Routed {len(agents)} agents. "
            f"Domain={primary} (conf={classification.confidence:.0%}). "
            f"Specialists={specialist_agents[:3]}"
        )

        return RouterResult(
            classification   = classification,
            assignments      = assignments,
            prompt_variants  = prompt_variants,
            specialist_agents= specialist_agents,
            generalist_agents= generalist_agents,
        )

    def update_agent_strength(
        self,
        agent:  str,
        domain: str,
        score:  float,
        alpha:  float = 0.2,
    ) -> None:
        """EMA update of agent domain strength from session result."""
        if agent not in self.agent_strengths:
            self.agent_strengths[agent] = {}
        prev = self.agent_strengths[agent].get(domain, 50.0)
        self.agent_strengths[agent][domain] = alpha * score + (1 - alpha) * prev

    def get_router_report(self, result: RouterResult) -> Dict:
        return {
            "primary_domain":    result.classification.primary_domain,
            "secondary_domain":  result.classification.secondary_domain,
            "domain_confidence": round(result.classification.confidence, 2),
            "specialist_agents": result.specialist_agents,
            "generalist_agents": result.generalist_agents,
            "agent_assignments": [
                {
                    "agent":    a,
                    "role":     assign.role,
                    "strength": round(assign.domain_strength, 1),
                    "reasoning":assign.reasoning,
                }
                for a, assign in result.assignments.items()
            ],
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
