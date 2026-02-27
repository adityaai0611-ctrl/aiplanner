# ═══════════════════════════════════════════════════════════════════════════════
# adaptive_prompt_evolution_engine.py — Feature 21: APEE
# Meta-learns which PROMPT TEMPLATES produce the highest-scoring plans.
# Evolves the prompts themselves across sessions using genetic operators.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import random
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Awaitable

logger = logging.getLogger(__name__)

APEE_DB_PATH            = "prompt_evolution.db"
APEE_POPULATION_SIZE    = 12      # prompt variants in the gene pool
APEE_ELITE_K            = 3       # top prompts always survive
APEE_MUTATION_RATE      = 0.20    # per-instruction mutation probability
APEE_CROSSOVER_RATE     = 0.65
APEE_MIN_SESSIONS       = 3       # need this many before evolving
APEE_EMA_ALPHA          = 0.30    # score update weight


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PromptGene:
    """One evolved prompt template."""
    gene_id:       str
    instructions:  List[str]      # ordered list of instruction clauses
    emphasis_map:  Dict[str, float]  # instruction → emphasis weight (0–2)
    generation:    int
    avg_score:     float
    session_count: int
    parent_ids:    List[str]
    mutation_log:  List[str]
    created_at:    str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def render(self, aim: str, steps: str, context: str = "") -> str:
        """Render this gene into an actual prompt string."""
        lines = []
        if context:
            lines.append(context)
        lines.append(f"AIM: {aim}")
        lines.append(f"\nCURRENT STEPS:\n{steps}")
        lines.append("\nINSTRUCTIONS (follow all, weighted by emphasis):")
        for instr in self.instructions:
            weight = self.emphasis_map.get(instr, 1.0)
            stars  = "★" * round(weight)
            lines.append(f"  {stars} {instr}")
        lines.append(
            "\nGenerate a complete, numbered execution plan. "
            "Apply highest-starred instructions most heavily."
        )
        return "\n".join(lines)


@dataclass
class APEEResult:
    best_gene:        PromptGene
    population:       List[PromptGene]
    generation:       int
    best_avg_score:   float
    score_improvement:float   # vs baseline gene


# ── Seed Instructions (initial gene pool) ────────────────────────────────────

SEED_INSTRUCTION_POOL: List[str] = [
    # Structural
    "Break each step into: Action → Owner → Output → Success Criterion",
    "Begin every step with a strong action verb (Build, Design, Validate, Deploy)",
    "Sequence steps so each builds on the previous — no orphan steps",
    "Group steps into phases: Discovery, Design, Build, Test, Launch",
    "State explicit entry and exit conditions for each step",
    # Quality
    "Every step must have a measurable deliverable, not vague activities",
    "Include risk flags inline for high-dependency steps",
    "Assign a time estimate (hours or days) to every step",
    "Specify the human role or team responsible for each step",
    "Add a verification step after every 3 implementation steps",
    # Completeness
    "Do not skip setup, onboarding, or teardown steps",
    "Include stakeholder communication steps at major milestones",
    "Explicitly address failure recovery for the 2 riskiest steps",
    "Add resource acquisition steps before steps that need them",
    "Include a final review and sign-off step",
    # Depth
    "Go 3 levels deep on the most technically complex step",
    "Cite specific tools, APIs, or frameworks for every technical step",
    "Quantify all estimates — no ranges like 'a few days'",
    "Anticipate the top 3 blockers and address them proactively",
    "Cross-reference steps that share resources to flag conflicts",
    # Innovation
    "Include at least one unconventional approach not obvious from the aim",
    "Challenge the most expensive step — can it be 50% cheaper?",
    "Identify which step could be parallelised for fastest delivery",
    "Propose an MVP version of the plan in the first 3 steps",
    "End with a retrospective step to capture lessons learned",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_APEE_MUTATOR = """You are a PROMPT ENGINEER performing genetic mutation.

CURRENT PROMPT INSTRUCTION (to mutate):
"{instruction}"

MUTATION TYPE: {mutation_type}
  rephrase    — restate the same intent more powerfully
  strengthen  — make the instruction more specific and demanding
  generalise  — make it apply to more plan types
  split       — split into two more focused instructions
  combine     — combine with a related concept (provided)

RELATED CONCEPT (for combine only): {related_concept}

CONTEXT: This instruction is used in AI planning prompts. Higher-scoring plans are
more specific, actionable, and comprehensive.

Output ONLY the mutated instruction text (no JSON, no explanation, single sentence):"""

PROMPT_APEE_CROSSOVER = """You are a PROMPT CROSSOVER SPECIALIST.

PARENT A INSTRUCTIONS:
{parent_a_instructions}

PARENT B INSTRUCTIONS:
{parent_b_instructions}

Both parents have produced good plans (A_score={score_a:.1f}, B_score={score_b:.1f}).
Create ONE offspring by intelligently combining the best instructions from both parents.

Rules:
  1. Take the top-performing instructions from each parent
  2. Resolve conflicts between contradictory instructions (keep the stronger one)
  3. Produce exactly {target_count} instructions total
  4. Order them from most to least impactful

Output ONLY a JSON list of instruction strings:
["instruction_1", "instruction_2", ...]"""


# ── Engine ────────────────────────────────────────────────────────────────────

class AdaptivePromptEvolutionEngine:
    """
    Treats prompt templates as evolving genomes.

    Each session:
    1. Sample a prompt gene from the population
    2. Generate plan using that gene
    3. Record score → update gene's avg_score
    4. Every N sessions → evolve population (mutate/crossover/elitism)

    Over time: prompts that produce high-scoring plans survive and reproduce.
    """

    def __init__(
        self,
        call_fn:  Callable[[str, str], Awaitable[str]],
        db_path:  str = APEE_DB_PATH,
        agent:    str = "gemini",
    ):
        self.call_fn = call_fn
        self.db_path = db_path
        self.agent   = agent
        self._population: List[PromptGene] = []
        self._generation = 0
        self._init_db()
        self._population = self._load_population()
        if not self._population:
            self._population = self._seed_population()
            self._save_population()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_genes (
                    gene_id       TEXT PRIMARY KEY,
                    instructions  TEXT,
                    emphasis_map  TEXT,
                    generation    INTEGER,
                    avg_score     REAL DEFAULT 0.0,
                    session_count INTEGER DEFAULT 0,
                    parent_ids    TEXT DEFAULT '[]',
                    mutation_log  TEXT DEFAULT '[]',
                    created_at    TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS apee_sessions (
                    session_id TEXT,
                    gene_id    TEXT,
                    score      REAL,
                    timestamp  TEXT
                )
            """)
            conn.commit()

    def _load_population(self) -> List[PromptGene]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_genes ORDER BY avg_score DESC"
            ).fetchall()
        genes = []
        for row in rows:
            try:
                genes.append(PromptGene(
                    gene_id      = row[0],
                    instructions = json.loads(row[1]),
                    emphasis_map = json.loads(row[2]),
                    generation   = row[3],
                    avg_score    = row[4],
                    session_count= row[5],
                    parent_ids   = json.loads(row[6]),
                    mutation_log = json.loads(row[7]),
                    created_at   = row[8],
                ))
            except Exception:
                continue
        return genes

    def _save_population(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for g in self._population:
                conn.execute(
                    "INSERT OR REPLACE INTO prompt_genes VALUES (?,?,?,?,?,?,?,?,?)",
                    (g.gene_id, json.dumps(g.instructions),
                     json.dumps(g.emphasis_map), g.generation,
                     g.avg_score, g.session_count,
                     json.dumps(g.parent_ids), json.dumps(g.mutation_log),
                     g.created_at)
                )
            conn.commit()

    # ── Seed ──────────────────────────────────────────────────────────────────

    def _seed_population(self) -> List[PromptGene]:
        """Create initial diverse population by sampling from instruction pool."""
        genes = []
        pool  = list(SEED_INSTRUCTION_POOL)
        random.shuffle(pool)
        n_per_gene = max(5, len(pool) // APEE_POPULATION_SIZE)

        for i in range(APEE_POPULATION_SIZE):
            start = i * n_per_gene % len(pool)
            instrs= pool[start:start + n_per_gene] or pool[:n_per_gene]
            # Random emphasis weights
            emphasis = {instr: round(random.uniform(0.5, 2.0), 1) for instr in instrs}
            gene = PromptGene(
                gene_id      = f"gene_{i+1}_g0",
                instructions = instrs,
                emphasis_map = emphasis,
                generation   = 0,
                avg_score    = 0.0,
                session_count= 0,
                parent_ids   = [],
                mutation_log = ["seeded"],
            )
            genes.append(gene)
        logger.info(f"[APEE] Seeded {len(genes)} initial prompt genes")
        return genes

    # ── Selection ─────────────────────────────────────────────────────────────

    def select_gene(self, strategy: str = "tournament") -> PromptGene:
        """
        Select one gene for use in this session.
        Balances exploitation (high scores) vs exploration (low sessions).
        """
        if not self._population:
            self._population = self._seed_population()

        if strategy == "tournament":
            k        = min(4, len(self._population))
            sample   = random.sample(self._population, k)
            # UCB1-style selection: score + exploration bonus
            def ucb(g: PromptGene) -> float:
                exploit = g.avg_score
                explore = 10.0 / max(g.session_count, 1) ** 0.5
                return exploit + explore
            return max(sample, key=ucb)

        elif strategy == "best":
            return max(self._population, key=lambda g: g.avg_score)

        # Random fallback
        return random.choice(self._population)

    def render_best_prompt(self, aim: str, steps: str, context: str = "") -> Tuple[str, str]:
        """Select best gene and render prompt. Returns (prompt, gene_id)."""
        gene   = self.select_gene()
        prompt = gene.render(aim, steps, context)
        return prompt, gene.gene_id

    # ── Score Recording ───────────────────────────────────────────────────────

    def record_session_score(self, gene_id: str, score: float, session_id: str) -> None:
        """Update gene's EMA score after a session."""
        for gene in self._population:
            if gene.gene_id == gene_id:
                if gene.session_count == 0:
                    gene.avg_score = score
                else:
                    gene.avg_score = APEE_EMA_ALPHA * score + (1 - APEE_EMA_ALPHA) * gene.avg_score
                gene.session_count += 1
                break

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO apee_sessions VALUES (?,?,?,?)",
                (session_id, gene_id, score, datetime.utcnow().isoformat())
            )
            conn.commit()
        self._save_population()

    # ── Mutation ──────────────────────────────────────────────────────────────

    async def mutate_gene(self, gene: PromptGene) -> PromptGene:
        """Apply random mutations to a gene's instructions."""
        new_instructions = list(gene.instructions)
        new_emphasis     = dict(gene.emphasis_map)
        mutation_log     = []
        mutation_types   = ["rephrase", "strengthen", "generalise", "combine"]

        for i, instr in enumerate(new_instructions):
            if random.random() >= APEE_MUTATION_RATE:
                continue
            mtype = random.choice(mutation_types)
            related = random.choice(SEED_INSTRUCTION_POOL) if mtype == "combine" else ""

            prompt = PROMPT_APEE_MUTATOR.format(
                instruction     = instr[:200],
                mutation_type   = mtype,
                related_concept = related[:100],
            )
            try:
                mutated = await self.call_fn(self.agent, prompt)
                mutated = mutated.strip().strip('"').strip("'")
                if mutated and len(mutated) > 10:
                    new_instructions[i] = mutated
                    new_emphasis[mutated] = new_emphasis.get(instr, 1.0)
                    mutation_log.append(f"{mtype}:{instr[:40]}→{mutated[:40]}")
            except Exception:
                # Heuristic mutation: add emphasis word
                boosters = ["ALWAYS", "MUST", "Explicitly", "Specifically"]
                new_instructions[i] = f"{random.choice(boosters)} {instr}"
                mutation_log.append(f"heuristic_boost:{instr[:40]}")

        # Random emphasis weight shift
        for instr in new_emphasis:
            if random.random() < 0.15:
                new_emphasis[instr] = round(
                    max(0.5, min(2.0, new_emphasis[instr] + random.uniform(-0.5, 0.5))), 1
                )

        new_gene = PromptGene(
            gene_id      = f"gene_mut_{gene.gene_id}_{self._generation}",
            instructions = new_instructions,
            emphasis_map = new_emphasis,
            generation   = self._generation + 1,
            avg_score    = gene.avg_score * 0.8,  # discount inherited score
            session_count= 0,
            parent_ids   = [gene.gene_id],
            mutation_log = mutation_log,
        )
        return new_gene

    # ── Crossover ─────────────────────────────────────────────────────────────

    async def crossover_genes(
        self,
        parent_a: PromptGene,
        parent_b: PromptGene,
    ) -> PromptGene:
        """Produce offspring by combining instructions from two parents."""
        target_count = max(
            len(parent_a.instructions),
            len(parent_b.instructions)
        ) // 2 + 3

        all_a = "\n".join(f"  • {i}" for i in parent_a.instructions)
        all_b = "\n".join(f"  • {i}" for i in parent_b.instructions)

        prompt = PROMPT_APEE_CROSSOVER.format(
            parent_a_instructions = all_a[:800],
            parent_b_instructions = all_b[:800],
            score_a               = parent_a.avg_score,
            score_b               = parent_b.avg_score,
            target_count          = target_count,
        )
        try:
            raw = await self.call_fn(self.agent, prompt)
            # Extract JSON array
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                instructions = json.loads(match.group())
                if isinstance(instructions, list) and instructions:
                    # Merge emphasis maps from both parents
                    emphasis = {}
                    for instr in instructions:
                        ea = parent_a.emphasis_map.get(instr, 1.0)
                        eb = parent_b.emphasis_map.get(instr, 1.0)
                        emphasis[instr] = round((ea + eb) / 2, 1)

                    return PromptGene(
                        gene_id      = f"gene_cx_{parent_a.gene_id[:8]}_{parent_b.gene_id[:8]}",
                        instructions = instructions[:15],
                        emphasis_map = emphasis,
                        generation   = self._generation + 1,
                        avg_score    = (parent_a.avg_score + parent_b.avg_score) / 2,
                        session_count= 0,
                        parent_ids   = [parent_a.gene_id, parent_b.gene_id],
                        mutation_log = ["crossover"],
                    )
        except Exception as e:
            logger.warning(f"[APEE] Crossover failed: {e}")

        # Fallback: split and merge
        half_a = parent_a.instructions[:len(parent_a.instructions)//2]
        half_b = parent_b.instructions[len(parent_b.instructions)//2:]
        merged = list(dict.fromkeys(half_a + half_b))[:12]
        emphasis = {i: 1.0 for i in merged}
        return PromptGene(
            gene_id      = f"gene_fallback_cx_{self._generation}",
            instructions = merged,
            emphasis_map = emphasis,
            generation   = self._generation + 1,
            avg_score    = (parent_a.avg_score + parent_b.avg_score) / 2,
            session_count= 0,
            parent_ids   = [parent_a.gene_id, parent_b.gene_id],
            mutation_log = ["fallback_crossover"],
        )

    # ── Evolution Step ────────────────────────────────────────────────────────

    async def evolve_population(self) -> APEEResult:
        """
        One generation of evolution:
        1. Elitism: top-K survive unchanged
        2. Crossover: breed new offspring from top performers
        3. Mutation: mutate mid-tier genes
        4. Replace bottom genes
        """
        pop = sorted(self._population, key=lambda g: g.avg_score, reverse=True)
        self._generation += 1

        # Count only genes with real sessions
        experienced = [g for g in pop if g.session_count >= 1]
        if len(experienced) < APEE_MIN_SESSIONS:
            logger.info(f"[APEE] Only {len(experienced)} experienced genes. Skipping evolution.")
            return APEEResult(
                best_gene         = pop[0],
                population        = pop,
                generation        = self._generation,
                best_avg_score    = pop[0].avg_score,
                score_improvement = 0.0,
            )

        # Elitism
        elite       = pop[:APEE_ELITE_K]
        baseline    = elite[0].avg_score
        new_pop     = list(elite)

        # Crossover zone (next tier)
        crossover_candidates = pop[APEE_ELITE_K:APEE_ELITE_K + 6]
        crossover_tasks = []
        for i in range(0, len(crossover_candidates) - 1, 2):
            if random.random() < APEE_CROSSOVER_RATE:
                crossover_tasks.append(
                    self.crossover_genes(crossover_candidates[i], crossover_candidates[i+1])
                )

        offspring = await asyncio.gather(*crossover_tasks)
        new_pop.extend(offspring)

        # Mutation zone
        mutation_candidates = pop[APEE_ELITE_K + 6:]
        mutation_tasks = [
            self.mutate_gene(g)
            for g in mutation_candidates[:4]
        ]
        mutants = await asyncio.gather(*mutation_tasks)
        new_pop.extend(mutants)

        # Pad with fresh seeds if population shrank
        while len(new_pop) < APEE_POPULATION_SIZE:
            new_pop.extend(self._seed_population()[:1])

        self._population = new_pop[:APEE_POPULATION_SIZE]
        self._save_population()

        best = max(self._population, key=lambda g: g.avg_score)
        improvement = best.avg_score - baseline

        logger.info(
            f"[APEE] Generation {self._generation}: "
            f"pop={len(self._population)} best_score={best.avg_score:.1f} "
            f"improvement={improvement:+.1f}"
        )
        return APEEResult(
            best_gene         = best,
            population        = self._population,
            generation        = self._generation,
            best_avg_score    = best.avg_score,
            score_improvement = improvement,
        )

    def get_apee_report(self) -> Dict:
        pop = sorted(self._population, key=lambda g: g.avg_score, reverse=True)
        return {
            "generation":        self._generation,
            "population_size":   len(self._population),
            "best_gene_id":      pop[0].gene_id if pop else None,
            "best_avg_score":    round(pop[0].avg_score, 2) if pop else 0,
            "score_distribution":[round(g.avg_score, 1) for g in pop],
            "experienced_genes": sum(1 for g in pop if g.session_count >= 1),
            "top_instructions":  pop[0].instructions[:5] if pop else [],
        }
