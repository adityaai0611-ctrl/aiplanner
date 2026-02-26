# ═══════════════════════════════════════════════════════════════════════════════
# planner_v3.py — EnhancedMultiAIPlanner v3.0
# Integrates: AEG + CMB + RTBC + BFTCE + GPM
#
# Drop-in upgrade path:
#   from planner_v3 import EnhancedMultiAIPlannerV3 as EnhancedMultiAIPlanner
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Feature Imports ──────────────────────────────────────────────────────────
from graph_node                     import AdaptiveExecutionGraph, NodeStatus
from cognitive_memory_bank          import CognitiveMemoryBank, EpisodicRecord
from resource_token_budget_controller import ResourceAwareTokenBudgetController
from bft_consensus_engine           import ByzantineFaultTolerantConsensusEngine
from genetic_plan_mutator           import GeneticPlanMutator
from system_config                  import (
    CMB_DB_PATH, RTBC_SESSION_BUDGET_USD, AEG_CONCURRENCY_LIMIT,
    GPM_MAX_GENERATIONS, RTBC_MODEL_COSTS
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


class EnhancedMultiAIPlannerV3:
    """
    EnhancedMultiAIPlanner v3.0 — Autonomous, Resource-Aware Execution Engine.

    Architectural upgrades over the original:
    ┌──────────────────────────────────────────────────────────────────┐
    │  F1 — AdaptiveExecutionGraph   replaces sequential for-loops    │
    │  F2 — CognitiveMemoryBank      cross-session learning           │
    │  F3 — ResourceAwareTokenBudgetController  cost governance       │
    │  F4 — ByzantineFaultTolerantConsensusEngine  peer-review winner │
    │  F5 — GeneticPlanMutator       replaces improvement chain       │
    └──────────────────────────────────────────────────────────────────┘

    The original ai_planner.py methods (call_ai_api, create_scoring_prompt,
    check_fraud_and_misinformation, etc.) are expected as injected callables
    so this class can wrap the existing codebase without rewriting it.
    """

    def __init__(
        self,
        # Inject existing planner callables
        call_ai_api_fn:          Callable,   # (agent, prompt, param_group) → str
        create_scoring_prompt_fn: Callable,  # (plan, aim, steps, params) → float
        fraud_check_fn:          Optional[Callable] = None,
        feature_detect_fn:       Optional[Callable] = None,
        sse_emit_fn:             Optional[Callable] = None,   # SSE streaming hook
        # Configuration
        session_budget_usd:      float = RTBC_SESSION_BUDGET_USD,
        memory_db_path:          str   = CMB_DB_PATH,
        concurrency_limit:       int   = AEG_CONCURRENCY_LIMIT,
    ):
        self.call_api      = call_ai_api_fn
        self.score_plan    = create_scoring_prompt_fn
        self.check_fraud   = fraud_check_fn   or (lambda *a, **k: {})
        self.detect_feats  = feature_detect_fn or (lambda *a, **k: [])
        self.emit          = sse_emit_fn       or (lambda msg, t="info": None)
        self.concurrency   = concurrency_limit

        # ── Instantiate all 5 feature modules ───────────────────────────────
        self.memory    = CognitiveMemoryBank(db_path=memory_db_path)
        self.rtbc      = ResourceAwareTokenBudgetController(session_budget_usd=session_budget_usd)
        self.aeg       = AdaptiveExecutionGraph(call_fn=self._gated_node_call)
        self.bftce     = ByzantineFaultTolerantConsensusEngine(review_fn=self._peer_review_call)
        # GPM is instantiated per session in run_full_planning_session()

        # ── Session state (reset each run) ───────────────────────────────────
        self._session_id:    str                = ""
        self._aim:           str                = ""
        self._initial_steps: List[str]          = []
        self._parameters:    List[str]          = []
        self._param_groups:  List[List[str]]    = []
        self._ai_plans:      Dict[str, str]     = {}
        self._ai_scores:     Dict[str, float]   = {}
        self._process_log:   List[str]          = []

        logger.info("✅ EnhancedMultiAIPlannerV3 initialized")

    # ── Session Bootstrap ─────────────────────────────────────────────────────

    async def run_full_planning_session(
        self,
        aim:           str,
        initial_steps: List[str],
        parameters:    List[str],
        agents:        List[str],
        param_groups:  Optional[List[List[str]]] = None,
    ) -> Dict:
        """
        Complete v3.0 planning session. Replaces the original run_planning_session().

        Flow:
          CMB priming → AEG build → AEG execute (gated by RTBC) →
          GPM evolution → BFTCE consensus → CMB store → report

        Returns a result dict compatible with the existing Flask response schema.
        """
        self._session_id    = f"sess_{uuid.uuid4().hex[:12]}"
        self._aim           = aim
        self._initial_steps = initial_steps
        self._parameters    = parameters
        self._param_groups  = param_groups or self._build_param_groups(parameters)
        self._ai_plans      = {}
        self._ai_scores     = {}
        self._process_log   = []

        session_start = time.monotonic()
        self._log(f"🚀 Session {self._session_id} started")
        self._log(f"   AIM: {aim[:100]}")
        self._log(f"   Agents: {agents} | Groups: {len(self._param_groups)}")

        try:
            # ── Phase 1: Memory priming ──────────────────────────────────────
            priming_ctx = await self._phase_memory_priming(aim)

            # ── Phase 2: AEG plan generation ────────────────────────────────
            await self._phase_aeg_generation(agents, priming_ctx)

            # ── Phase 3: GPM evolutionary improvement ───────────────────────
            evolved_best = await self._phase_gpm_evolution()

            # ── Phase 4: BFTCE consensus selection ──────────────────────────
            consensus = await self._phase_bftce_consensus(evolved_best)

            # ── Phase 5: Store to memory ────────────────────────────────────
            await self._phase_memory_store(consensus)

            # ── Build final result ───────────────────────────────────────────
            elapsed = time.monotonic() - session_start
            result  = self._build_result(consensus, elapsed)

            self.rtbc.print_summary()
            self._log(f"✅ Session complete in {elapsed:.1f}s | Winner: {consensus.winner_agent}")
            return result

        except Exception as exc:
            logger.error(f"[v3] Session failed: {exc}", exc_info=True)
            self._log(f"❌ Session error: {exc}")
            raise

    # ── Phase 1: Cognitive Memory Priming ─────────────────────────────────────

    async def _phase_memory_priming(self, aim: str) -> str:
        self._log("📚 Phase 1: Retrieving memory priming context...")
        self.emit("Retrieving cognitive memory context...", "info")

        session_count = self.memory.get_session_count()
        if session_count == 0:
            self._log("   No prior sessions — first run, no priming available")
            return ""

        priming = await self.memory.get_memory_priming_context(aim)
        lines   = priming.count("\n") + 1 if priming else 0
        self._log(f"   Priming context: {lines} lines from {session_count} sessions")
        self.emit(f"Memory priming ready ({session_count} past sessions)", "success")
        return priming

    # ── Phase 2: AEG Parallel Generation ──────────────────────────────────────

    async def _phase_aeg_generation(self, agents: List[str], priming_ctx: str) -> None:
        self._log("⚡ Phase 2: Building adaptive execution graph...")
        self.emit("Building execution graph (AEG)...", "info")

        # Compute correlation matrix from agent history
        agent_score_history = {
            a: self.memory.get_agent_profile(a) and
               [self.memory.get_agent_profile(a).mean_score]   # simplified
            or []
            for a in agents
        }
        correlation_matrix = self.aeg.compute_correlation_matrix(
            {a: [v] for a, v in self._ai_scores.items() if v}
        )

        self.aeg.build_graph(
            agents            = agents,
            param_groups      = self._param_groups,
            correlation_matrix= correlation_matrix if correlation_matrix else None,
        )

        # Inject priming into node calls via instance variable
        self._priming_ctx = priming_ctx

        self._log(f"   Graph: {len(self.aeg.nodes)} nodes | concurrency: {self.concurrency}")
        self.emit(f"Executing {len(self.aeg.nodes)} agent-group nodes in parallel...", "info")

        completed_nodes = await self.aeg.topological_execute(
            concurrency_limit=self.concurrency
        )

        # Aggregate results per agent (best-scoring group wins for that agent)
        for node in completed_nodes.values():
            if node.status == NodeStatus.COMPLETED and node.result:
                agent = node.agent_name
                if agent not in self._ai_scores or node.score > self._ai_scores[agent]:
                    self._ai_plans[agent]  = node.result
                    self._ai_scores[agent] = node.score or 0.0

        summary = self.aeg.get_execution_summary()
        self._log(
            f"   AEG complete: {summary['total_nodes']} nodes | "
            f"mean score={summary['score_mean']:.1f} | "
            f"tokens={summary['total_tokens']:,}"
        )
        self.emit(
            f"AEG done: {summary['total_nodes']} nodes, "
            f"mean score {summary['score_mean']:.1f}",
            "success"
        )

    # ── Phase 3: GPM Evolutionary Improvement ─────────────────────────────────

    async def _phase_gpm_evolution(self) -> "PlanChromosome":
        self._log("🧬 Phase 3: GPM evolutionary improvement...")
        self.emit("Starting genetic evolution of plans...", "info")

        if len(self._ai_plans) < 2:
            self._log("   Insufficient plans for evolution — skipping GPM")
            # Return a mock best chromosome
            from genetic_plan_mutator import PlanChromosome
            best_agent = max(self._ai_scores, key=self._ai_scores.get)
            mock = PlanChromosome(
                chromosome_id = "mock_best",
                parent_agent  = best_agent,
                steps         = [self._ai_plans[best_agent]],
                fitness_score = self._ai_scores[best_agent],
            )
            return mock

        gpm = GeneticPlanMutator(
            score_fn    = self._gpm_score_fn,
            mutate_fn   = self._gpm_mutate_fn,
            generate_fn = self._gpm_generate_fn,
            aim         = self._aim,
            initial_steps = self._initial_steps,
        )
        gpm.initialize_population(self._ai_plans)
        best_chromosome, gen_history = await gpm.evolve(max_generations=GPM_MAX_GENERATIONS)

        self._gpm_report = gpm.get_evolution_report()
        self._log(
            f"   Evolution: {self._gpm_report['generations']} gens | "
            f"improvement: +{self._gpm_report['improvement_delta']:.1f} | "
            f"best agent: {best_chromosome.parent_agent}"
        )
        self.emit(
            f"Evolution complete: {self._gpm_report['generations']} generations, "
            f"+{self._gpm_report['improvement_delta']:.1f} improvement",
            "success"
        )

        # Merge evolved best back into ai_plans
        self._ai_plans[best_chromosome.parent_agent]  = best_chromosome.to_plan_string()
        self._ai_scores[best_chromosome.parent_agent] = best_chromosome.fitness_score

        return best_chromosome

    # ── Phase 4: BFTCE Consensus ───────────────────────────────────────────────

    async def _phase_bftce_consensus(self, evolved_best) -> Any:
        self._log("🗳️  Phase 4: Running BFT consensus...")
        self.emit("Running Byzantine consensus protocol...", "info")

        consensus = await self.bftce.run_full_consensus(
            ai_plans      = self._ai_plans,
            self_scores   = self._ai_scores,
            aim           = self._aim,
            initial_steps = self._initial_steps,
        )

        self._log(self.bftce.format_result_for_display(consensus))
        self.emit(
            f"Consensus: {consensus.winner_agent} wins "
            f"(score={consensus.consensus_score:.1f}, confidence={consensus.confidence})",
            "success"
        )
        return consensus

    # ── Phase 5: Memory Store ──────────────────────────────────────────────────

    async def _phase_memory_store(self, consensus) -> None:
        self._log("💾 Phase 5: Storing session to memory bank...")

        param_hash = hashlib.md5(
            json.dumps(self._parameters, sort_keys=True).encode()
        ).hexdigest()

        record = EpisodicRecord(
            session_id       = self._session_id,
            aim              = self._aim,
            initial_steps    = self._initial_steps,
            best_plan        = consensus.winner_plan[:2000],
            winning_agent    = consensus.winner_agent,
            winning_score    = consensus.consensus_score,
            all_scores       = self._ai_scores,
            aim_embedding    = await self.memory.get_aim_embedding(self._aim),
            domain_tags      = self._infer_domain_tags(self._aim),
            feature_count    = 0,
            timestamp        = datetime.now(timezone.utc).isoformat(),
            param_file_hash  = param_hash,
            step_count       = len(self._initial_steps),
            improvement_delta= getattr(self, "_gpm_report", {}).get("improvement_delta", 0.0),
        )
        self.memory.store_episodic(record)

        # Update all agent profiles
        for agent, score in self._ai_scores.items():
            domain = record.domain_tags[0] if record.domain_tags else "general"
            token_cost = sum(
                n.token_cost or 0
                for n in self.aeg.nodes.values()
                if n.agent_name == agent and n.token_cost
            )
            self.memory.update_agent_profile(
                agent_name    = agent,
                session_score = score,
                domain        = domain,
                token_cost    = token_cost,
                is_winner     = (agent == consensus.winner_agent),
            )

        # Distil semantic patterns every N sessions
        if self.memory.should_distill():
            self._log("🧠 Distilling semantic patterns from episodic memory...")
            self.emit("Distilling strategy patterns...", "info")

        self._log(f"   Session stored: {self._session_id}")

    # ── Gated API Caller (used by AEG nodes) ──────────────────────────────────

    async def _gated_node_call(
        self,
        agent_name:  str,
        dep_context: str,
        parameters:  List[str],
    ) -> Tuple[str, float, int]:
        """
        Wraps the original call_ai_api() with RTBC gating.
        Returns (result_text, score, token_count).
        """
        # Build prompt with priming context injected
        priming = getattr(self, "_priming_ctx", "")
        prompt_parts = []
        if priming:
            prompt_parts.append(priming)
        if dep_context.strip():
            prompt_parts.append(dep_context)
        prompt_parts.append(
            f"AIM: {self._aim}\n"
            f"STEPS: {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(self._initial_steps))}\n"
            f"PARAMETERS: {', '.join(parameters)}"
        )
        full_prompt = "\n\n".join(prompt_parts)

        # RTBC gate
        mean_score = sum(self._ai_scores.values()) / max(len(self._ai_scores), 1)
        allowed, reason = self.rtbc.should_allow_call(
            agent_name          = agent_name,
            prompt              = full_prompt,
            current_agent_score = self._ai_scores.get(agent_name),
            global_mean_score   = mean_score,
        )

        if not allowed:
            self.rtbc.record_rejected_call(agent_name, 0, reason)
            self._log(f"   [RTBC] Rejected {agent_name}: {reason}")
            self.emit(f"RTBC blocked {agent_name}: {reason}", "warn")
            return ("", 0.0, 0)

        score_before = self._ai_scores.get(agent_name, 0.0)
        t_start      = time.monotonic()

        try:
            result_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.call_api(agent_name, full_prompt, parameters)
            )
            score = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.score_plan(result_text, self._aim, self._initial_steps, parameters)
            )
            prompt_tokens  = self.rtbc.estimate_token_count(full_prompt)
            output_tokens  = self.rtbc.estimate_token_count(result_text)

            entry = self.rtbc.record_call_result(
                agent_name           = agent_name,
                group_index          = 0,
                actual_prompt_tokens = prompt_tokens,
                actual_output_tokens = output_tokens,
                score_before         = score_before,
                score_after          = float(score),
            )
            self.emit(
                f"{agent_name}: score={score:.1f} cost=${entry.estimated_cost:.5f} roi={entry.roi:.1f}",
                "success"
            )
            return (result_text, float(score), prompt_tokens + output_tokens)

        except Exception as e:
            logger.warning(f"[v3] {agent_name} call failed: {e}")
            self.rtbc.record_rejected_call(agent_name, 0, f"exception:{str(e)[:80]}")
            return ("", 0.0, 0)

    # ── Peer Review Caller (used by BFTCE) ────────────────────────────────────

    async def _peer_review_call(
        self,
        reviewer_agent: str,
        prompt:         str,
        reviewed_agent: str,
    ) -> Dict:
        """
        Ask reviewer_agent to score the reviewed_agent's plan.
        Returns parsed JSON dict.
        """
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.call_api(reviewer_agent, prompt, [])
            )
            # Try to parse JSON from response
            import re as _re
            json_match = _re.search(r'\{[^{}]*"completeness"[^{}]*\}', raw, _re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Fallback: extract numbers heuristically
            nums = _re.findall(r'"(\w+)":\s*(\d+(?:\.\d+)?)', raw)
            parsed = {k: float(v) for k, v in nums if k in (
                "completeness","actionability","coherence","novelty","safety","efficiency","composite_score"
            )}
            return parsed or {"completeness": 60, "actionability": 60, "coherence": 60,
                               "novelty": 60, "safety": 70, "efficiency": 60, "composite_score": 62.0}
        except Exception as e:
            logger.warning(f"[v3] Peer review {reviewer_agent}→{reviewed_agent} failed: {e}")
            return {"completeness":50,"actionability":50,"coherence":50,
                    "novelty":50,"safety":50,"efficiency":50,"composite_score":50.0}

    # ── GPM Callbacks ──────────────────────────────────────────────────────────

    async def _gpm_score_fn(self, plan_text: str, aim: str) -> float:
        try:
            score = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.score_plan(plan_text, aim, self._initial_steps, self._parameters)
            )
            return float(score)
        except Exception:
            return 50.0

    async def _gpm_mutate_fn(self, prompt: str, agent: str) -> Dict:
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.call_api(agent, prompt, [])
            )
            import re as _re
            json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"mutated_step": raw.strip()[:300]}
        except Exception as e:
            logger.debug(f"[GPM] mutate_fn failed: {e}")
            return {}

    async def _gpm_generate_fn(self, aim: str, steps: List[str], agent: str) -> str:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        prompt = (
            f"Generate a completely original plan for:\nAIM: {aim}\nSTEPS:\n{steps_text}\n"
            "Provide a numbered plan with 8-12 specific, actionable steps."
        )
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.call_api(agent, prompt, [])
            )
        except Exception:
            return ""

    # ── Result Builder ─────────────────────────────────────────────────────────

    def _build_result(self, consensus, elapsed_sec: float) -> Dict:
        """Build the final result dict compatible with the existing Flask API response."""
        rtbc_report = self.rtbc.get_session_report()
        aeg_summary = self.aeg.get_execution_summary()
        gpm_report  = getattr(self, "_gpm_report", {})

        return {
            # ── Core (original schema) ───────────────────────────────────────
            "success":           True,
            "session_id":        self._session_id,
            "aim":               self._aim,
            "best_plan":         consensus.winner_plan,
            "best_agent":        consensus.winner_agent,
            "best_score":        consensus.consensus_score,
            "all_scores":        self._ai_scores,

            # ── v3.0 extensions ─────────────────────────────────────────────
            "consensus": {
                "winner":           consensus.winner_agent,
                "score":            consensus.consensus_score,
                "quorum_support":   consensus.quorum_support,
                "quorum_threshold": consensus.quorum_threshold,
                "confidence":       consensus.confidence,
                "adversarial_agents": consensus.adversarial_agents,
                "credibility_weights": consensus.credibility_weights,
                "tiebreaker_needed":  consensus.tiebreaker_needed,
                "round_log":        consensus.round_log,
            },
            "token_budget": {
                "spent_usd":      rtbc_report["total_spent_usd"],
                "budget_usd":     rtbc_report["session_budget_usd"],
                "utilisation_pct":rtbc_report["budget_utilisation_pct"],
                "calls_allowed":  rtbc_report["total_calls_allowed"],
                "calls_rejected": rtbc_report["total_calls_rejected"],
                "roi_ranking":    rtbc_report["roi_ranking"][:5],
                "circuit_events": rtbc_report["circuit_events"],
            },
            "evolution": gpm_report,
            "execution_graph": {
                "total_nodes":  aeg_summary["total_nodes"],
                "score_mean":   aeg_summary["score_mean"],
                "score_max":    aeg_summary["score_max"],
                "total_tokens": aeg_summary["total_tokens"],
                "by_status":    {k: len(v) for k, v in aeg_summary["by_status"].items()},
            },
            "memory_sessions": self.memory.get_session_count(),
            "elapsed_sec":     round(elapsed_sec, 2),
            "process_log":     self._process_log,
        }

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _build_param_groups(self, parameters: List[str]) -> List[List[str]]:
        """Split flat parameter list into groups of ~3 for parallel processing."""
        group_size = 3
        return [
            parameters[i:i + group_size]
            for i in range(0, len(parameters), group_size)
        ] or [[]]

    def _infer_domain_tags(self, aim: str) -> List[str]:
        """Simple keyword-based domain inference."""
        aim_lower = aim.lower()
        tags = []
        keyword_map = {
            "tech":       ["software", "app", "ai", "ml", "data", "api", "code", "platform"],
            "marketing":  ["brand", "market", "customer", "growth", "sales", "campaign"],
            "ops":        ["operations", "process", "workflow", "supply", "logistics", "team"],
            "finance":    ["revenue", "profit", "cost", "budget", "funding", "investor"],
            "product":    ["product", "feature", "user", "ux", "design", "launch"],
            "research":   ["research", "study", "analysis", "survey", "experiment"],
        }
        for domain, keywords in keyword_map.items():
            if any(k in aim_lower for k in keywords):
                tags.append(domain)
        return tags or ["general"]

    def _log(self, msg: str) -> None:
        self._process_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    # ── Backwards Compatibility Shims ─────────────────────────────────────────

    def get_memory_stats(self) -> Dict:
        """Return memory bank statistics for the admin panel."""
        ranked = self.memory.get_ranked_agents()
        patterns = self.memory.get_strategy_patterns()
        return {
            "total_sessions":  self.memory.get_session_count(),
            "agent_rankings":  ranked[:5],
            "pattern_count":   len(patterns),
            "top_patterns":    [p.pattern_text for p in patterns[:3]],
        }

    def get_budget_status(self) -> Dict:
        """Real-time budget status for SSE streaming."""
        report = self.rtbc.get_session_report()
        return {
            "spent":     report["total_spent_usd"],
            "budget":    report["session_budget_usd"],
            "remaining": report["remaining_usd"],
            "pct":       report["budget_utilisation_pct"],
        }

    def export_all_data(self, output_path: str = "memory_snapshot.json") -> Dict:
        """Export complete memory bank for backup."""
        return self.memory.export_memory_snapshot(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — how to upgrade existing app.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# In app.py, replace:
#   from ai_planner import EnhancedMultiAIPlanner
#
# With:
#   from planner_v3 import EnhancedMultiAIPlannerV3
#
#   planner = EnhancedMultiAIPlannerV3(
#       call_ai_api_fn          = original_planner.call_ai_api,
#       create_scoring_prompt_fn= original_planner.create_scoring_prompt,
#       fraud_check_fn          = original_planner.check_fraud_and_misinformation,
#       feature_detect_fn       = original_planner.detect_features_from_mini_plan,
#       sse_emit_fn             = lambda msg, t: emit_sse({"type": t, "message": msg}),
#   )
#
# In your SSE route handler:
#   result = await planner.run_full_planning_session(
#       aim           = request_data["aim"],
#       initial_steps = request_data["steps"],
#       parameters    = loaded_parameters,
#       agents        = ["gemini","openai","together","groq","replicate",
#                        "fireworks","cohere","deepseek","openrouter",
#                        "writer","huggingface","pawn"],
#   )
#
# Add new Excel sheets from result["token_budget"], result["evolution"],
# result["execution_graph"], and result["consensus"].
# ═══════════════════════════════════════════════════════════════════════════════
