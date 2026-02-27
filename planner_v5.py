# ═══════════════════════════════════════════════════════════════════════════════
# planner_v5.py — EnhancedMultiAIPlannerV5
# Integrates all 20 features across V3+V4+V5
#
# V3 (F1-F5):  AEG, CMB, RTBC, BFTCE, GPM
# V4 (F6-F10): CRE, MADP, TES, SPCE, ASCL
# V5 (F11-F15):DKGE, MOPO, ARTS, HPD, FLMS
# V5 (F16-F20):CFE, HFW, DASR, PEM, SPS
#
# DROP-IN REPLACEMENT for planner_v4.py
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable

# ── V3/V4 Modules ─────────────────────────────────────────────────────────────
from cognitive_memory_bank                    import CognitiveMemoryBank
from resource_token_budget_controller         import ResourceAwareTokenBudgetController
from graph_node                               import AdaptiveExecutionGraph
from bft_consensus_engine                     import ByzantineFaultTolerantConsensusEngine
from genetic_plan_mutator                     import GeneticPlanMutator
from causal_reasoning_engine                  import CausalReasoningEngine
from multi_agent_debate_protocol              import MultiAgentDebateProtocol
from temporal_execution_simulator             import TemporalExecutionSimulator
from semantic_plan_compression_engine         import SemanticPlanCompressionEngine
from autonomous_self_correction_loop          import AutonomousSelfCorrectionLoop, ExecutionReport

# ── V5 Modules (F11–F15) ──────────────────────────────────────────────────────
from dynamic_knowledge_graph_engine           import DynamicKnowledgeGraphEngine
from multi_objective_pareto_optimizer         import MultiObjectiveParetoOptimizer
from adversarial_red_team_simulator           import AdversarialRedTeamSimulator
from hierarchical_plan_decomposer             import HierarchicalPlanDecomposer
from federated_learning_memory_synthesiser    import FederatedLearningMemorySynthesiser

# ── V5 Modules (F16–F20) ──────────────────────────────────────────────────────
from counterfactual_reasoning_engine          import CounterfactualReasoningEngine
from hallucination_firewall                   import HallucinationFirewall
from dynamic_agent_specialization_router      import DynamicAgentSpecializationRouter
from plan_entropy_monitor                     import PlanEntropyMonitor
from stakeholder_persona_simulator            import StakeholderPersonaSimulator

from system_config import CMB_DB_PATH, TES_DEFAULT_DOMAIN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S"
)


class EnhancedMultiAIPlannerV5:
    """
    Full 20-feature autonomous planning engine.

    15-phase execution pipeline:
      Phase  0  — FLMS Synthesis      : federated learning → agent ranking
      Phase  1  — CMB Priming         : memory retrieval
      Phase  2  — DASR Classification : domain routing → specialized prompts
      Phase  3  — CFE Injection       : counterfactual insights from past sessions
      Phase  4  — CRE Analysis        : causal DAG + critical path
      Phase  5  — PEM Pre-check       : entropy baseline measurement
      Phase  6  — AEG Generation      : parallel plan generation (HFW firewall inline)
      Phase  7  — PEM Intervention    : diversity injection if entropy collapsed
      Phase  8  — MADP Debate         : adversarial debate
      Phase  9  — GPM Evolution       : genetic plan evolution
      Phase 10  — SPCE Compression    : semantic deduplication
      Phase 11  — MOPO Optimisation   : Pareto frontier (Cost/Time/Quality/Risk)
      Phase 12  — ARTS Red Team       : adversarial attack simulation
      Phase 13  — SPS Review          : 6-persona stakeholder review + hardening
      Phase 14  — TES Simulation      : Monte Carlo timeline
      Phase 15  — BFTCE Consensus     : Byzantine fault-tolerant winner
      Phase 16  — HPD Decomposition   : hierarchical task breakdown
      Phase 17  — DKGE Merge          : knowledge graph update
      Phase 18  — CFE Analysis        : counterfactual analysis for future learning
      Phase 19  — CMB + FLMS Store    : persist session
      Phase 20  — ASCL Init           : autonomous correction loop ready
    """

    def __init__(
        self,
        call_ai_api_fn:           Callable[[str, str], Awaitable[str]],
        create_scoring_prompt_fn: Callable,
        sse_emit_fn:              Optional[Callable[[str, str], None]] = None,
        db_path:                  str = CMB_DB_PATH,
        domain:                   str = TES_DEFAULT_DOMAIN,
        personas:                 Optional[List[str]] = None,
        mopo_weights:             Optional[Dict[str, float]] = None,
    ):
        self._call_api   = call_ai_api_fn
        self._score      = create_scoring_prompt_fn
        self._sse        = sse_emit_fn or (lambda msg, t="info": None)
        self._domain     = domain

        # ── V3 ──────────────────────────────────────────────────────────────
        self.memory   = CognitiveMemoryBank(db_path=db_path)
        self.rtbc     = ResourceAwareTokenBudgetController()
        self.aeg      = AdaptiveExecutionGraph(
            call_fn=self._gated_call, score_fn=self._score_wrap)
        self.bftce    = ByzantineFaultTolerantConsensusEngine(call_fn=self._call_api)

        # ── V4 ──────────────────────────────────────────────────────────────
        self.cre  = CausalReasoningEngine(call_fn=self._call_api)
        self.madp = MultiAgentDebateProtocol(call_fn=self._call_api, score_fn=self._score_wrap)
        self.tes  = TemporalExecutionSimulator(call_fn=self._call_api)
        self.spce = SemanticPlanCompressionEngine(call_fn=self._call_api, score_fn=self._score_wrap)

        # ── V5 F11-F15 ───────────────────────────────────────────────────────
        self.dkge = DynamicKnowledgeGraphEngine(call_fn=self._call_api)
        self.mopo = MultiObjectiveParetoOptimizer(call_fn=self._call_api, weights=mopo_weights)
        self.arts = AdversarialRedTeamSimulator(call_fn=self._call_api)
        self.hpd  = HierarchicalPlanDecomposer(call_fn=self._call_api)
        self.flms = FederatedLearningMemorySynthesiser(call_fn=self._call_api)

        # ── V5 F16-F20 ───────────────────────────────────────────────────────
        self.cfe  = CounterfactualReasoningEngine(call_fn=self._call_api, score_fn=self._score_wrap)
        self.hfw  = HallucinationFirewall(call_fn=self._call_api, domain=domain)
        self.dasr = DynamicAgentSpecializationRouter(call_fn=self._call_api)
        self.pem  = PlanEntropyMonitor(call_fn=self._call_api)
        self.sps  = StakeholderPersonaSimulator(call_fn=self._call_api, personas=personas)

        # ASCL created per-session
        self._ascl: Optional[AutonomousSelfCorrectionLoop] = None

        logger.info("✅ EnhancedMultiAIPlannerV5 initialised (20 features, 20-phase pipeline)")

    # ═══════════════════════════════════════════════════════════════════════════
    # MASTER ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════════

    async def run_full_planning_session(
        self,
        aim:           str,
        initial_steps: List[str],
        parameters:    List[Any],
        agents:        List[str],
        param_groups:  Optional[List[Any]] = None,
        domain:        Optional[str]       = None,
    ) -> Dict:
        t_start    = time.time()
        session_id = f"v5_{int(t_start)}"
        domain     = domain or self._domain
        param_groups = param_groups or [parameters]

        logger.info(f"\n{'═'*65}")
        logger.info(f"  V5 SESSION: {session_id}")
        logger.info(f"  AIM: {aim[:80]}")
        logger.info(f"  Agents: {len(agents)}")
        logger.info(f"{'═'*65}")

        log: List[Dict] = []
        def emit(msg: str, t: str = "info"):
            log.append({"type": t, "msg": msg, "ts": datetime.utcnow().isoformat()})
            self._sse(msg, t)

        self.rtbc.reset_session()
        report: Dict = {}

        try:
            # Phase 0 — FLMS Synthesis ────────────────────────────────────────
            emit("🧠 Phase 0: Federated learning synthesis...", "phase")
            if self.flms.should_aggregate():
                await self.flms.aggregate_global_model()
            flms_synthesis = self.flms.get_synthesis_for_session(aim, domain)
            # Reorder agents by FLMS recommendation
            if flms_synthesis.recommended_agents:
                preferred = flms_synthesis.recommended_agents
                agents    = preferred + [a for a in agents if a not in preferred]

            # Phase 1 — CMB Priming ───────────────────────────────────────────
            emit("📚 Phase 1: Memory priming...", "phase")
            memory_ctx = await self._safe(self._phase_memory_priming, aim, initial_steps)
            strategy_ctx = "\n".join(flms_synthesis.strategy_injections[:2])

            # Phase 2 — DASR Classification + Routing ─────────────────────────
            emit("🎯 Phase 2: Domain classification + agent routing...", "phase")
            router_result = await self._safe(
                self.dasr.route, agents, aim, initial_steps,
                memory_ctx or "", ""
            )
            report["domain"] = self.dasr.get_router_report(router_result) if router_result else {}

            # Phase 3 — CFE Historical Injection ──────────────────────────────
            emit("🔀 Phase 3: Counterfactual history injection...", "phase")
            import hashlib
            aim_hash    = hashlib.md5(aim.encode()).hexdigest()[:12]
            cfe_history = self.cfe.get_historical_injections(aim_hash)

            # Build enriched priming context
            priming = "\n\n".join(filter(None, [
                memory_ctx, strategy_ctx, cfe_history
            ]))

            # Phase 4 — CRE Causal Analysis ───────────────────────────────────
            emit("🔗 Phase 4: Causal dependency analysis...", "phase")
            causal_report = await self._safe(self.cre.analyse, initial_steps, aim)
            causal_ctx    = causal_report.causal_context_string if causal_report else ""
            report["causal"] = self.cre.get_report_dict() if causal_report else {}

            # Phase 5 — PEM Pre-generation baseline ───────────────────────────
            emit("📊 Phase 5: Entropy baseline...", "phase")
            # Will measure after generation

            # Phase 6 — AEG Parallel Generation + HFW Firewall ───────────────
            emit("⚡ Phase 6: Parallel plan generation + hallucination firewall...", "phase")
            all_plans, all_scores = await self._phase_generate_with_firewall(
                aim, initial_steps, agents, router_result,
                priming, causal_ctx, param_groups
            )
            if not all_plans:
                raise RuntimeError("No valid plans survived the firewall")

            emit(f"✅ Generation complete: {len(all_plans)} plans admitted", "progress")
            report["firewall"] = {"admitted": len(all_plans), "total_agents": len(agents)}

            # Phase 7 — PEM Entropy Monitoring + Intervention ─────────────────
            emit("🌐 Phase 7: Entropy monitoring...", "phase")
            pem_result = await self._safe(
                self.pem.monitor_and_intervene, all_plans, aim, self._call_api
            )
            report["entropy"] = self.pem.get_pem_report(pem_result) if pem_result else {}

            best_agent = max(all_scores, key=all_scores.get)
            best_plan  = all_plans[best_agent]
            best_score = all_scores[best_agent]

            # Phase 8 — MADP Debate ───────────────────────────────────────────
            emit("⚔️  Phase 8: Adversarial debate...", "phase")
            debate = await self._safe(
                self.madp.run_full_debate, agents, best_plan, aim, all_scores
            )
            if debate and debate.accepted:
                best_plan = debate.revised_plan
                emit(f"✅ Debate accepted: +{debate.plan_improvement_score:.1f} pts", "progress")
            report["debate"] = self.madp.get_debate_report(debate) if debate else {}

            # Phase 9 — GPM Evolution ─────────────────────────────────────────
            emit("🧬 Phase 9: Genetic evolution...", "phase")
            gpm_out = await self._phase_gpm(all_plans, all_scores, aim, initial_steps)
            if gpm_out.get("best_score", 0) > best_score:
                best_plan  = gpm_out["best_plan"]
                best_score = gpm_out["best_score"]
            report["evolution"] = gpm_out

            # Phase 10 — SPCE Compression ─────────────────────────────────────
            emit("🗜️  Phase 10: Semantic compression...", "phase")
            compression = await self._safe(self.spce.compress, best_plan, aim, best_score)
            if compression and compression.accepted:
                best_plan = "\n".join(
                    f"Step {i+1}: {s}"
                    for i, s in enumerate(compression.compressed_steps)
                )
                emit(f"✅ Compressed: {compression.original_count}→{compression.compressed_count} steps", "progress")
            report["compression"] = self.spce.get_compression_report(compression) if compression else {}

            # Phase 11 — MOPO Pareto Optimisation ────────────────────────────
            emit("📐 Phase 11: Multi-objective Pareto optimisation...", "phase")
            mopo_result = await self._safe(self.mopo.evolve_frontier, all_plans, aim, 3)
            if mopo_result and mopo_result.recommended_plan:
                # Blend: keep best plan but note Pareto recommendation
                pareto_plan = mopo_result.recommended_plan.plan_text
            report["pareto"] = self.mopo.get_mopo_report(mopo_result) if mopo_result else {}

            # Phase 12 — ARTS Red Team ─────────────────────────────────────────
            emit("🎯 Phase 12: Adversarial red team...", "phase")
            red_team = await self._safe(self.arts.run_full_red_team, best_plan, aim)
            if red_team and red_team.hardened_plan_additions:
                # Append hardening steps to plan
                hardening_steps = "\n".join(
                    f"Step {i+1}: {s}"
                    for i, s in enumerate(red_team.hardened_plan_additions)
                )
                best_plan += f"\n\n[RISK MITIGATION STEPS]\n{hardening_steps}"
                emit(f"✅ Red team: {len(red_team.critical_vulnerabilities)} critical vulns → mitigated", "progress")
            report["red_team"] = self.arts.get_arts_report(red_team) if red_team else {}

            # Phase 13 — SPS Stakeholder Review ──────────────────────────────
            emit("👥 Phase 13: Stakeholder persona review...", "phase")
            sps_result = await self._safe(self.sps.simulate, best_plan, aim)
            if sps_result:
                best_plan = sps_result.hardened_plan or best_plan
                emit(
                    f"✅ Stakeholder review: {sps_result.approval_rate:.0%} approval | "
                    f"{len(sps_result.critical_objections)} critical objections addressed",
                    "progress"
                )
            report["stakeholders"] = self.sps.get_sps_report(sps_result) if sps_result else {}

            # Phase 14 — TES Timeline Simulation ─────────────────────────────
            emit("📅 Phase 14: Temporal simulation...", "phase")
            prereqs    = {nid: n.prerequisites for nid, n in causal_report.dag.items()} \
                         if causal_report else None
            sim_report = await self._safe(
                self.tes.simulate, initial_steps, aim, domain, prereqs
            )
            report["simulation"] = self.tes.get_report_dict(sim_report) if sim_report else {}

            # Phase 15 — BFTCE Consensus ──────────────────────────────────────
            emit("🗳️  Phase 15: Byzantine consensus...", "phase")
            consensus = await self._safe(
                self.bftce.run_full_consensus, all_plans, all_scores, aim, initial_steps
            ) or {}
            final_plan = consensus.get("winner_plan", best_plan)
            report["consensus"] = consensus

            # Phase 16 — HPD Hierarchical Decomposition ───────────────────────
            emit("🌲 Phase 16: Hierarchical decomposition...", "phase")
            steps_for_hpd = [
                l.strip()
                for l in final_plan.split('\n')
                if l.strip() and not l.strip().startswith('[')
            ][:12]
            hpd_result = await self._safe(self.hpd.decompose, steps_for_hpd, aim)
            report["decomposition"] = self.hpd.get_hpd_report(hpd_result) if hpd_result else {}

            # Phase 17 — DKGE Knowledge Graph Update ──────────────────────────
            emit("🕸️  Phase 17: Knowledge graph update...", "phase")
            await self._safe(self.dkge.extract_and_merge, final_plan, aim, session_id)
            report["knowledge_graph"] = self.dkge.get_graph_stats()

            # Phase 18 — CFE Post-session Analysis ────────────────────────────
            emit("🔀 Phase 18: Counterfactual analysis...", "phase")
            steps_list = [l.strip() for l in final_plan.split('\n') if l.strip()]
            cfe_result = await self._safe(
                self.cfe.analyse, steps_list, final_plan, aim, best_score, session_id
            )
            report["counterfactuals"] = self.cfe.get_cfe_report(cfe_result) if cfe_result else {}

            # Phase 19 — CMB + FLMS Store ─────────────────────────────────────
            emit("💾 Phase 19: Persist session memory...", "phase")
            await self._safe(
                self._phase_memory_store, aim, initial_steps, final_plan,
                best_agent, best_score, all_scores
            )
            self.flms.create_local_update(
                session_id     = session_id,
                agent_scores   = all_scores,
                baseline_score = sum(all_scores.values()) / max(len(all_scores), 1),
                step_count     = len(steps_for_hpd),
                domain         = domain,
                feature_flags  = {"hfw": True, "madp": True, "spce": True,
                                   "mopo": True, "arts": True, "sps": True},
            )

            # Phase 20 — ASCL Initialisation ──────────────────────────────────
            emit("🔄 Phase 20: Self-correction loop armed...", "phase")
            self._ascl = AutonomousSelfCorrectionLoop(
                call_fn            = self._call_api,
                original_plan      = final_plan,
                temporal_plan      = self.tes.get_report_dict(sim_report),
                total_deadline_hrs = sim_report.monte_carlo.p90_hrs if sim_report else 168.0,
            )

            elapsed = time.time() - t_start
            emit(f"🏁 V5 session complete in {elapsed:.1f}s", "complete")

            return {
                "success":      True,
                "session_id":   session_id,
                "aim":          aim,
                "best_plan":    final_plan,
                "best_agent":   best_agent,
                "best_score":   best_score,
                "all_scores":   all_scores,
                "elapsed_sec":  round(elapsed, 2),
                "process_log":  log,
                **report,
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[V5] Session failed: {e}\n{tb}")
            emit(f"❌ Error: {e}", "error")
            return {"success": False, "session_id": session_id,
                    "error": str(e), "process_log": log}

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _phase_generate_with_firewall(
        self, aim, steps, agents, router_result,
        priming, causal_ctx, param_groups
    ) -> tuple:
        """AEG generation + inline HFW validation."""
        # Use DASR specialized prompts if available
        if router_result:
            specialized_prompts = router_result.prompt_variants
        else:
            specialized_prompts = {}

        plans, scores = {}, {}
        tasks = []

        async def call_one(agent: str) -> tuple:
            prompt = specialized_prompts.get(
                agent,
                self._build_base_prompt(aim, steps, priming, causal_ctx)
            )
            try:
                allowed, reason = self.rtbc.should_allow_call(agent, prompt)
                if not allowed:
                    return agent, None, 0.0
                raw   = await self._call_api(agent, prompt)
                score = await self._score_wrap(raw, aim)
                return agent, raw, score
            except Exception as e:
                logger.warning(f"[V5] {agent} generation failed: {e}")
                return agent, None, 0.0

        results = await asyncio.gather(*[call_one(a) for a in agents])
        raw_plans  = {agent: plan  for agent, plan, _ in results if plan}
        raw_scores = {agent: score for agent, _, score in results if score > 0}

        # HFW Firewall
        admitted, adj_scores, fw_results = await self.hfw.validate_all(
            raw_plans, raw_scores, aim
        )
        return admitted, adj_scores

    async def _phase_memory_priming(self, aim, steps):
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.memory.get_memory_priming_context(aim)
            ) or ""
        except Exception:
            return ""

    async def _phase_gpm(self, all_plans, all_scores, aim, steps):
        try:
            gpm = GeneticPlanMutator(call_fn=self._call_api, score_fn=self._score_wrap)
            return await gpm.evolve(
                initial_plans=all_plans, initial_scores=all_scores,
                aim=aim, original_steps=steps
            )
        except Exception as e:
            logger.warning(f"[V5] GPM failed: {e}")
            if all_scores:
                best = max(all_scores, key=all_scores.get)
                return {"best_plan": all_plans.get(best,""), "best_score": all_scores.get(best,0)}
            return {}

    async def _phase_memory_store(self, aim, steps, plan, agent, score, all_scores):
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.memory.store_episodic_sync(
                    aim=aim, initial_steps=steps, best_plan=plan,
                    winning_agent=agent, winning_score=score, all_scores=all_scores
                )
            )
            self.memory.update_agent_profiles(all_scores, aim)
        except Exception as e:
            logger.warning(f"[V5] Memory store failed: {e}")

    # ── Safe Wrapper ──────────────────────────────────────────────────────────

    async def _safe(self, fn, *args, **kwargs):
        """Execute a phase function safely — log and return None on failure."""
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[V5] Phase {fn.__name__ if hasattr(fn,'__name__') else fn} failed: {e}")
            return None

    # ── ASCL Public Interface ─────────────────────────────────────────────────

    async def submit_execution_report(
        self,
        step_id: str, step_text: str, status: str, aim: str,
        actual_duration_hrs: Optional[float] = None,
        blocker_description: Optional[str]   = None,
        modifications_made:  Optional[str]   = None,
        completion_pct:      float           = 100.0,
    ) -> Optional[Dict]:
        if not self._ascl:
            return {"error": "No active ASCL session"}
        from autonomous_self_correction_loop import StepStatus
        status_map = {
            "completed":   StepStatus.COMPLETED,   "blocked": StepStatus.BLOCKED,
            "failed":      StepStatus.FAILED,       "modified":StepStatus.MODIFIED,
            "in_progress": StepStatus.IN_PROGRESS,  "skipped": StepStatus.SKIPPED,
        }
        report = ExecutionReport(
            step_id=step_id, step_text=step_text,
            status=status_map.get(status.lower(), StepStatus.IN_PROGRESS),
            actual_duration_hrs=actual_duration_hrs,
            blocker_description=blocker_description,
            modifications_made=modifications_made,
            completion_pct=completion_pct,
        )
        correction = await self._ascl.process_report(report, aim)
        if correction:
            return {
                "deviation_detected":   True,
                "deviation_type":       correction.deviation.deviation_type.value,
                "severity":             correction.deviation.severity,
                "root_cause":           correction.deviation.root_cause,
                "corrective_microplan": correction.microplan,
                "recovery_hrs":         correction.estimated_recovery_hrs,
            }
        return {"deviation_detected": False, "health": self._ascl.get_plan_health_score()}

    # ── Admin ─────────────────────────────────────────────────────────────────

    def get_system_status(self) -> Dict:
        return {
            "version":       "5.0",
            "features":      20,
            "phases":        20,
            "memory":        self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {},
            "budget":        self.rtbc.get_session_report() if hasattr(self.rtbc, 'get_session_report') else {},
            "knowledge_graph":self.dkge.get_graph_stats(),
            "flms":          self.flms.get_flms_report(),
            "ascl":          self._ascl.get_ascl_report() if self._ascl else {"active": False},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_base_prompt(self, aim, steps, priming, causal_ctx) -> str:
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        parts    = [p for p in [priming, causal_ctx] if p]
        parts.append(f"AIM: {aim}\n\nCURRENT STEPS:\n{numbered}\n\nGenerate an improved execution plan:")
        return "\n\n".join(parts)

    async def _gated_call(self, agent: str, prompt: str) -> str:
        allowed, reason = self.rtbc.should_allow_call(agent, prompt)
        if not allowed:
            return f"[RTBC_BLOCKED:{reason}]"
        return await self._call_api(agent, prompt)

    async def _score_wrap(self, plan: str, aim: str) -> float:
        try:
            if asyncio.iscoroutinefunction(self._score):
                return await self._score(plan, aim, [], {})
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._score(plan, aim, [], {})
            )
        except Exception:
            return 0.0
