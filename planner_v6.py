# ═══════════════════════════════════════════════════════════════════════════════
# planner_v6.py — EnhancedMultiAIPlannerV6
# All 25 features. 25-phase pipeline.
#
# V3  F1–F5  : AEG, CMB, RTBC, BFTCE, GPM
# V4  F6–F10 : CRE, MADP, TES, SPCE, ASCL
# V5  F11–20 : DKGE, MOPO, ARTS, HPD, FLMS, CFE, HFW, DASR, PEM, SPS
# V6  F21–25 : APEE, CPCD, ESS, PGT, CCE
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable

# ── V3 ────────────────────────────────────────────────────────────────────────
from cognitive_memory_bank                  import CognitiveMemoryBank
from resource_token_budget_controller       import ResourceAwareTokenBudgetController
from graph_node                             import AdaptiveExecutionGraph
from bft_consensus_engine                   import ByzantineFaultTolerantConsensusEngine
from genetic_plan_mutator                   import GeneticPlanMutator

# ── V4 ────────────────────────────────────────────────────────────────────────
from causal_reasoning_engine                import CausalReasoningEngine
from multi_agent_debate_protocol            import MultiAgentDebateProtocol
from temporal_execution_simulator           import TemporalExecutionSimulator
from semantic_plan_compression_engine       import SemanticPlanCompressionEngine
from autonomous_self_correction_loop        import AutonomousSelfCorrectionLoop, ExecutionReport

# ── V5 F11–15 ─────────────────────────────────────────────────────────────────
from dynamic_knowledge_graph_engine         import DynamicKnowledgeGraphEngine
from multi_objective_pareto_optimizer       import MultiObjectiveParetoOptimizer
from adversarial_red_team_simulator         import AdversarialRedTeamSimulator
from hierarchical_plan_decomposer           import HierarchicalPlanDecomposer
from federated_learning_memory_synthesiser  import FederatedLearningMemorySynthesiser

# ── V5 F16–20 ─────────────────────────────────────────────────────────────────
from counterfactual_reasoning_engine        import CounterfactualReasoningEngine
from hallucination_firewall                 import HallucinationFirewall
from dynamic_agent_specialization_router    import DynamicAgentSpecializationRouter
from plan_entropy_monitor                   import PlanEntropyMonitor
from stakeholder_persona_simulator          import StakeholderPersonaSimulator

# ── V6 F21–25 ─────────────────────────────────────────────────────────────────
from adaptive_prompt_evolution_engine       import AdaptivePromptEvolutionEngine
from cross_plan_contradiction_detector      import CrossPlanContradictionDetector
from execution_simulation_sandbox           import ExecutionSimulationSandbox
from plan_genealogy_tracker                 import PlanGenealogyTracker, OriginType
from confidence_calibration_engine         import ConfidenceCalibrationEngine

from system_config import CMB_DB_PATH, TES_DEFAULT_DOMAIN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S"
)


class EnhancedMultiAIPlannerV6:
    """
    25-feature, 25-phase autonomous planning engine.

    Key V6 additions:
      F21 APEE — prompts evolve genetically across sessions
      F22 CPCD — contradictions between agents resolved before consensus
      F23 ESS  — plan simulated 20x before delivery; failure points fixed
      F24 PGT  — every step traceable to its origin agent/feature
      F25 CCE  — agent confidence calibrated against historical accuracy
    """

    def __init__(
        self,
        call_ai_api_fn:           Callable[[str, str], Awaitable[str]],
        create_scoring_prompt_fn: Callable,
        sse_emit_fn:              Optional[Callable] = None,
        db_path:                  str = CMB_DB_PATH,
        domain:                   str = TES_DEFAULT_DOMAIN,
        resource_pool:            Optional[Dict] = None,
        deadline_hrs:             Optional[float] = None,
    ):
        self._call_api  = call_ai_api_fn
        self._score     = create_scoring_prompt_fn
        self._sse       = sse_emit_fn or (lambda m, t="info": None)
        self._domain    = domain

        # ── V3 ──────────────────────────────────────────────────────────────
        self.memory  = CognitiveMemoryBank(db_path=db_path)
        self.rtbc    = ResourceAwareTokenBudgetController()
        self.aeg     = AdaptiveExecutionGraph(
            call_fn=self._gated, score_fn=self._sw)
        self.bftce   = ByzantineFaultTolerantConsensusEngine(
            call_fn=self._call_api)

        # ── V4 ──────────────────────────────────────────────────────────────
        self.cre  = CausalReasoningEngine(call_fn=self._call_api)
        self.madp = MultiAgentDebateProtocol(
            call_fn=self._call_api, score_fn=self._sw)
        self.tes  = TemporalExecutionSimulator(call_fn=self._call_api)
        self.spce = SemanticPlanCompressionEngine(
            call_fn=self._call_api, score_fn=self._sw)

        # ── V5 ──────────────────────────────────────────────────────────────
        self.dkge = DynamicKnowledgeGraphEngine(call_fn=self._call_api)
        self.mopo = MultiObjectiveParetoOptimizer(call_fn=self._call_api)
        self.arts = AdversarialRedTeamSimulator(call_fn=self._call_api)
        self.hpd  = HierarchicalPlanDecomposer(call_fn=self._call_api)
        self.flms = FederatedLearningMemorySynthesiser(call_fn=self._call_api)
        self.cfe  = CounterfactualReasoningEngine(
            call_fn=self._call_api, score_fn=self._sw)
        self.hfw  = HallucinationFirewall(
            call_fn=self._call_api, domain=domain)
        self.dasr = DynamicAgentSpecializationRouter(call_fn=self._call_api)
        self.pem  = PlanEntropyMonitor(call_fn=self._call_api)
        self.sps  = StakeholderPersonaSimulator(call_fn=self._call_api)

        # ── V6 ──────────────────────────────────────────────────────────────
        self.apee = AdaptivePromptEvolutionEngine(call_fn=self._call_api)
        self.cpcd = CrossPlanContradictionDetector(call_fn=self._call_api)
        self.ess  = ExecutionSimulationSandbox(
            call_fn=self._call_api,
            resource_pool=resource_pool,
            deadline_hrs=deadline_hrs,
        )
        self.pgt  = PlanGenealogyTracker()
        self.cce  = ConfidenceCalibrationEngine(call_fn=self._call_api)

        self._ascl: Optional[AutonomousSelfCorrectionLoop] = None
        logger.info("✅ EnhancedMultiAIPlannerV6 — 25 features ready")

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

        t0         = time.time()
        session_id = f"v6_{int(t0)}"
        domain     = domain or self._domain
        param_groups = param_groups or [parameters]

        logger.info(f"{'═'*65}")
        logger.info(f"  V6  {session_id} | {aim[:70]}")
        logger.info(f"{'═'*65}")

        log: List[Dict] = []
        def emit(msg: str, t: str = "info"):
            log.append({"t": t, "msg": msg, "ts": datetime.utcnow().isoformat()})
            self._sse(msg, t)

        self.rtbc.reset_session()
        self.pgt.start_session(session_id)
        R: Dict = {}   # running report dict

        try:
            # ── Phase 0: FLMS ─────────────────────────────────────────────
            emit("🌐 P0  Federated learning synthesis", "phase")
            if self.flms.should_aggregate():
                await self.flms.aggregate_global_model()
            synth  = self.flms.get_synthesis_for_session(aim, domain)
            agents = _reorder_agents(agents, synth.recommended_agents)

            # ── Phase 1: CMB Memory ────────────────────────────────────────
            emit("📚 P1  Memory priming", "phase")
            mem_ctx = await self._safe(
                lambda: asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.memory.get_memory_priming_context(aim)
                )
            ) or ""

            # ── Phase 2: APEE — evolved prompt selection ───────────────────
            emit("🧬 P2  Adaptive prompt selection (APEE)", "phase")
            numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(initial_steps))
            apee_prompt, apee_gene_id = self.apee.render_best_prompt(aim, numbered, mem_ctx)
            R["apee"] = {"gene_id": apee_gene_id, "generation": self.apee._generation}

            # ── Phase 3: DASR — domain routing ────────────────────────────
            emit("🎯 P3  Domain classification + agent routing (DASR)", "phase")
            router = await self._safe(self.dasr.route, agents, aim, initial_steps, mem_ctx)
            R["domain"] = self.dasr.get_router_report(router) if router else {}

            # ── Phase 4: CFE history injection ────────────────────────────
            emit("🔀 P4  Counterfactual injection", "phase")
            import hashlib
            aim_hash = hashlib.md5(aim.encode()).hexdigest()[:12]
            cfe_ctx  = self.cfe.get_historical_injections(aim_hash)
            context  = "\n\n".join(filter(None, [
                mem_ctx,
                "\n".join(synth.strategy_injections[:2]),
                cfe_ctx,
            ]))

            # ── Phase 5: CRE — causal DAG ─────────────────────────────────
            emit("🔗 P5  Causal analysis (CRE)", "phase")
            causal = await self._safe(self.cre.analyse, initial_steps, aim)
            R["causal"] = self.cre.get_report_dict() if causal else {}
            causal_ctx = causal.causal_context_string if causal else ""

            # ── Phase 6: AEG + HFW ─────────────────────────────────────────
            emit("⚡ P6  Parallel generation + firewall (AEG+HFW)", "phase")
            all_plans, all_scores = await self._generate_and_filter(
                aim, initial_steps, agents, router,
                context, causal_ctx, apee_prompt, param_groups
            )
            if not all_plans:
                raise RuntimeError("All plans quarantined by HFW")
            self.pgt.register_agent_plans(all_plans, all_scores)
            emit(f"  ✅ {len(all_plans)} plans admitted", "progress")
            R["firewall"] = {"admitted": len(all_plans), "total": len(agents)}

            # ── Phase 7: CCE — confidence calibration ─────────────────────
            emit("📊 P7  Confidence calibration (CCE)", "phase")
            cce_result = await self._safe(
                self.cce.calibrate, all_plans, all_scores, aim, session_id
            )
            if cce_result:
                all_scores = cce_result.adjusted_scores or all_scores
            R["calibration"] = self.cce.get_cce_report(cce_result) if cce_result else {}

            # ── Phase 8: CPCD — contradiction detection ────────────────────
            emit("⚠️  P8  Contradiction detection + reconciliation (CPCD)", "phase")
            cpcd = await self._safe(
                self.cpcd.detect_and_reconcile, all_plans, all_scores, aim
            )
            if cpcd:
                all_plans = cpcd.reconciled_plans or all_plans
            R["contradictions"] = self.cpcd.get_cpcd_report(cpcd) if cpcd else {}

            # ── Phase 9: PEM — entropy check ──────────────────────────────
            emit("🌐 P9  Entropy monitoring (PEM)", "phase")
            pem = await self._safe(
                self.pem.monitor_and_intervene, all_plans, aim, self._call_api
            )
            R["entropy"] = self.pem.get_pem_report(pem) if pem else {}

            best_agent = max(all_scores, key=all_scores.get)
            best_plan  = all_plans[best_agent]
            best_score = all_scores[best_agent]

            # ── Phase 10: MADP — debate ────────────────────────────────────
            emit("⚔️  P10 Adversarial debate (MADP)", "phase")
            debate = await self._safe(
                self.madp.run_full_debate, agents, best_plan, aim, all_scores
            )
            if debate and debate.accepted:
                best_plan = debate.revised_plan
                self.pgt.record_transformation(
                    best_plan, debate.revised_plan,
                    OriginType.MADP_DEBATE, None, best_score, "debate_synthesis"
                )
            R["debate"] = self.madp.get_debate_report(debate) if debate else {}

            # ── Phase 11: GPM — evolution ─────────────────────────────────
            emit("🧬 P11 Genetic evolution (GPM)", "phase")
            gpm = await self._safe(
                self._run_gpm, all_plans, all_scores, aim, initial_steps
            ) or {}
            if gpm.get("best_score", 0) > best_score:
                best_plan  = gpm["best_plan"]
                best_score = gpm["best_score"]
            R["evolution"] = gpm

            # ── Phase 12: SPCE — compression ──────────────────────────────
            emit("🗜️  P12 Semantic compression (SPCE)", "phase")
            comp = await self._safe(self.spce.compress, best_plan, aim, best_score)
            if comp and comp.accepted:
                best_plan = "\n".join(
                    f"Step {i+1}: {s}"
                    for i, s in enumerate(comp.compressed_steps)
                )
            R["compression"] = self.spce.get_compression_report(comp) if comp else {}

            # ── Phase 13: MOPO — Pareto ────────────────────────────────────
            emit("📐 P13 Multi-objective Pareto (MOPO)", "phase")
            mopo = await self._safe(self.mopo.evolve_frontier, all_plans, aim, 3)
            R["pareto"] = self.mopo.get_mopo_report(mopo) if mopo else {}

            # ── Phase 14: ARTS — red team ─────────────────────────────────
            emit("🎯 P14 Adversarial red team (ARTS)", "phase")
            arts = await self._safe(self.arts.run_full_red_team, best_plan, aim)
            if arts and arts.hardened_plan_additions:
                for new_step in arts.hardened_plan_additions:
                    self.pgt.record_transformation(
                        best_plan, new_step, OriginType.ARTS_MITIGATION,
                        None, best_score, "red_team_hardening"
                    )
                best_plan += "\n" + "\n".join(
                    f"Step: {s}" for s in arts.hardened_plan_additions
                )
            R["red_team"] = self.arts.get_arts_report(arts) if arts else {}

            # ── Phase 15: SPS — stakeholder review ────────────────────────
            emit("👥 P15 Stakeholder review (SPS)", "phase")
            sps = await self._safe(self.sps.simulate, best_plan, aim)
            if sps and sps.hardened_plan:
                for new_step in _diff_new_steps(best_plan, sps.hardened_plan):
                    self.pgt.record_transformation(
                        best_plan, new_step, OriginType.SPS_HARDENING,
                        None, best_score, "stakeholder_objection"
                    )
                best_plan = sps.hardened_plan
            R["stakeholders"] = self.sps.get_sps_report(sps) if sps else {}

            # ── Phase 16: ESS — execution sandbox ─────────────────────────
            emit("🔬 P16 Execution simulation sandbox (ESS)", "phase")
            ess = await self._safe(self.ess.simulate, best_plan, aim)
            if ess and ess.pre_emptive_fixes:
                fix_text = "\n".join(
                    f"Step: {f}" for f in ess.pre_emptive_fixes
                )
                best_plan += f"\n[PRE-EMPTIVE FIXES]\n{fix_text}"
                emit(
                    f"  ✅ ESS: {ess.completion_rate:.0%} completion rate | "
                    f"verdict={ess.sandbox_verdict}", "progress"
                )
            R["sandbox"] = self.ess.get_ess_report(ess) if ess else {}

            # ── Phase 17: TES — timeline ───────────────────────────────────
            emit("📅 P17 Timeline simulation (TES)", "phase")
            prereqs = (
                {nid: n.prerequisites for nid, n in causal.dag.items()}
                if causal else None
            )
            sim = await self._safe(
                self.tes.simulate, initial_steps, aim, domain, prereqs
            )
            R["simulation"] = self.tes.get_report_dict(sim) if sim else {}

            # ── Phase 18: BFTCE — consensus ────────────────────────────────
            emit("🗳️  P18 Byzantine consensus (BFTCE)", "phase")
            consensus = await self._safe(
                self.bftce.run_full_consensus, all_plans, all_scores, aim, initial_steps
            ) or {}
            final_plan = consensus.get("winner_plan", best_plan)
            R["consensus"] = consensus

            # ── Phase 19: HPD — decomposition ─────────────────────────────
            emit("🌲 P19 Hierarchical decomposition (HPD)", "phase")
            hpd_steps = [
                l.strip()
                for l in final_plan.split('\n')
                if l.strip() and not l.strip().startswith('[')
            ][:12]
            hpd = await self._safe(self.hpd.decompose, hpd_steps, aim)
            R["decomposition"] = self.hpd.get_hpd_report(hpd) if hpd else {}

            # ── Phase 20: DKGE — knowledge graph ──────────────────────────
            emit("🕸️  P20 Knowledge graph update (DKGE)", "phase")
            await self._safe(self.dkge.extract_and_merge, final_plan, aim, session_id)
            R["knowledge_graph"] = self.dkge.get_graph_stats()

            # ── Phase 21: PGT — genealogy report ──────────────────────────
            emit("📜 P21 Plan genealogy (PGT)", "phase")
            genealogy = self.pgt.build_genealogy_report(final_plan, session_id, all_scores)
            R["genealogy"] = self.pgt.get_pgt_report(genealogy)

            # ── Phase 22: CFE — counterfactual analysis ────────────────────
            emit("🔀 P22 Counterfactual analysis (CFE)", "phase")
            steps_list = [
                l.strip() for l in final_plan.split('\n') if l.strip()
            ]
            cfe = await self._safe(
                self.cfe.analyse, steps_list, final_plan, aim, best_score, session_id
            )
            R["counterfactuals"] = self.cfe.get_cfe_report(cfe) if cfe else {}

            # ── Phase 23: APEE score recording ────────────────────────────
            emit("🧬 P23 Prompt gene score update (APEE)", "phase")
            self.apee.record_session_score(apee_gene_id, best_score, session_id)
            if self.apee._generation % 5 == 0:
                await self._safe(self.apee.evolve_population)

            # ── Phase 24: CMB + FLMS store ─────────────────────────────────
            emit("💾 P24 Persist session (CMB+FLMS)", "phase")
            await self._safe(
                self._store_session, aim, initial_steps, final_plan,
                best_agent, best_score, all_scores, domain, session_id
            )

            # ── Phase 25: ASCL init ────────────────────────────────────────
            emit("🔄 P25 Self-correction loop armed (ASCL)", "phase")
            self._ascl = AutonomousSelfCorrectionLoop(
                call_fn           = self._call_api,
                original_plan     = final_plan,
                temporal_plan     = self.tes.get_report_dict(sim) if sim else {},
                total_deadline_hrs= sim.monte_carlo.p90_hrs if sim else 168.0,
            )

            elapsed = round(time.time() - t0, 2)
            emit(f"🏁 V6 complete in {elapsed}s", "done")

            return {
                "success":    True,
                "session_id": session_id,
                "version":    "6.0",
                "aim":        aim,
                "best_plan":  final_plan,
                "best_agent": best_agent,
                "best_score": best_score,
                "all_scores": all_scores,
                "elapsed_sec":elapsed,
                "process_log":log,
                **R,
            }

        except Exception as e:
            import traceback
            emit(f"❌ {e}", "error")
            return {
                "success":    False,
                "session_id": session_id,
                "error":      str(e),
                "traceback":  traceback.format_exc(),
                "process_log":log,
            }

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _generate_and_filter(
        self, aim, steps, agents, router,
        context, causal_ctx, apee_prompt, param_groups,
    ) -> tuple:
        """Phase 6: AEG generation → HFW filter."""
        specialized = router.prompt_variants if router else {}

        async def call_one(agent: str) -> tuple:
            prompt = specialized.get(agent) or apee_prompt or \
                     self._base_prompt(aim, steps, context, causal_ctx)
            ok, reason = self.rtbc.should_allow_call(agent, prompt)
            if not ok:
                return agent, None, 0.0
            try:
                raw   = await self._call_api(agent, prompt)
                score = await self._sw(raw, aim)
                return agent, raw, score
            except Exception:
                return agent, None, 0.0

        results    = await asyncio.gather(*[call_one(a) for a in agents])
        raw_plans  = {a: p for a, p, _ in results if p}
        raw_scores = {a: s for a, _, s in results if s > 0}

        admitted, adj_scores, _ = await self.hfw.validate_all(
            raw_plans, raw_scores, aim
        )
        return admitted, adj_scores

    async def _run_gpm(self, all_plans, all_scores, aim, steps):
        gpm = GeneticPlanMutator(call_fn=self._call_api, score_fn=self._sw)
        return await gpm.evolve(
            initial_plans=all_plans, initial_scores=all_scores,
            aim=aim, original_steps=steps
        )

    async def _store_session(
        self, aim, steps, plan, agent, score, all_scores, domain, session_id
    ):
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.memory.store_episodic_sync(
                    aim=aim, initial_steps=steps, best_plan=plan,
                    winning_agent=agent, winning_score=score, all_scores=all_scores
                )
            )
            self.memory.update_agent_profiles(all_scores, aim)
        except Exception as e:
            logger.warning(f"[V6] Memory store: {e}")

        self.flms.create_local_update(
            session_id=session_id, agent_scores=all_scores,
            baseline_score=sum(all_scores.values()) / max(len(all_scores), 1),
            step_count=len(steps), domain=domain,
            feature_flags={"hfw":True,"cpcd":True,"ess":True,"cce":True,"apee":True},
        )

    # ── ASCL public interface ─────────────────────────────────────────────────

    async def submit_execution_report(
        self,
        step_id: str, step_text: str, status: str, aim: str,
        actual_duration_hrs: Optional[float] = None,
        blocker_description: Optional[str]   = None,
        modifications_made:  Optional[str]   = None,
        completion_pct: float = 100.0,
    ) -> Optional[Dict]:
        if not self._ascl:
            return {"error": "No active session"}
        from autonomous_self_correction_loop import StepStatus
        sm = {
            "completed": StepStatus.COMPLETED, "blocked": StepStatus.BLOCKED,
            "failed":    StepStatus.FAILED,    "modified":StepStatus.MODIFIED,
            "in_progress":StepStatus.IN_PROGRESS, "skipped":StepStatus.SKIPPED,
        }
        rep = ExecutionReport(
            step_id=step_id, step_text=step_text,
            status=sm.get(status.lower(), StepStatus.IN_PROGRESS),
            actual_duration_hrs=actual_duration_hrs,
            blocker_description=blocker_description,
            modifications_made=modifications_made,
            completion_pct=completion_pct,
        )
        cor = await self._ascl.process_report(rep, aim)
        if cor:
            return {
                "deviation_detected":   True,
                "deviation_type":       cor.deviation.deviation_type.value,
                "severity":             cor.deviation.severity,
                "root_cause":           cor.deviation.root_cause,
                "corrective_microplan": cor.microplan,
                "recovery_hrs":         cor.estimated_recovery_hrs,
            }
        return {"deviation_detected": False, "health": self._ascl.get_plan_health_score()}

    # ── Admin ─────────────────────────────────────────────────────────────────

    def get_system_status(self) -> Dict:
        return {
            "version":        "6.0",
            "total_features": 25,
            "phases":         25,
            "apee_generation":self.apee._generation,
            "apee_report":    self.apee.get_apee_report(),
            "kg_stats":       self.dkge.get_graph_stats(),
            "flms":           self.flms.get_flms_report(),
            "ascl":           self._ascl.get_ascl_report() if self._ascl else {"active":False},
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _safe(self, fn, *args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            name = getattr(fn, '__name__', str(fn))
            logger.warning(f"[V6] {name} failed: {e}")
            return None

    async def _gated(self, agent: str, prompt: str) -> str:
        ok, reason = self.rtbc.should_allow_call(agent, prompt)
        if not ok:
            return f"[BLOCKED:{reason}]"
        return await self._call_api(agent, prompt)

    async def _sw(self, plan: str, aim: str) -> float:
        try:
            if asyncio.iscoroutinefunction(self._score):
                return await self._score(plan, aim, [], {})
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._score(plan, aim, [], {})
            )
        except Exception:
            return 0.0

    def _base_prompt(self, aim, steps, context, causal_ctx) -> str:
        n = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        parts = [p for p in [context, causal_ctx] if p]
        parts.append(f"AIM: {aim}\n\nSTEPS:\n{n}\n\nGenerate an improved plan:")
        return "\n\n".join(parts)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _reorder_agents(agents: List[str], preferred: List[str]) -> List[str]:
    seen = set()
    result = []
    for a in preferred:
        if a in agents and a not in seen:
            result.append(a)
            seen.add(a)
    for a in agents:
        if a not in seen:
            result.append(a)
    return result


def _diff_new_steps(original: str, revised: str) -> List[str]:
    """Find steps in revised that don't appear in original."""
    import re
    def get_steps(text):
        return set(
            re.sub(r'^\s*(step\s*\d+[:\.\)]\s*|\d+[:\.\)]\s*)', '', l, flags=re.IGNORECASE).strip()
            for l in text.split('\n') if l.strip()
        )
    orig_steps = get_steps(original)
    new_steps  = get_steps(revised)
    return list(new_steps - orig_steps)
