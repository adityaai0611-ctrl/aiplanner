# ═══════════════════════════════════════════════════════════════════════════════
# planner_v4.py — EnhancedMultiAIPlannerV4
# Integrates all 10 features: V3 (AEG, CMB, RTBC, BFTCE, GPM) +
# V4 (CRE, MADP, TES, SPCE, ASCL)
# ═══════════════════════════════════════════════════════════════════════════════
#
# DROP-IN REPLACEMENT for planner_v3.py
#
# In app.py, replace:
#   from planner_v3 import EnhancedMultiAIPlannerV3
#   planner = EnhancedMultiAIPlannerV3(...)
# With:
#   from planner_v4 import EnhancedMultiAIPlannerV4
#   planner = EnhancedMultiAIPlannerV4(...)
#
# Result dict is backwards-compatible with V3, with new keys added:
#   causal_analysis, debate, compression, simulation, ascl_handle
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable

# ── V3 Modules ────────────────────────────────────────────────────────────────
from cognitive_memory_bank          import CognitiveMemoryBank
from resource_token_budget_controller import ResourceAwareTokenBudgetController
from graph_node                     import AdaptiveExecutionGraph
from bft_consensus_engine           import ByzantineFaultTolerantConsensusEngine
from genetic_plan_mutator           import GeneticPlanMutator

# ── V4 Modules ────────────────────────────────────────────────────────────────
from causal_reasoning_engine        import CausalReasoningEngine
from multi_agent_debate_protocol    import MultiAgentDebateProtocol
from temporal_execution_simulator   import TemporalExecutionSimulator
from semantic_plan_compression_engine import SemanticPlanCompressionEngine
from autonomous_self_correction_loop  import AutonomousSelfCorrectionLoop, ExecutionReport

# ── Config ────────────────────────────────────────────────────────────────────
from system_config import (
    CRE_INJECT_INTO_PROMPTS, TES_DEFAULT_DOMAIN,
    SPCE_ANTI_REGRESSION_MAX_DROP, MADP_SCORE_IMPROVEMENT_MIN,
    CMB_DB_PATH, RTBC_SESSION_BUDGET_USD, RTBC_AGENT_BUDGET_USD,
    GPM_MAX_GENERATIONS, BFTCE_QUORUM_FRACTION,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S"
)


class EnhancedMultiAIPlannerV4:
    """
    Full autonomous, resource-aware, self-correcting planning engine.

    10-phase execution pipeline:
      Phase 0 — CMB Priming         (V3)
      Phase 1 — CRE Causal Analysis (V4 NEW)
      Phase 2 — AEG Parallel Gen    (V3)
      Phase 3 — MADP Debate         (V4 NEW)
      Phase 4 — GPM Evolution       (V3)
      Phase 5 — SPCE Compression    (V4 NEW)
      Phase 6 — TES Simulation      (V4 NEW)
      Phase 7 — BFTCE Consensus     (V3)
      Phase 8 — CMB Store           (V3)
      Phase 9 — ASCL Handle Return  (V4 NEW — ongoing after delivery)
    """

    def __init__(
        self,
        call_ai_api_fn:           Callable[[str, str], Awaitable[str]],
        create_scoring_prompt_fn: Callable[[str, str, List[str], Any], float],
        fraud_check_fn:           Optional[Callable] = None,
        feature_detect_fn:        Optional[Callable] = None,
        sse_emit_fn:              Optional[Callable[[str, str], None]] = None,
        db_path:                  str = CMB_DB_PATH,
        domain:                   str = TES_DEFAULT_DOMAIN,
    ):
        """
        Args:
            call_ai_api_fn:           async(agent_name, prompt) → str
            create_scoring_prompt_fn: (plan, aim, steps, params) → float
            fraud_check_fn:           optional fraud detection callable
            feature_detect_fn:        optional feature extraction callable
            sse_emit_fn:              optional SSE streaming hook(message, type)
            db_path:                  SQLite path for CMB memory bank
            domain:                   execution domain hint for TES estimates
        """
        self._call_api   = call_ai_api_fn
        self._score      = create_scoring_prompt_fn
        self._fraud      = fraud_check_fn
        self._features   = feature_detect_fn
        self._sse        = sse_emit_fn or (lambda msg, t="info": None)
        self._domain     = domain

        # ── V3 modules ─────────────────────────────────────────────────────
        self.memory  = CognitiveMemoryBank(db_path=db_path)
        self.rtbc    = ResourceAwareTokenBudgetController()
        self.aeg     = AdaptiveExecutionGraph(
            call_fn  = self._gated_call,
            score_fn = self._score_wrap,
        )
        self.bftce   = ByzantineFaultTolerantConsensusEngine(
            call_fn = self._call_api
        )

        # ── V4 modules ─────────────────────────────────────────────────────
        self.cre  = CausalReasoningEngine(
            call_fn       = self._call_api,
            primary_agent = "gemini",
        )
        self.madp = MultiAgentDebateProtocol(
            call_fn  = self._call_api,
            score_fn = self._score_wrap,
        )
        self.tes  = TemporalExecutionSimulator(
            call_fn       = self._call_api,
            primary_agent = "gemini",
        )
        self.spce = SemanticPlanCompressionEngine(
            call_fn  = self._call_api,
            score_fn = self._score_wrap,
            agent    = "gemini",
        )

        # ASCL is created per-session (holds session state)
        self._ascl: Optional[AutonomousSelfCorrectionLoop] = None

        logger.info("✅ EnhancedMultiAIPlannerV4 initialised (10-phase pipeline)")

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
        domain:        Optional[str] = None,
    ) -> Dict:
        """
        Runs the complete 10-phase V4 planning pipeline.

        Returns a result dict compatible with V3 schema + V4 extensions.
        """
        t_start    = time.time()
        session_id = f"v4_{int(t_start)}"
        domain     = domain or self._domain
        param_groups = param_groups or [parameters]

        logger.info(f"\n{'═'*60}")
        logger.info(f"  V4 SESSION START: {session_id}")
        logger.info(f"  AIM: {aim[:80]}")
        logger.info(f"  Agents: {len(agents)}, Steps: {len(initial_steps)}")
        logger.info(f"{'═'*60}")

        process_log = []
        self._log   = lambda msg, t="info": (
            process_log.append({"type": t, "message": msg, "ts": datetime.utcnow().isoformat()}),
            self._sse(msg, t)
        )

        # Reset per-session state
        self.rtbc.reset_session()

        try:
            # ── Phase 0: CMB Priming ──────────────────────────────────────
            self._log("🧠 Phase 0: Memory priming...", "phase")
            memory_context = await self._phase_memory_priming(aim, initial_steps)

            # ── Phase 1: CRE Causal Analysis ─────────────────────────────
            self._log("🔗 Phase 1: Causal dependency analysis...", "phase")
            causal_report = await self._phase_causal_analysis(initial_steps, aim)

            # ── Phase 2: AEG Parallel Generation ─────────────────────────
            self._log("⚡ Phase 2: Parallel plan generation (AEG)...", "phase")
            aeg_result = await self._phase_aeg_generation(
                aim, initial_steps, parameters, agents,
                param_groups, memory_context, causal_report
            )
            all_plans = aeg_result.get("plans", {})
            all_scores= aeg_result.get("scores", {})

            if not all_plans:
                raise RuntimeError("AEG generation produced no valid plans")

            best_agent = max(all_scores, key=all_scores.get)
            best_plan  = all_plans[best_agent]
            best_score = all_scores[best_agent]
            self._log(f"✅ AEG complete. Best: {best_agent} (score={best_score:.1f})", "progress")

            # ── Phase 3: MADP Debate ──────────────────────────────────────
            self._log("⚔️  Phase 3: Multi-agent debate...", "phase")
            debate_result = await self._phase_debate(agents, best_plan, aim, all_scores)
            if debate_result.accepted:
                best_plan = debate_result.revised_plan
                self._log(f"✅ Debate accepted: +{debate_result.plan_improvement_score:.1f} pts", "progress")
            else:
                self._log("⚠️  Debate rejected (insufficient improvement). Keeping original.", "warning")

            # ── Phase 4: GPM Evolution ────────────────────────────────────
            self._log("🧬 Phase 4: Genetic plan evolution...", "phase")
            gpm_result = await self._phase_gpm_evolution(all_plans, all_scores, aim, initial_steps)
            evolved_plan = gpm_result.get("best_plan", best_plan)
            evolved_score= gpm_result.get("best_score", best_score)
            if evolved_score > best_score:
                best_plan  = evolved_plan
                best_score = evolved_score
                self._log(f"✅ GPM improved score to {best_score:.1f}", "progress")

            # ── Phase 5: SPCE Compression ─────────────────────────────────
            self._log("🗜️  Phase 5: Semantic plan compression...", "phase")
            compression = await self._phase_compression(best_plan, aim, best_score)
            if compression.accepted:
                best_plan = "\n".join(
                    f"Step {i+1}: {s}" for i, s in enumerate(compression.compressed_steps)
                )
                self._log(
                    f"✅ Compressed {compression.original_count} → {compression.compressed_count} steps "
                    f"({compression.compression_ratio:.0%} reduction)", "progress"
                )

            # ── Phase 6: TES Simulation ───────────────────────────────────
            self._log("📅 Phase 6: Temporal execution simulation...", "phase")
            sim_report = await self._phase_simulation(initial_steps, aim, causal_report, domain)
            self._log(
                f"✅ TES: P50={sim_report.monte_carlo.p50_hrs:.1f}h "
                f"P90={sim_report.monte_carlo.p90_hrs:.1f}h "
                f"conflicts={len(sim_report.conflicts)}", "progress"
            )

            # ── Phase 7: BFTCE Consensus ──────────────────────────────────
            self._log("🗳️  Phase 7: Byzantine consensus...", "phase")
            consensus = await self._phase_bftce_consensus(all_plans, all_scores, aim, initial_steps)
            if consensus and consensus.get("winner_plan"):
                # Weight: 60% BFTCE winner, 40% evolved plan
                final_plan = consensus["winner_plan"]
                self._log(f"✅ Consensus winner: {consensus.get('winner_agent', 'unknown')}", "progress")
            else:
                final_plan = best_plan
                self._log("⚠️  Consensus inconclusive. Using evolved plan.", "warning")

            # ── Phase 8: CMB Store ────────────────────────────────────────
            self._log("💾 Phase 8: Storing session to memory...", "phase")
            await self._phase_memory_store(
                aim, initial_steps, final_plan, best_agent, best_score, all_scores
            )

            # ── Phase 9: Initialise ASCL ──────────────────────────────────
            self._log("🔄 Phase 9: Autonomous correction loop initialised", "phase")
            self._ascl = AutonomousSelfCorrectionLoop(
                call_fn             = self._call_api,
                original_plan       = final_plan,
                temporal_plan       = self.tes.get_report_dict(sim_report),
                agent               = "gemini",
                total_deadline_hrs  = sim_report.monte_carlo.p90_hrs,
            )

            elapsed = time.time() - t_start
            self._log(f"🏁 V4 session complete in {elapsed:.1f}s", "complete")

            return self._build_result(
                session_id    = session_id,
                aim           = aim,
                final_plan    = final_plan,
                best_agent    = best_agent,
                best_score    = best_score,
                all_scores    = all_scores,
                all_plans     = all_plans,
                causal_report = causal_report,
                debate_result = debate_result,
                compression   = compression,
                sim_report    = sim_report,
                consensus     = consensus or {},
                gpm_result    = gpm_result,
                elapsed_sec   = elapsed,
                process_log   = process_log,
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[V4] Session failed: {e}\n{tb}")
            self._log(f"❌ Session error: {e}", "error")
            return {
                "success":    False,
                "session_id": session_id,
                "error":      str(e),
                "traceback":  tb,
                "process_log":process_log,
            }

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _phase_memory_priming(self, aim: str, steps: List[str]) -> str:
        """Phase 0: retrieve similar sessions, agent rankings, strategy patterns."""
        try:
            ctx = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.memory.get_memory_priming_context(aim)
            )
            return ctx or ""
        except Exception as e:
            logger.warning(f"[V4] Memory priming failed: {e}")
            return ""

    async def _phase_causal_analysis(
        self, steps: List[str], aim: str
    ):
        """Phase 1: CRE — build causal DAG, critical path, bottlenecks."""
        try:
            return await self.cre.analyse(steps, aim)
        except Exception as e:
            logger.warning(f"[V4] CRE failed: {e}")
            return None

    async def _phase_aeg_generation(
        self,
        aim:           str,
        steps:         List[str],
        parameters:    List[Any],
        agents:        List[str],
        param_groups:  List[Any],
        memory_ctx:    str,
        causal_report,
    ) -> Dict:
        """Phase 2: parallel agent plan generation via AEG, gated by RTBC."""
        # Build enriched system prompt
        causal_ctx = ""
        if causal_report and CRE_INJECT_INTO_PROMPTS:
            causal_ctx = causal_report.causal_context_string

        priming = "\n\n".join(filter(None, [memory_ctx, causal_ctx]))

        plans, scores = {}, {}
        try:
            aeg_output = await self.aeg.execute(
                aim         = aim,
                steps       = steps,
                agents      = agents,
                param_groups= param_groups,
                context     = priming,
            )
            plans  = aeg_output.get("plans", {})
            scores = aeg_output.get("scores", {})
        except Exception as e:
            logger.warning(f"[V4] AEG failed: {e}. Running sequential fallback.")
            # Sequential fallback
            for agent in agents[:4]:
                try:
                    prompt = self._build_agent_prompt(aim, steps, priming)
                    raw    = await self._gated_call(agent, prompt)
                    plans[agent]  = raw
                    scores[agent] = await self._score_wrap(raw, aim)
                except Exception:
                    pass

        return {"plans": plans, "scores": scores}

    async def _phase_debate(
        self,
        agents:      List[str],
        best_plan:   str,
        aim:         str,
        all_scores:  Dict[str, float],
    ):
        """Phase 3: 3-round adversarial debate."""
        try:
            return await self.madp.run_full_debate(
                agents             = agents,
                best_plan          = best_plan,
                aim                = aim,
                credibility_scores = all_scores,
            )
        except Exception as e:
            logger.warning(f"[V4] MADP failed: {e}")
            # Return neutral non-accepted result
            from multi_agent_debate_protocol import DebateResult
            return DebateResult(
                original_plan          = best_plan,
                revised_plan           = best_plan,
                debate_verdict         = "draw",
                plan_improvement_score = 0.0,
                rounds                 = [],
                total_concessions      = 0,
                changes_made           = [],
                proponent_agent        = agents[0] if agents else "unknown",
                skeptic_agent          = agents[1] if len(agents) > 1 else "unknown",
                synthesiser_agent      = agents[2] if len(agents) > 2 else "unknown",
                accepted               = False,
            )

    async def _phase_gpm_evolution(
        self,
        all_plans:  Dict[str, str],
        all_scores: Dict[str, float],
        aim:        str,
        steps:      List[str],
    ) -> Dict:
        """Phase 4: genetic evolution."""
        try:
            gpm = GeneticPlanMutator(
                call_fn  = self._call_api,
                score_fn = self._score_wrap,
            )
            result = await gpm.evolve(
                initial_plans = all_plans,
                initial_scores= all_scores,
                aim           = aim,
                original_steps= steps,
            )
            return result
        except Exception as e:
            logger.warning(f"[V4] GPM failed: {e}")
            if all_scores:
                best = max(all_scores, key=all_scores.get)
                return {"best_plan": all_plans.get(best, ""), "best_score": all_scores.get(best, 0.0)}
            return {}

    async def _phase_compression(self, plan_text: str, aim: str, score_before: float):
        """Phase 5: SPCE — deduplicate and compress."""
        try:
            return await self.spce.compress(plan_text, aim, score_before)
        except Exception as e:
            logger.warning(f"[V4] SPCE failed: {e}")
            from semantic_plan_compression_engine import _no_compression_result, _parse_steps
            return _no_compression_result(_parse_steps(plan_text), score_before)

    async def _phase_simulation(self, steps: List[str], aim: str, causal_report, domain: str):
        """Phase 6: TES — Monte Carlo timeline simulation."""
        try:
            prereqs = None
            if causal_report:
                prereqs = {
                    nid: node.prerequisites
                    for nid, node in causal_report.dag.items()
                }
            return await self.tes.simulate(steps, aim, domain, prereqs)
        except Exception as e:
            logger.warning(f"[V4] TES failed: {e}")
            # Minimal fallback simulation
            return await self.tes.simulate(steps, aim, domain)

    async def _phase_bftce_consensus(
        self,
        all_plans:  Dict[str, str],
        all_scores: Dict[str, float],
        aim:        str,
        steps:      List[str],
    ) -> Dict:
        """Phase 7: BFTCE peer review and consensus."""
        try:
            result = await self.bftce.run_full_consensus(
                agent_plans = all_plans,
                agent_scores= all_scores,
                aim         = aim,
                steps       = steps,
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"[V4] BFTCE failed: {e}")
            return {}

    async def _phase_memory_store(
        self,
        aim: str, steps: List[str], plan: str,
        agent: str, score: float, all_scores: Dict
    ):
        """Phase 8: persist session to CMB."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.memory.store_episodic_sync(
                    aim=aim, initial_steps=steps, best_plan=plan,
                    winning_agent=agent, winning_score=score, all_scores=all_scores
                )
            )
            self.memory.update_agent_profiles(all_scores, aim)
        except Exception as e:
            logger.warning(f"[V4] Memory store failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ASCL PUBLIC INTERFACE  (called by app.py after session delivery)
    # ═══════════════════════════════════════════════════════════════════════════

    async def submit_execution_report(
        self,
        step_id:             str,
        step_text:           str,
        status:              str,
        aim:                 str,
        actual_duration_hrs: Optional[float] = None,
        blocker_description: Optional[str]   = None,
        modifications_made:  Optional[str]   = None,
        completion_pct:      float           = 100.0,
    ) -> Optional[Dict]:
        """
        Called by the frontend/executor to report step execution status.
        Returns correction dict if a deviation is detected, else None.

        Example (in app.py):
            result = await planner.submit_execution_report(
                step_id="step_3", step_text="Build MVP", status="blocked",
                aim=original_aim, blocker_description="AWS quota exceeded"
            )
        """
        if not self._ascl:
            return {"error": "No active ASCL session. Run run_full_planning_session first."}

        from autonomous_self_correction_loop import StepStatus
        status_map = {
            "completed":   StepStatus.COMPLETED,
            "blocked":     StepStatus.BLOCKED,
            "failed":      StepStatus.FAILED,
            "modified":    StepStatus.MODIFIED,
            "in_progress": StepStatus.IN_PROGRESS,
            "skipped":     StepStatus.SKIPPED,
        }
        report = ExecutionReport(
            step_id             = step_id,
            step_text           = step_text,
            status              = status_map.get(status.lower(), StepStatus.IN_PROGRESS),
            actual_duration_hrs = actual_duration_hrs,
            blocker_description = blocker_description,
            modifications_made  = modifications_made,
            completion_pct      = completion_pct,
        )
        correction = await self._ascl.process_report(report, aim)
        if correction:
            return {
                "deviation_detected":    True,
                "deviation_type":        correction.deviation.deviation_type.value,
                "severity":              correction.deviation.severity,
                "root_cause":            correction.deviation.root_cause,
                "corrective_microplan":  correction.microplan,
                "insert_before_step":    correction.insert_before_step,
                "recovery_hrs":          correction.estimated_recovery_hrs,
                "confidence":            correction.correction_confidence,
                "if_correction_fails":   correction.if_correction_fails,
            }
        return {"deviation_detected": False, "health_score": self._ascl.get_plan_health_score()}

    def record_correction_outcome(self, deviation_id: str, succeeded: bool) -> None:
        """Report back whether a correction micro-plan resolved the issue."""
        if self._ascl:
            self._ascl.record_correction_outcome(deviation_id, succeeded)

    # ═══════════════════════════════════════════════════════════════════════════
    # BACKWARDS COMPAT + ADMIN ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_memory_stats(self) -> Dict:
        try:
            return self.memory.get_stats()
        except Exception:
            return {}

    def get_budget_status(self) -> Dict:
        try:
            return self.rtbc.get_session_report()
        except Exception:
            return {}

    def get_ascl_status(self) -> Dict:
        if self._ascl:
            return self._ascl.get_ascl_report()
        return {"active": False}

    def export_all_data(self) -> Dict:
        return {
            "memory_snapshot": self.memory.export_memory_snapshot(),
            "budget_report":   self.get_budget_status(),
            "ascl_history":    self._ascl.get_correction_history() if self._ascl else [],
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULT BUILDER
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_result(
        self,
        session_id, aim, final_plan, best_agent, best_score,
        all_scores, all_plans, causal_report, debate_result,
        compression, sim_report, consensus, gpm_result,
        elapsed_sec, process_log,
    ) -> Dict:
        result = {
            # ── V3-compatible core keys ──────────────────────────────────
            "success":       True,
            "session_id":    session_id,
            "aim":           aim,
            "best_plan":     final_plan,
            "best_agent":    best_agent,
            "best_score":    best_score,
            "all_scores":    all_scores,
            "elapsed_sec":   round(elapsed_sec, 2),
            "process_log":   process_log,

            # ── V4 extension keys ────────────────────────────────────────
            "causal_analysis": self.cre.get_report_dict() if causal_report else {},
            "debate": self.madp.get_debate_report(debate_result),
            "evolution": gpm_result,
            "compression": self.spce.get_compression_report(compression) if compression else {},
            "simulation":  self.tes.get_report_dict(sim_report) if sim_report else {},
            "consensus":   consensus,
            "token_budget":self.get_budget_status(),
        }

        # Agent plan snippets for history UI
        result["agent_plans"] = {
            agent: plan[:500] + "..."
            if len(plan) > 500 else plan
            for agent, plan in all_plans.items()
        }
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _gated_call(self, agent: str, prompt: str) -> str:
        """RTBC-gated API call."""
        allowed, reason = self.rtbc.should_allow_call(agent, prompt)
        if not allowed:
            logger.warning(f"[RTBC] Blocked {agent}: {reason}")
            return f"[RTBC_BLOCKED: {reason}]"
        try:
            response = await self._call_api(agent, prompt)
            self.rtbc.record_call_result(
                agent       = agent,
                prompt      = prompt,
                response    = response,
                score_before= 0.0,
                score_after = 0.0,
            )
            return response
        except Exception as e:
            logger.warning(f"[V4] API call failed for {agent}: {e}")
            return ""

    async def _score_wrap(self, plan_text: str, aim: str) -> float:
        """Wrap scoring function for use as async callable."""
        try:
            if asyncio.iscoroutinefunction(self._score):
                return await self._score(plan_text, aim, [], {})
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._score(plan_text, aim, [], {})
            )
        except Exception:
            return 0.0

    def _build_agent_prompt(self, aim: str, steps: List[str], context: str) -> str:
        """Simple prompt builder for AEG sequential fallback."""
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        parts    = []
        if context:
            parts.append(context)
        parts.append(
            f"AIM: {aim}\n\nCURRENT STEPS:\n{numbered}\n\n"
            "Generate an improved, detailed execution plan for this aim."
        )
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# ASCL ENDPOINT — new route for app.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# Add this to app.py:
#
# @app.route('/api/execution-report', methods=['POST'])
# def execution_report():
#     """Receive step execution status updates and get corrections if needed."""
#     data = request.json
#     result = asyncio.run(planner.submit_execution_report(
#         step_id             = data['step_id'],
#         step_text           = data['step_text'],
#         status              = data['status'],
#         aim                 = data['aim'],
#         actual_duration_hrs = data.get('actual_duration_hrs'),
#         blocker_description = data.get('blocker_description'),
#         modifications_made  = data.get('modifications_made'),
#         completion_pct      = data.get('completion_pct', 100.0),
#     ))
#     return jsonify(result), 200
#
# @app.route('/api/plan-health', methods=['GET'])
# def plan_health():
#     return jsonify(planner.get_ascl_status()), 200
#
# ═══════════════════════════════════════════════════════════════════════════════
