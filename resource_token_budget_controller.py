# ═══════════════════════════════════════════════════════════════════════════════
# resource_token_budget_controller.py — Feature 3: RTBC
# Per-session cost oracle with circuit breakers and ROI-based pruning
# ═══════════════════════════════════════════════════════════════════════════════

import time
import logging
import threading
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from system_config import (
    RTBC_SESSION_BUDGET_USD, RTBC_AGENT_BUDGET_USD,
    RTBC_MIN_ROI_THRESHOLD, RTBC_CIRCUIT_TRIP_USD,
    RTBC_CIRCUIT_COOLDOWN_SEC, RTBC_EXPECTED_OUTPUT_TOKENS,
    RTBC_REALLOC_TRIGGER, RTBC_MODEL_COSTS
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"     # Normal operation — calls allowed
    OPEN      = "open"       # Tripped — blocking all calls
    HALF_OPEN = "half_open"  # Probing — allow one call to test recovery


class BudgetClass(Enum):
    ECONOMY  = "economy"
    STANDARD = "standard"
    PREMIUM  = "premium"
    UNCAPPED = "uncapped"


@dataclass
class TokenLedgerEntry:
    """Immutable record of one completed API call."""
    call_id:        str
    agent_name:     str
    group_index:    int
    prompt_tokens:  int
    output_tokens:  int
    estimated_cost: float     # USD
    score_before:   float
    score_after:    float
    score_delta:    float     # score_after - score_before
    roi:            float     # score_delta / estimated_cost  (higher = better)
    timestamp:      float
    circuit_state:  CircuitState
    allowed:        bool
    reject_reason:  Optional[str] = None


@dataclass
class BudgetReallocResult:
    """Result of adaptive budget reallocation."""
    agents_boosted:  List[str]      # received extra budget
    agents_cut:      List[str]      # reduced to zero
    new_budgets:     Dict[str, float]
    rationale:       str


class ResourceAwareTokenBudgetController:
    """
    Wraps every call_ai_api() invocation with:

    1. Cost estimation before the call (token counting)
    2. ROI-based allow/reject gate
    3. Circuit breaker per agent (CLOSED → OPEN → HALF_OPEN → CLOSED)
    4. Adaptive budget reallocation at 50% session spend
    5. Full ledger for post-session reporting

    Usage:
        rtbc = ResourceAwareTokenBudgetController()
        allowed, reason = rtbc.should_allow_call(agent, prompt)
        if allowed:
            result = await call_ai_api(...)
            rtbc.record_call_result(agent, group, tokens_in, tokens_out, score_before, score_after)
    """

    def __init__(
        self,
        session_budget_usd:    float = RTBC_SESSION_BUDGET_USD,
        agent_budget_usd:      float = RTBC_AGENT_BUDGET_USD,
        min_roi_threshold:     float = RTBC_MIN_ROI_THRESHOLD,
        circuit_trip_usd:      float = RTBC_CIRCUIT_TRIP_USD,
        circuit_cooldown_sec:  int   = RTBC_CIRCUIT_COOLDOWN_SEC,
    ):
        self.session_budget     = session_budget_usd
        self.agent_budget       = agent_budget_usd
        self.min_roi_threshold  = min_roi_threshold
        self.circuit_trip       = circuit_trip_usd
        self.cooldown           = circuit_cooldown_sec

        self.ledger:            List[TokenLedgerEntry]    = []
        self.circuit_states:    Dict[str, CircuitState]   = {}
        self.circuit_trip_time: Dict[str, float]          = {}
        self.agent_spend:       Dict[str, float]          = {}
        self.agent_custom_budget: Dict[str, float]        = {}  # set by realloc
        self._call_counter:     int                       = 0
        self._lock              = threading.Lock()
        self._realloc_done:     bool                      = False

        logger.info(
            f"[RTBC] Initialized. Session budget: ${session_budget_usd:.2f} | "
            f"Agent budget: ${agent_budget_usd:.3f} | Min ROI: {min_roi_threshold}"
        )

    # ── Token / Cost Estimation ───────────────────────────────────────────────

    @classmethod
    def estimate_token_count(cls, text: str, model_family: str = "gpt") -> int:
        """
        Heuristic token estimation.
        GPT family: ~4 chars/token. Other models: ~3.5 chars/token.
        """
        if model_family == "gpt":
            return max(1, len(text) // 4)
        return max(1, len(text) * 2 // 7)

    def estimate_call_cost(
        self,
        agent_name:            str,
        prompt:                str,
        expected_output_tokens: int = RTBC_EXPECTED_OUTPUT_TOKENS,
    ) -> Tuple[float, int]:
        """
        Returns (estimated_usd, prompt_token_count).
        Uses per-provider pricing from RTBC_MODEL_COSTS.
        """
        family = "gpt" if agent_name in ("openai", "openrouter") else "other"
        prompt_tokens = self.estimate_token_count(prompt, family)

        cost_in, cost_out = RTBC_MODEL_COSTS.get(agent_name, (0.001, 0.002))
        estimated_usd = (
            prompt_tokens      / 1000 * cost_in
            + expected_output_tokens / 1000 * cost_out
        )
        return round(estimated_usd, 6), prompt_tokens

    def _session_total_spent(self) -> float:
        return sum(e.estimated_cost for e in self.ledger if e.allowed)

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def _get_circuit_state(self, agent_name: str) -> CircuitState:
        state = self.circuit_states.get(agent_name, CircuitState.CLOSED)

        if state == CircuitState.OPEN:
            # Check if cooldown has elapsed → transition to HALF_OPEN
            trip_t = self.circuit_trip_time.get(agent_name, 0)
            if time.monotonic() - trip_t >= self.cooldown:
                self.circuit_states[agent_name] = CircuitState.HALF_OPEN
                logger.info(f"[RTBC] Circuit HALF_OPEN for {agent_name}")
                return CircuitState.HALF_OPEN

        return state

    def _maybe_trip_circuit(self, agent_name: str) -> None:
        """Trip circuit OPEN if agent spend exceeds circuit_trip threshold."""
        agent_spent = self.agent_spend.get(agent_name, 0)
        if agent_spent >= self.circuit_trip:
            self.circuit_states[agent_name]    = CircuitState.OPEN
            self.circuit_trip_time[agent_name] = time.monotonic()
            logger.warning(
                f"[RTBC] Circuit OPEN for {agent_name} "
                f"(spent ${agent_spent:.4f} >= trip ${self.circuit_trip:.2f})"
            )

    def reset_circuit(self, agent_name: str) -> None:
        """Manually close circuit for an agent (e.g., after successful HALF_OPEN probe)."""
        self.circuit_states[agent_name] = CircuitState.CLOSED
        logger.info(f"[RTBC] Circuit CLOSED for {agent_name}")

    # ── Core Gate ─────────────────────────────────────────────────────────────

    def should_allow_call(
        self,
        agent_name:          str,
        prompt:              str,
        current_agent_score: Optional[float] = None,
        global_mean_score:   Optional[float] = None,
        global_score_std:    Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Gate function. Returns (allowed: bool, reason: str).

        Blocks if ANY of:
        1. Circuit state == OPEN
        2. Session total spend >= session_budget
        3. Agent spend >= agent custom budget (post-realloc) or default
        4. Estimated ROI < min_roi_threshold AND agent score already > mean+0.5σ
        """
        with self._lock:
            circuit = self._get_circuit_state(agent_name)

            # 1. Circuit open
            if circuit == CircuitState.OPEN:
                return False, f"circuit_open:{agent_name}"

            # 2. Session budget exhausted
            session_spent = self._session_total_spent()
            if session_spent >= self.session_budget:
                return False, f"session_budget_exhausted (${session_spent:.4f} >= ${self.session_budget:.2f})"

            # Trigger reallocation at 50% spend
            if not self._realloc_done and session_spent >= self.session_budget * RTBC_REALLOC_TRIGGER:
                self._reallocate_budgets()
                self._realloc_done = True

            # 3. Agent budget
            agent_spent = self.agent_spend.get(agent_name, 0)
            budget_cap  = self.agent_custom_budget.get(agent_name, self.agent_budget)
            if agent_spent >= budget_cap:
                return False, f"agent_budget_cap (${agent_spent:.4f} >= ${budget_cap:.4f})"

            # 4. ROI check (only when we have score context)
            if current_agent_score is not None and global_mean_score is not None:
                est_cost, _ = self.estimate_call_cost(agent_name, prompt)
                if est_cost > 0:
                    # Estimate score delta from recent ledger
                    agent_entries = [e for e in self.ledger if e.agent_name == agent_name and e.allowed]
                    avg_delta = statistics.mean([e.score_delta for e in agent_entries[-5:]]) if agent_entries else 5.0
                    expected_roi = avg_delta / est_cost if est_cost > 0 else self.min_roi_threshold

                    std_factor = global_score_std or 1.0
                    # Already leading — skip if ROI is poor
                    if (current_agent_score > global_mean_score + 0.5 * std_factor
                            and expected_roi < self.min_roi_threshold):
                        return False, f"low_roi_skip (roi={expected_roi:.1f} < {self.min_roi_threshold}, score already leading)"

            # Allowed
            if circuit == CircuitState.HALF_OPEN:
                # Allow this single probe call
                logger.info(f"[RTBC] HALF_OPEN probe allowed for {agent_name}")

            return True, "allowed"

    # ── Record Results ────────────────────────────────────────────────────────

    def record_call_result(
        self,
        agent_name:           str,
        group_index:          int,
        actual_prompt_tokens: int,
        actual_output_tokens: int,
        score_before:         float,
        score_after:          float,
    ) -> TokenLedgerEntry:
        """
        Record a completed call. Update agent spend. Check circuit trip.
        If this was a HALF_OPEN probe that succeeded (score_delta > 0), close circuit.
        """
        with self._lock:
            self._call_counter += 1
            cost_in, cost_out = RTBC_MODEL_COSTS.get(agent_name, (0.001, 0.002))
            actual_cost = (
                actual_prompt_tokens  / 1000 * cost_in
                + actual_output_tokens / 1000 * cost_out
            )
            score_delta = score_after - score_before
            roi = score_delta / actual_cost if actual_cost > 0 else 0.0

            entry = TokenLedgerEntry(
                call_id       = f"call_{self._call_counter:04d}",
                agent_name    = agent_name,
                group_index   = group_index,
                prompt_tokens = actual_prompt_tokens,
                output_tokens = actual_output_tokens,
                estimated_cost= actual_cost,
                score_before  = score_before,
                score_after   = score_after,
                score_delta   = score_delta,
                roi           = roi,
                timestamp     = time.monotonic(),
                circuit_state = self._get_circuit_state(agent_name),
                allowed       = True,
            )
            self.ledger.append(entry)

            # Update agent cumulative spend
            self.agent_spend[agent_name] = self.agent_spend.get(agent_name, 0) + actual_cost

            # HALF_OPEN → CLOSED if probe succeeded
            if self.circuit_states.get(agent_name) == CircuitState.HALF_OPEN:
                if roi >= self.min_roi_threshold:
                    self.reset_circuit(agent_name)
                else:
                    # Failed probe → back to OPEN
                    self.circuit_states[agent_name]    = CircuitState.OPEN
                    self.circuit_trip_time[agent_name] = time.monotonic()

            self._maybe_trip_circuit(agent_name)

            logger.debug(
                f"[RTBC] Call recorded: {agent_name}:g{group_index} "
                f"cost=${actual_cost:.5f} roi={roi:.1f} delta={score_delta:+.1f}"
            )
            return entry

    def record_rejected_call(
        self,
        agent_name:  str,
        group_index: int,
        reason:      str,
    ) -> None:
        """Log a rejected call for audit trail."""
        with self._lock:
            self._call_counter += 1
            entry = TokenLedgerEntry(
                call_id       = f"call_{self._call_counter:04d}",
                agent_name    = agent_name,
                group_index   = group_index,
                prompt_tokens = 0,
                output_tokens = 0,
                estimated_cost= 0.0,
                score_before  = 0.0,
                score_after   = 0.0,
                score_delta   = 0.0,
                roi           = 0.0,
                timestamp     = time.monotonic(),
                circuit_state = self._get_circuit_state(agent_name),
                allowed       = False,
                reject_reason = reason,
            )
            self.ledger.append(entry)

    # ── Budget Reallocation ───────────────────────────────────────────────────

    def _reallocate_budgets(self) -> BudgetReallocResult:
        """
        Called at 50% spend threshold.
        Ranks agents by ROI, boosts top-3, zeroes bottom-3.
        """
        agent_rois: Dict[str, List[float]] = {}
        for entry in self.ledger:
            if entry.allowed and entry.roi != 0:
                agent_rois.setdefault(entry.agent_name, []).append(entry.roi)

        if not agent_rois:
            return BudgetReallocResult([], [], {}, "No ROI data available")

        avg_roi = {a: statistics.mean(rois) for a, rois in agent_rois.items()}
        ranked  = sorted(avg_roi, key=avg_roi.get, reverse=True)

        remaining = self.session_budget - self._session_total_spent()
        boosted = ranked[:3]
        cut     = ranked[-3:] if len(ranked) >= 6 else []

        # Distribute 80% of remaining budget to top-3 agents
        boost_each = (remaining * 0.80) / len(boosted) if boosted else self.agent_budget
        new_budgets = {}
        for a in boosted:
            current_spent = self.agent_spend.get(a, 0)
            new_budgets[a] = current_spent + boost_each
        for a in cut:
            new_budgets[a] = self.agent_spend.get(a, 0)  # cap at what's already spent → no more calls

        self.agent_custom_budget.update(new_budgets)

        result = BudgetReallocResult(
            agents_boosted = boosted,
            agents_cut     = cut,
            new_budgets    = new_budgets,
            rationale      = f"Realloc at 50% spend. Top ROI: {boosted}. Zero-budget: {cut}."
        )
        logger.info(f"[RTBC] Budget realloc: boosted={boosted}, cut={cut}")
        return result

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_session_report(self) -> Dict:
        """Comprehensive cost/ROI report for Excel output and frontend display."""
        with self._lock:
            allowed  = [e for e in self.ledger if e.allowed]
            rejected = [e for e in self.ledger if not e.allowed]

            total_cost    = sum(e.estimated_cost for e in allowed)
            total_tokens  = sum(e.prompt_tokens + e.output_tokens for e in allowed)

            by_agent: Dict[str, Dict] = {}
            for e in allowed:
                if e.agent_name not in by_agent:
                    by_agent[e.agent_name] = {
                        "calls": 0, "total_cost": 0.0, "total_tokens": 0,
                        "rois": [], "score_deltas": []
                    }
                a = by_agent[e.agent_name]
                a["calls"]        += 1
                a["total_cost"]   += e.estimated_cost
                a["total_tokens"] += e.prompt_tokens + e.output_tokens
                a["rois"].append(e.roi)
                a["score_deltas"].append(e.score_delta)

            agent_summary = {}
            for agent, a in by_agent.items():
                agent_summary[agent] = {
                    "calls":       a["calls"],
                    "total_cost":  round(a["total_cost"], 5),
                    "avg_roi":     round(statistics.mean(a["rois"]), 2) if a["rois"] else 0,
                    "avg_score_delta": round(statistics.mean(a["score_deltas"]), 2) if a["score_deltas"] else 0,
                    "circuit_state": self.circuit_states.get(agent, CircuitState.CLOSED).value,
                }

            roi_ranking = sorted(
                agent_summary.items(), key=lambda x: x[1]["avg_roi"], reverse=True
            )

            circuit_events = [
                {
                    "agent":  a,
                    "state":  s.value,
                    "spend":  round(self.agent_spend.get(a, 0), 5)
                }
                for a, s in self.circuit_states.items()
                if s != CircuitState.CLOSED
            ]

            return {
                "session_budget_usd":    self.session_budget,
                "total_spent_usd":       round(total_cost, 5),
                "remaining_usd":         round(self.session_budget - total_cost, 5),
                "budget_utilisation_pct":round(total_cost / self.session_budget * 100, 1),
                "total_tokens_used":     total_tokens,
                "total_calls_allowed":   len(allowed),
                "total_calls_rejected":  len(rejected),
                "by_agent":              agent_summary,
                "roi_ranking":           [{"agent": a, **s} for a, s in roi_ranking],
                "circuit_events":        circuit_events,
                "realloc_performed":     self._realloc_done,
                "reject_reasons":        [e.reject_reason for e in rejected],
                "projected_final_cost":  round(
                    total_cost / max(len(allowed), 1) * self._call_counter, 5
                ),
            }

    def print_summary(self) -> None:
        """Human-readable console summary."""
        r = self.get_session_report()
        print(f"\n{'─'*55}")
        print(f"  💰 RTBC SESSION REPORT")
        print(f"{'─'*55}")
        print(f"  Spent:   ${r['total_spent_usd']:.4f} / ${r['session_budget_usd']:.2f}  ({r['budget_utilisation_pct']}%)")
        print(f"  Tokens:  {r['total_tokens_used']:,}")
        print(f"  Calls:   {r['total_calls_allowed']} allowed, {r['total_calls_rejected']} rejected")
        print(f"  Realloc: {'✓' if r['realloc_performed'] else '✗'}")
        if r["circuit_events"]:
            print(f"  Circuits: {[c['agent'] + ':' + c['state'] for c in r['circuit_events']]}")
        print(f"  Top ROI: {r['roi_ranking'][0]['agent']} (roi={r['roi_ranking'][0]['avg_roi']:.1f})" if r['roi_ranking'] else "")
        print(f"{'─'*55}\n")
