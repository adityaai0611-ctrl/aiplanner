# ═══════════════════════════════════════════════════════════════════════════════
# system_config.py — EnhancedMultiAIPlanner v3.0 Global Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Feature 1: Adaptive Execution Graph ────────────────────────────────────
AEG_CONCURRENCY_LIMIT       = 4       # max parallel asyncio tasks
AEG_PRUNE_SIGMA_THRESHOLD   = 1.5    # std devs below mean → prune node
AEG_CORRELATION_THRESHOLD   = 0.80   # Pearson r → add dependency edge
AEG_MAX_RETRIES             = 2       # per-node retry on transient failure
AEG_RETRY_BACKOFF_SEC       = 2.0    # seconds between retries

# ─── Feature 2: Cognitive Memory Bank ───────────────────────────────────────
CMB_DB_PATH                 = "planner_memory.db"
CMB_EMBEDDING_DIMS          = 768     # Gemini text-embedding-004
CMB_RETRIEVAL_TOP_K         = 3       # similar sessions to inject
CMB_MIN_SIMILARITY          = 0.75   # cosine sim floor
CMB_EMA_ALPHA               = 0.20   # exponential moving average weight
CMB_DISTILLATION_INTERVAL   = 10     # sessions between semantic distillation
CMB_CACHE_MAX_SIZE          = 256    # max embedding cache entries

# ─── Feature 3: Resource-Aware Token Budget Controller ───────────────────────
RTBC_SESSION_BUDGET_USD     = 0.50   # per-session hard cap
RTBC_AGENT_BUDGET_USD       = 0.05   # per-agent soft cap
RTBC_MIN_ROI_THRESHOLD      = 50.0  # score_delta / cost floor
RTBC_CIRCUIT_TRIP_USD       = 0.30  # trip circuit at this cumulative spend
RTBC_CIRCUIT_COOLDOWN_SEC   = 60    # seconds before HALF_OPEN state
RTBC_REALLOC_TRIGGER        = 0.50  # reallocate budget when 50% consumed
RTBC_EXPECTED_OUTPUT_TOKENS = 800   # default expected response length

# Token costs per 1K tokens (input, output) in USD
RTBC_MODEL_COSTS = {
    "gemini":     (0.000075, 0.000300),
    "openai":     (0.000500, 0.001500),
    "together":   (0.000140, 0.000280),
    "groq":       (0.000590, 0.000790),
    "fireworks":  (0.000200, 0.000200),
    "cohere":     (0.001000, 0.002000),
    "deepseek":   (0.000140, 0.000280),
    "openrouter": (0.000600, 0.002000),
    "writer":     (0.000800, 0.001600),
    "huggingface":(0.000100, 0.000100),
    "replicate":  (0.000350, 0.000700),
    "pawn":       (0.000100, 0.000100),
}

# ─── Feature 4: Byzantine Fault-Tolerant Consensus Engine ────────────────────
BFTCE_QUORUM_FRACTION       = 2/3    # 8/12 agents must agree
BFTCE_ADVERSARIAL_THRESHOLD = 15.0  # score delta → adversarial flag
BFTCE_CREDIBILITY_PENALTY   = 0.50  # multiplier applied on adversarial flag
BFTCE_PEER_REVIEWS_PER_AGENT = 3    # blind reviews per agent
BFTCE_DIMENSION_WEIGHTS = {         # 6-dimension scoring weights
    "completeness":   0.25,
    "actionability":  0.20,
    "coherence":      0.20,
    "novelty":        0.15,
    "safety":         0.10,
    "efficiency":     0.10,
}

# ─── Feature 5: Genetic Plan Mutator ─────────────────────────────────────────
GPM_CROSSOVER_PROB          = 0.70   # probability of crossover vs. cloning
GPM_MUTATION_PROB_PER_STEP  = 0.15  # per-step mutation probability
GPM_ELITISM_K               = 2      # elite individuals guaranteed to survive
GPM_MAX_GENERATIONS         = 5      # hard generation limit
GPM_PLATEAU_PATIENCE        = 3      # gens without improvement → stop
GPM_IMMIGRANT_RATE          = 0.10  # fraction of population replaced randomly
GPM_TOURNAMENT_SIZE         = 3      # competitors in tournament selection
GPM_MIN_POPULATION          = 4      # minimum population size

# ─── Feature 6: Causal Reasoning Engine ──────────────────────────────────────
CRE_BOTTLENECK_FAN_THRESHOLD   = 3
CRE_CASCADE_RISK_THRESHOLD     = 0.70
CRE_INJECT_INTO_PROMPTS        = True
CRE_MAX_MITIGATION_STEPS       = 2

# ─── Feature 7: Multi-Agent Debate Protocol ──────────────────────────────────
MADP_DEBATE_ROUNDS             = 3
MADP_TOKENS_PER_ROUND          = 600
MADP_MIN_CONCESSIONS_REQUIRED  = 1
MADP_SCORE_IMPROVEMENT_MIN     = 3.0
MADP_PROPONENT_CREDIBILITY_BONUS = 0.05

# ─── Feature 8: Temporal Execution Simulator ─────────────────────────────────
TES_MONTE_CARLO_N              = 500
TES_DURATION_VARIANCE          = 0.30
TES_PARALLEL_MIN_SAVINGS_PCT   = 15.0
TES_CONFLICT_DETECTION_ENABLED = True
TES_DEFAULT_DOMAIN             = "technology"

# ─── Feature 9: Semantic Plan Compression Engine ─────────────────────────────
SPCE_REDUNDANCY_THRESHOLD      = 0.85
SPCE_MERGE_THRESHOLD_LO        = 0.70
SPCE_MERGE_THRESHOLD_HI        = 0.85
SPCE_TARGET_COMPRESSION_PCT    = 0.30
SPCE_ANTI_REGRESSION_MAX_DROP  = 3.0
SPCE_MIN_STEPS_AFTER_COMPRESS  = 4

# ─── Feature 10: Autonomous Self-Correction Loop ─────────────────────────────
ASCL_ESCALATION_THRESHOLD      = 3
ASCL_MICROPLAN_MAX_STEPS       = 5
ASCL_HEALTH_DECAY_PER_DEVIATION= 8.0
ASCL_TIMELINE_SLIP_TRIGGER_PCT = 1.30
ASCL_WEBHOOK_ENDPOINT          = None
