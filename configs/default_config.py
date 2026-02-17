"""
default_config.py
-----------------
Central configuration dictionary.  Every runnable script imports from here
and overrides only what it needs.

Domain-randomization ranges (``dr_*``) define the uniform sampling bounds
used during training so the agent learns a cost-aware conditional policy.
"""

DEFAULT_CONFIG = dict(
    # --- Environment (nominal / default values) ---
    unit_revenue=12.0,
    jit_unit_cost=8.0,
    llt_unit_cost=4.0,
    holding_cost_per_unit=1.0,
    shortage_penalty_per_unit=10.0,
    capacity_penalty_per_unit=5.0,
    storage_capacity=200.0,
    max_order=50.0,
    episode_length=150,
    demand_distribution="poisson",
    demand_mean=20.0,
    demand_std=5.0,
    demand_spike_prob=0.0,
    demand_spike_multiplier=3.0,
    jit_lead_time=1,
    llt_lead_time=6,
    fill_rate_min=0.8,
    fill_rate_max=1.0,
    supply_disruption_prob=0.0,
    hard_capacity=True,
    demand_history_len=5,
    seed=42,

    # --- Domain-randomization ranges (training only) ---
    #     Each range is [low, high] inclusive, sampled uniformly per episode.
    domain_randomization=False,
    dr_jit_cost=(2.0, 15.0),
    dr_llt_cost=(1.0, 12.0),
    dr_holding_cost=(0.2, 3.0),
    dr_shortage_penalty=(3.0, 25.0),
    dr_storage_capacity=(80.0, 400.0),
    dr_llt_lead_time=(2, 10),               # integer range
    dr_demand_mean=(10.0, 40.0),

    # --- Normalization constants (max values for obs scaling) ---
    #     Used to normalise scenario params into roughly [0, 1].
    norm_cost_max=25.0,
    norm_capacity_max=500.0,
    norm_lead_time_max=12.0,

    # --- Training ---
    total_timesteps=300_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,

    # --- Paths ---
    model_save_path="models/dual_source_rl",
    log_dir="logs/",
)
