"""Config for the GRPO alignment phase. See docs/grpo_alignment.txt.

Style matches training/trainer.py:TrainerConfig -- a flat dataclass with
defaults, overridden by CLI flags in train_grpo.py / env knobs in the job.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GRPOAlignConfig:
    # --- policy (trainable; NOT the AWQ eval model) ---
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- GRPO rollouts ---
    group_size: int = 8            # G rollouts per prompt (the group)
    temperature: float = 1.0       # exploration; groups need spread to give signal
    top_p: float = 1.0
    max_new_tokens: int = 1024     # <think> + <answer> budget (answer is scored)
    max_prompt_tokens: int = 3072  # scout-injected prompts are long; truncate above

    # --- optimization ---
    lr: float = 1e-6
    kl_coef: float = 0.04          # KL to the frozen reference (anti-drift)
    batch_prompts: int = 8         # per_device train batch (TRL: must be divisible
    #                                by group_size; effective prompts = batch/group)
    grad_accum: int = 1
    n_epochs: int = 1              # passes over the prompt set
    max_steps: int = 0             # 0 = run to n_epochs; else cap (smoke tests)
    seed: int = 42

    # --- reward (alignment/reward.py:RewardConfig knobs, surfaced here) ---
    reward_embedder: str = "sentence-transformers/all-mpnet-base-v2"
    match_thr: float = 0.50
    min_depth_words: int = 60     # a position must be ARTICULATED, not just named
    #                               (v1 was depth-blind; `route` namedropped all
    #                               positions and scored 0.072). PROVISIONAL --
    #                               fit against OvertonBench's human-rated
    #                               reference responses before trusting the scale.
    weight: str = "uniform"       # matches the eval's unweighted cluster count
    l_precision: float = 0.50     # exponent (reward is multiplicative now)
    l_verbose: float = 0.0        # off: the eval is monotone in coverage

    # --- scout retrieval baked into prompts (frozen during RL) ---
    tau: float = 0.25
    alpha: float = 1.0

    # --- io ---
    prompts_max: int = 0           # 0 = all usable graph questions
    save_dir: str = "grpo_lora"    # LoRA adapter output
    log_every: int = 10
