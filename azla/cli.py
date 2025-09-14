import argparse
import logging
import numpy as np
from accelerate import Accelerator
from trl import PPOConfig, PPOTrainer

from .config import Config
from .logging_utils import configure_logging, get_logger
from .utils import set_seed
from .modeling import ModelManager
from .datasets import load_jsonl_prompts
from .mcts import MCTSConfigRuntime
from .selfplay import play_one
from .curriculum import CurriculumController
from .train_ppo import ppo_step_with_imitation
from .checkpoint import merge_and_save


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run", type=str, choices=["smoke", "train"], default="smoke")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    configure_logging(cfg.logging.level)
    log = get_logger("azla")
    set_seed(cfg.training.seed)
    accel = Accelerator()
    mm_a = ModelManager(
        cfg.model.pretrained,
        quantization_4bit=cfg.model.quantization_4bit,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.target_modules,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
    )
    mm_b = ModelManager(
        cfg.model.pretrained,
        quantization_4bit=cfg.model.quantization_4bit,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.target_modules,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
    )
    tokenizer = mm_a.tokenizer
    mcts_rt = MCTSConfigRuntime(
        num_sims=cfg.mcts.num_sims,
        c_puct=cfg.mcts.c_puct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_frac=cfg.mcts.dirichlet_frac,
        progressive_widening_k=cfg.mcts.progressive_widening_k,
        progressive_widening_alpha=cfg.mcts.progressive_widening_alpha,
        root_top_k=cfg.mcts.root_top_k,
    )
    prompts = ["Hello world"]
    if cfg.dataset.path:
        prompts = load_jsonl_prompts(cfg.dataset.path, cfg.dataset.field_prompt, cfg.dataset.max_samples)
    rng = np.random.default_rng(cfg.training.seed)
    if args.run == "smoke":
        out = play_one(mm_a, mm_b, tokenizer, mcts_rt, cfg.model.max_length // 8, cfg.model.temperature, rng, prompts[0])
        log.info("smoke_done", extra={"winner": out["winner"], "les_a": out["les_a"], "les_b": out["les_b"]})
        return
    ppo_cfg = PPOConfig(
        steps=cfg.training.steps,
        learning_rate=cfg.training.learning_rate,
        mini_batch_size=cfg.training.mini_batch_size,
        batch_size=cfg.training.batch_size,
        ppo_epochs=cfg.training.ppo_epochs,
        target_kl=cfg.training.kl_target,
    )
    ppo_trainer = PPOTrainer(ppo_cfg, mm_a.model, ref_model=mm_b.model, tokenizer=tokenizer)
    cur = CurriculumController(
        cfg.mcts,
        window=cfg.curriculum.window,
        inc_thr=cfg.curriculum.increase_threshold,
        dec_thr=cfg.curriculum.decrease_threshold,
        min_sims=cfg.curriculum.min_sims,
        max_sims=cfg.curriculum.max_sims,
    )
    for step in range(cfg.training.steps):
        batch_prompts = prompts[: cfg.training.batch_size]
        results = []
        for p in batch_prompts:
            out = play_one(mm_a, mm_b, tokenizer, mcts_rt, cfg.model.max_length, cfg.model.temperature, rng, p)
            results.append(out)
            cur.update(max(out["les_a"], out["les_b"]))
        mcts_rt.num_sims = cur.adjust().num_sims
        prompts_text = []
        responses_text = []
        rewards = []
        pi_tokens = []
        pi_dists = []
        for r in results:
            if r["winner"] == 0:
                ids = r["ids_a"]
                pi = r["pi_a"]
                rw = r["les_a"]
            else:
                ids = r["ids_b"]
                pi = r["pi_b"]
                rw = r["les_b"]
            split = max(1, len(ids) // 3)
            prompts_text.append(tokenizer.decode(ids[:split], skip_special_tokens=True))
            resp_tokens = ids[split:]
            responses_text.append(tokenizer.decode(resp_tokens, skip_special_tokens=True))
            rewards.append(rw)
            pi_tokens.append(resp_tokens)
            pi_dists.append([d for _, d in pi[split:]])
        ppo_step_with_imitation(ppo_trainer, tokenizer, prompts_text, responses_text, rewards, pi_tokens, pi_dists, imitation_weight=0.5)
        log.info("train_step", extra={"step": step, "avg_reward": float(sum(rewards) / max(1, len(rewards))), "num_sims": mcts_rt.num_sims, "mode": "trl_ppo"})
    out_dir = f"{cfg.logging.out_dir}/final"
    try:
        merge_and_save(mm_a.model, tokenizer, out_dir)
    except Exception as e:
        merge_and_save(mm_a.model, tokenizer, out_dir)
    log.info("model_saved", extra={"path": out_dir})


if __name__ == "__main__":
    main()
