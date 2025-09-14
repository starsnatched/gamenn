from typing import Dict, List, Tuple
import torch

from .utils import set_seed


def _chat_prompt_ids(tokenizer, text: str, device):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").input_ids.squeeze(0).to(device)
        except Exception:
            pass
    return tokenizer(text, return_tensors="pt").input_ids.squeeze(0).to(device)


def ppo_step_with_imitation(
    ppo_trainer,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    pi_tokens: List[List[int]],
    pi_dists: List[List[Dict[int, float]]],
    imitation_weight: float = 0.5,
):
    device = ppo_trainer.accelerator.device
    query_tensors = [_chat_prompt_ids(tokenizer, p, device) for p in prompts]
    response_tensors = [tokenizer(r, return_tensors="pt").input_ids.squeeze(0).to(device) for r in responses]
    reward_tensors = [torch.tensor(float(r), dtype=torch.float32, device=ppo_trainer.accelerator.device) for r in rewards]
    ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    if imitation_weight > 0:
        ppo_trainer.model.train()
        opt = ppo_trainer.optimizer
        opt.zero_grad()
        loss_sum = 0.0
        for resp_ids, dist_list in zip(pi_tokens, pi_dists):
            ids = torch.tensor([resp_ids], device=ppo_trainer.accelerator.device)
            attn = torch.ones_like(ids)
            out = ppo_trainer.model.pretrained_model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1, :].contiguous()
            step_dists = dist_list[: logits.shape[1]]
            target_idx = []
            target_prob = []
            for t, d in enumerate(step_dists):
                if not d:
                    continue
                top_ids = torch.tensor(list(d.keys()), device=logits.device)
                target_idx.append(top_ids)
                target_prob.append(torch.tensor([d[i.item()] for i in top_ids], device=logits.device))
            if not target_idx:
                continue
            ce = 0.0
            for t, idxs in enumerate(target_idx):
                logp = torch.log_softmax(logits[0, t, idxs], dim=-1)
                probs = target_prob[t] / max(target_prob[t].sum().item(), 1e-8)
                ce = ce - torch.sum(probs * logp)
            loss_sum = loss_sum + ce / len(target_idx)
        if isinstance(loss_sum, float):
            loss_sum = torch.tensor(0.0, device=ppo_trainer.accelerator.device)
        loss = imitation_weight * loss_sum
        ppo_trainer.accelerator.backward(loss)
        opt.step()
        ppo_trainer.model.eval()


def simple_ppo_with_imitation(
    policy_model,
    ref_model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    pi_tokens: List[List[int]],
    pi_dists: List[List[Dict[int, float]]],
    kl_coef: float = 0.05,
    imitation_weight: float = 0.5,
    learning_rate: float = 5e-5,
):
    device = next(policy_model.parameters()).device
    opt = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    total_loss = torch.tensor(0.0, device=device)
    for prompt, response, reward, resp_ids, dist_list in zip(prompts, responses, rewards, pi_tokens, pi_dists):
        q = tokenizer(prompt, return_tensors="pt").to(device)
        r = tokenizer(response, return_tensors="pt").to(device)
        inp = torch.cat([q.input_ids, r.input_ids], dim=1)
        attn = torch.ones_like(inp)
        out = policy_model(input_ids=inp, attention_mask=attn)
        if isinstance(out, tuple):
            logits = out[0]
            values = out[2]
        else:
            logits = out.logits
            values = out.value
        qlen = q.input_ids.shape[1]
        rlen = r.input_ids.shape[1]
        if qlen + rlen < 2:
            continue
        logits_resp = logits[:, qlen - 1 : qlen + rlen - 1, :]
        target_resp = inp[:, qlen: qlen + rlen]
        logp = torch.log_softmax(logits_resp, dim=-1).gather(-1, target_resp.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_out = ref_model.pretrained_model(input_ids=inp, attention_mask=attn)
            ref_logits = (ref_out[0] if isinstance(ref_out, tuple) else ref_out.logits)[:, qlen - 1 : qlen + rlen - 1, :]
            logp_ref = torch.log_softmax(ref_logits, dim=-1).gather(-1, target_resp.unsqueeze(-1)).squeeze(-1)
        kl = (logp - logp_ref).mean()
        pg_loss = -(float(reward) * logp.mean() - kl_coef * kl)
        val_pred = values[:, qlen - 1 : qlen + rlen - 1].squeeze(-1)
        val_target = torch.full_like(val_pred, float(reward))
        vf_loss = 0.5 * torch.mean((val_pred - val_target) ** 2)
        ce_loss = torch.tensor(0.0, device=device)
        if imitation_weight > 0 and len(dist_list) > 0:
            steps = min(len(dist_list), logits_resp.shape[1])
            for t in range(steps):
                d = dist_list[t]
                if not d:
                    continue
                top_ids = torch.tensor(list(d.keys()), device=device)
                probs = torch.tensor([d[i.item()] for i in top_ids], device=device)
                probs = probs / max(probs.sum().item(), 1e-8)
                ls = torch.log_softmax(logits_resp[0, t, top_ids], dim=-1)
                ce_loss = ce_loss - torch.sum(probs * ls)
            ce_loss = ce_loss / max(1, steps)
        total_loss = total_loss + pg_loss + vf_loss + imitation_weight * ce_loss
    opt.zero_grad()
    total_loss.backward()
    opt.step()
    policy_model.eval()
