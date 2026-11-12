# train_ppo.py - Task 4: PPO
from common import *
from train_reward import load_reward_model

CLIP_EPS, BETA_KL, C_V, C_H, GAMMA = 0.2, 0.05, 0.1, 0.01, 1.0
K_EPOCHS, LR, BATCH, ITERATIONS = 4, 1e-5, 16, 200

torch.manual_seed(SEED)
tok = load_tokenizer()
pad = tok.pad_token_id
sft_train, sft_val = load_sft_splits(tok)

# ---- three models in memory ----
policy = load_causal_lm("checkpoints/sft")                             # trained
value_head = nn.Linear(policy.config.hidden_size, 1).to(device)        # trained
ref = load_causal_lm("checkpoints/sft")                                # frozen
reward_model = load_reward_model()                                     # frozen
for p in ref.parameters():
    p.requires_grad = False
print("ref frozen   :", all(not p.requires_grad for p in ref.parameters()))
print("reward frozen:", all(not p.requires_grad for p in reward_model.parameters()))

params = list(policy.parameters()) + list(value_head.parameters())
opt = torch.optim.AdamW(params, lr=LR)


def policy_forward(ids, mask, P):
    """For every generated token a_t: log pi(a_t|s_t), V(s_t), entropy.  Each (B, T).
    P = prompt length. Token at position P+k is predicted by position P+k-1,
    so we read logits and hidden states from positions P-1 ... L-2."""
    out = policy(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    logits = out.logits[:, P - 1:-1]                       # (B, T, V)
    hidden = out.hidden_states[-1][:, P - 1:-1]            # (B, T, H) last hidden layer
    logp_all = F.log_softmax(logits, dim=-1)
    actions = ids[:, P:]                                   # the generated tokens
    logp = logp_all.gather(2, actions.unsqueeze(2)).squeeze(2)
    values = value_head(hidden).squeeze(2)
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, values, entropy


log = []
start = time.time()
for it in range(ITERATIONS):
    # ---------- 1. rollout: sample one summary for 16 prompts ----------
    b = collate_prompts(sft_train[it * BATCH:(it + 1) * BATCH], pad)
    p_ids, p_mask = b["input_ids"].to(device), b["attention_mask"].to(device)
    P = p_ids.shape[1]

    with torch.no_grad():
        new = generate(policy, p_ids, p_mask, sample=True)          # (B, T)
        g_mask = generated_mask(new, tok.eos_token_id)              # 1 until first EOS
        ids = torch.cat([p_ids, new], dim=1)
        mask = torch.cat([p_mask, g_mask.long()], dim=1)
        T = new.shape[1]

        # ---------- 2. store old values (no gradients) ----------
        old_logp, old_values, _ = policy_forward(ids, mask, P)
        ref_logp = token_logprobs(ref, ids, mask)[:, P - 1:]
        score = reward_model(ids, mask)                             # (B,)

        # ---------- 3. per-token reward ----------
        kl = (old_logp - ref_logp) * g_mask
        rewards = -BETA_KL * kl
        last = g_mask.sum(1).long() - 1                             # last token of each summary
        rewards[torch.arange(len(last)), last] += score

        # ---------- 4. returns and advantages ----------
        returns = torch.zeros_like(rewards)
        running = torch.zeros(len(last), device=device)
        for t in reversed(range(T)):
            running = rewards[:, t] + GAMMA * running
            returns[:, t] = running
        adv = returns - old_values
        real = g_mask.bool()
        adv = (adv - adv[real].mean()) / (adv[real].std() + 1e-8)

    # ---------- 5. K epochs of PPO updates on the same rollout ----------
    clip_fracs = []
    for epoch in range(K_EPOCHS):
        logp, values, entropy = policy_forward(ids, mask, P)
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)

        l_ppo = masked_mean(torch.min(ratio * adv, clipped * adv), g_mask)
        l_value = masked_mean((values - returns) ** 2, g_mask)
        l_entropy = masked_mean(entropy, g_mask)
        loss = -l_ppo + C_V * l_value - C_H * l_entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        outside = ((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).float()
        clip_fracs.append(masked_mean(outside, g_mask).item())

    # ---------- 6. logging ----------
    log.append({"iter": it + 1,
                "score": score.mean().item(),
                "kl": kl.sum(1).mean().item(),
                "length": g_mask.sum(1).mean().item(),
                "clip_frac": sum(clip_fracs) / K_EPOCHS,
                "clip_frac_epoch1": clip_fracs[0],
                "l_ppo": l_ppo.item(), "l_value": l_value.item(), "l_entropy": l_entropy.item()})
    print(log[-1])
minutes = (time.time() - start) / 60

# ---------- 7. save the policy only (no value head) ----------
policy.save_pretrained("checkpoints/ppo")
json.dump(log, open("results/ppo_log.json", "w"))

iters = [r["iter"] for r in log]
save_plot(iters, {"PPO mean reward score": [r["score"] for r in log],
                  "PPO mean KL to reference": [r["kl"] for r in log],
                  "PPO mean length": [r["length"] for r in log],
                  "PPO clip fraction": [r["clip_frac"] for r in log]}, "iteration", "results/fig3_ppo.png")

final_score = sum(r["score"] for r in log[-10:]) / len(log[-10:])     # mean of the last 10 iterations
row = f"| PPO | {loss.item():.4f} | final mean r: {final_score:.4f} | {minutes:.1f} min |\n"
open("results/table2_training.md", "a").write(row)
print(row)
