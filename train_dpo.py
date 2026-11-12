# train_dpo.py - Task 5: DPO
from common import *

BETA, LR, BATCH, LOG_EVERY = 0.1, 2e-6, 8, 250

torch.manual_seed(SEED)
tok = load_tokenizer()
pad = tok.pad_token_id
pref_train, pref_val = load_preference_splits(tok)
train_loader = DataLoader(pref_train, batch_size=BATCH, shuffle=True, collate_fn=partial(collate_pref, pad_id=pad))
val_loader = DataLoader(pref_val, batch_size=BATCH, collate_fn=partial(collate_pref, pad_id=pad))

policy = load_causal_lm("checkpoints/sft")       # trained
ref = load_causal_lm("checkpoints/sft")          # frozen
for p in ref.parameters():
    p.requires_grad = False
print("ref frozen:", all(not p.requires_grad for p in ref.parameters()))

opt = torch.optim.AdamW(policy.parameters(), lr=LR)


def seq_logprob(model, ids, mask, prompt_lens):
    """log pi(y | x) = sum of the log-probs of the summary tokens.  Shape (B,)."""
    logp = token_logprobs(model, ids, mask)
    return (logp * summary_mask(mask, prompt_lens)).sum(1)


def dpo_margin(b):
    """The expression inside sigma(.) in the DPO loss.  Shape (B,)."""
    w_ids, w_mask = b["chosen_ids"].to(device), b["chosen_mask"].to(device)
    l_ids, l_mask = b["rejected_ids"].to(device), b["rejected_mask"].to(device)
    plens = b["prompt_lens"].to(device)

    pi_w = seq_logprob(policy, w_ids, w_mask, plens)
    pi_l = seq_logprob(policy, l_ids, l_mask, plens)
    with torch.no_grad():
        ref_w = seq_logprob(ref, w_ids, w_mask, plens)
        ref_l = seq_logprob(ref, l_ids, l_mask, plens)
    return BETA * (pi_w - ref_w) - BETA * (pi_l - ref_l)


@torch.no_grad()
def validation_accuracy():
    """Implicit reward accuracy: share of pairs where the margin is positive."""
    correct = []
    for b in val_loader:
        correct += (dpo_margin(b) > 0).tolist()
    return sum(correct) / len(correct)


log = []
start = time.time()
for step, b in enumerate(train_loader, 1):
    loss = -F.logsigmoid(dpo_margin(b)).mean()
    if step == 1:
        print("first step loss (should be 0.6931):", loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % LOG_EVERY == 0 or step == len(train_loader):
        acc = validation_accuracy()
        log.append({"step": step, "train_loss": loss.item(), "val_acc": acc})
        print(log[-1])
minutes = (time.time() - start) / 60

policy.save_pretrained("checkpoints/dpo")
json.dump(log, open("results/dpo_log.json", "w"))

steps = [r["step"] for r in log]
save_plot(steps, {"DPO train loss": [r["train_loss"] for r in log],
                  "DPO implicit reward accuracy": [r["val_acc"] for r in log]}, "step", "results/fig4_dpo.png")

row = f"| DPO | {log[-1]['train_loss']:.4f} | implicit reward acc.: {log[-1]['val_acc']:.4f} | {minutes:.1f} min |\n"
open("results/table2_training.md", "a").write(row)
print(row)
