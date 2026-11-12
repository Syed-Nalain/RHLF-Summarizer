# train_reward.py - Task 3: reward model
from common import *

LR, BATCH, LOG_EVERY = 1e-5, 8, 250


class RewardModel(nn.Module):
    """Transformer body (no LM head) + one linear layer that gives a score."""

    def __init__(self, path="checkpoints/sft"):
        super().__init__()
        self.body = load_causal_lm(path).model                    # drop the LM head
        self.head = nn.Linear(self.body.config.hidden_size, 1).to(device)

    def forward(self, ids, mask):
        hidden = self.body(input_ids=ids, attention_mask=mask).last_hidden_state   # (B, L, H)
        positions = torch.arange(ids.shape[1], device=ids.device)
        last = (mask * positions).argmax(1)             # index of the last real token
        rows = torch.arange(ids.shape[0], device=ids.device)
        return self.head(hidden[rows, last]).squeeze(1)  # (B,)


def load_reward_model():
    """The trained reward model, frozen."""
    rm = RewardModel()
    rm.load_state_dict(torch.load("checkpoints/reward/reward.pt", map_location=device))
    for p in rm.parameters():
        p.requires_grad = False
    return rm


if __name__ == "__main__":
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    pad = tok.pad_token_id
    pref_train, pref_val = load_preference_splits(tok)
    train_loader = DataLoader(pref_train, batch_size=BATCH, shuffle=True, collate_fn=partial(collate_pref, pad_id=pad))
    val_loader = DataLoader(pref_val, batch_size=BATCH, collate_fn=partial(collate_pref, pad_id=pad))

    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=LR)

    def scores(b):
        r_w = rm(b["chosen_ids"].to(device), b["chosen_mask"].to(device))        # winner
        r_l = rm(b["rejected_ids"].to(device), b["rejected_mask"].to(device))    # loser
        return r_w, r_l

    @torch.no_grad()
    def validation():
        """Pairwise accuracy, and the list True/False (correct or not) for every pair."""
        correct = []
        for b in val_loader:
            r_w, r_l = scores(b)
            correct += (r_w > r_l).tolist()
        return sum(correct) / len(correct), correct

    log = []
    start = time.time()
    for step, b in enumerate(train_loader, 1):
        r_w, r_l = scores(b)
        loss = -F.logsigmoid(r_w - r_l).mean()        # Bradley-Terry loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == len(train_loader):
            acc, correct = validation()
            log.append({"step": step, "train_loss": loss.item(), "val_acc": acc})
            print(log[-1])
    minutes = (time.time() - start) / 60

    os.makedirs("checkpoints/reward", exist_ok=True)
    torch.save(rm.state_dict(), "checkpoints/reward/reward.pt")
    json.dump(log, open("results/reward_log.json", "w"))

    steps = [r["step"] for r in log]
    save_plot(steps, {"Reward model train loss": [r["train_loss"] for r in log],
                      "Reward model validation accuracy": [r["val_acc"] for r in log]}, "step", "results/fig2_reward.png")

    # three validation pairs the reward model gets wrong (for Task 7.1)
    wrong = [pref_val[i] for i in range(len(pref_val)) if not correct[i]][:3]
    text = "# Three pairs the reward model gets wrong\n\n"
    for ex in wrong:
        text += f"**Post (end):** ...{ex['prompt'][-400:]}\n\n**Human chose:**{ex['chosen']}\n\n**Human rejected:**{ex['rejected']}\n\n---\n\n"
    open("results/reward_mistakes.md", "w", encoding="utf-8").write(text)

    row = f"| Reward model | {log[-1]['train_loss']:.4f} | pairwise acc.: {log[-1]['val_acc']:.4f} | {minutes:.1f} min |\n"
    open("results/table2_training.md", "a").write(row)
    print(row)
