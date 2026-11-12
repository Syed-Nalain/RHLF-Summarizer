# train_sft.py - Task 2: supervised fine-tuning
from common import *

LR, BATCH, LOG_EVERY = 5e-5, 16, 250

torch.manual_seed(SEED)
tok = load_tokenizer()
pad = tok.pad_token_id
sft_train, sft_val = load_sft_splits(tok)
eval_posts = load_eval_posts(tok)

train_loader = DataLoader(sft_train, batch_size=BATCH, shuffle=True, collate_fn=partial(collate_sft, pad_id=pad))
val_loader = DataLoader(sft_val, batch_size=BATCH, collate_fn=partial(collate_sft, pad_id=pad))

model = load_causal_lm()
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def summary_logprobs(b):
    """Log-probs and the summary mask for one batch."""
    ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
    logp = token_logprobs(model, ids, mask)
    smask = summary_mask(mask, b["prompt_lens"].to(device))
    return logp, smask


@torch.no_grad()
def validation_loss():
    total, count = 0.0, 0.0
    for b in val_loader:
        logp, smask = summary_logprobs(b)
        total += -(logp * smask).sum().item()
        count += smask.sum().item()
    return total / count


def show_samples(title):
    """Greedy summaries for the first 5 evaluation posts."""
    b = collate_prompts(eval_posts[:5], pad)
    new = generate(model, b["input_ids"].to(device), b["attention_mask"].to(device))
    text = f"## {title}\n\n"
    for i in range(5):
        text += f"{i + 1}. {tok.decode(new[i], skip_special_tokens=True).strip()}\n"
    print(text)
    return text + "\n"


samples = show_samples("Before SFT")

log = []
start = time.time()
for step, b in enumerate(train_loader, 1):
    logp, smask = summary_logprobs(b)
    loss = -masked_mean(logp, smask)          # mean NLL of the summary tokens only

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % LOG_EVERY == 0 or step == len(train_loader):
        val = validation_loss()
        log.append({"step": step, "train_loss": loss.item(), "val_loss": val})
        print(log[-1])
minutes = (time.time() - start) / 60

model.save_pretrained("checkpoints/sft")
samples += show_samples("After SFT")
open("results/sft_samples.md", "w", encoding="utf-8").write(samples)
json.dump(log, open("results/sft_log.json", "w"))

steps = [r["step"] for r in log]
save_plot(steps, {"SFT train loss": [r["train_loss"] for r in log],
                  "SFT validation loss": [r["val_loss"] for r in log]}, "step", "results/fig1_sft.png")

row = f"| SFT | {log[-1]['train_loss']:.4f} | val. loss: {log[-1]['val_loss']:.4f} | {minutes:.1f} min |\n"
open("results/table2_training.md", "a").write(row)
print(row)
