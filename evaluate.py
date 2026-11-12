# evaluate.py - Task 6: compare SFT, PPO and DPO on the 300 evaluation posts
from common import *
from train_reward import load_reward_model
from rouge_score import rouge_scorer

BATCH = 16
NAMES = ["sft", "ppo", "dpo"]

tok = load_tokenizer()
pad = tok.pad_token_id
eval_posts = load_eval_posts(tok)
n = len(eval_posts)

ref = load_causal_lm("checkpoints/sft")          # KL is measured against SFT
reward_model = load_reward_model()
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# res["ppo"]["score"] is a list with one number per post, and so on
res = {name: {"summary": [], "score": [], "kl": [], "length": [], "rouge": []} for name in NAMES}

for name in NAMES:
    model = load_causal_lm("checkpoints/" + name)
    for i in range(0, n, BATCH):
        posts = eval_posts[i:i + BATCH]
        b = collate_prompts(posts, pad)
        p_ids, p_mask = b["input_ids"].to(device), b["attention_mask"].to(device)
        P = p_ids.shape[1]

        with torch.no_grad():
            new = generate(model, p_ids, p_mask)                      # greedy
            g_mask = generated_mask(new, tok.eos_token_id)
            ids = torch.cat([p_ids, new], dim=1)
            mask = torch.cat([p_mask, g_mask.long()], dim=1)

            score = reward_model(ids, mask)
            logp = token_logprobs(model, ids, mask)[:, P - 1:]        # this policy
            ref_logp = token_logprobs(ref, ids, mask)[:, P - 1:]      # SFT
            kl = ((logp - ref_logp) * g_mask).sum(1)

        for j in range(len(posts)):
            text = tok.decode(new[j], skip_special_tokens=True).strip()
            res[name]["summary"].append(text)
            res[name]["score"].append(score[j].item())
            res[name]["kl"].append(kl[j].item())
            res[name]["length"].append(g_mask[j].sum().item())
            res[name]["rouge"].append(rouge.score(posts[j]["summary"], text)["rougeL"].fmeasure)
    print("done:", name)


def mean(xs):
    return sum(xs) / len(xs)


# ---- Table 3 ----
table = "| Policy | Mean r | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |\n|---|---|---|---|---|---|\n"
for name in NAMES:
    r = res[name]
    if name == "sft":
        win = "-"
    else:
        wins = [r["score"][i] > res["sft"]["score"][i] for i in range(n)]
        win = f"{mean(wins):.3f}"
    table += f"| {name.upper()} | {mean(r['score']):.4f} | {win} | {mean(r['kl']):.4f} | {mean(r['length']):.1f} | {mean(r['rouge']):.4f} |\n"
print(table)
open("results/table3_comparison.md", "w").write(table)

# ---- all summaries, one JSON object per line ----
with open("results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
    for i in range(n):
        row = {"id": i, "prompt": eval_posts[i]["prompt"], "reference": eval_posts[i]["summary"].strip()}
        for name in NAMES:
            row[name] = res[name]["summary"][i]
            row[name + "_score"] = res[name]["score"][i]
        f.write(json.dumps(row) + "\n")

# ---- 10 qualitative samples (add your own comments in the file afterwards) ----
text = "# Qualitative samples\n\n"
for i in range(min(10, len(eval_posts))):
    text += f"## Post {i}\n\n...{eval_posts[i]['prompt'][-500:]}\n\n"
    text += f"- **Human:** {eval_posts[i]['summary'].strip()}\n"
    for name in NAMES:
        text += f"- **{name.upper()}** (r = {res[name]['score'][i]:.2f}): {res[name]['summary'][i]}\n"
    text += "- **Comment:** \n\n"
open("results/samples.md", "w", encoding="utf-8").write(text)
