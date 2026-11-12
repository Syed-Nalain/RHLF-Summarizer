# human_eval.py - Section 3.2: blind human evaluation on 30 posts
# Run 1: creates results/human_eval.csv (summaries A/B/C in random order).
#        Each member types A, B or C in their own column (member1 / member2).
# Run 2: counts the picks and prints Table 4.
import os, csv, json, random

SHEET = "results/human_eval.csv"
KEY = "results/human_eval_key.csv"      # which model is A/B/C - do not open while rating
NAMES = ["sft", "ppo", "dpo"]

if not os.path.exists(SHEET):
    rows = [json.loads(line) for line in open("results/eval_summaries.jsonl", encoding="utf-8")]
    rng = random.Random(42)                     # fixed seed
    chosen = rng.sample(rows, 30)

    sheet = [["id", "post", "A", "B", "C", "member1", "member2"]]
    key = [["id", "A", "B", "C"]]
    for r in chosen:
        order = NAMES[:]
        rng.shuffle(order)                      # hide which model wrote which
        sheet.append([r["id"], r["prompt"], r[order[0]], r[order[1]], r[order[2]], "", ""])
        key.append([r["id"]] + order)
    csv.writer(open(SHEET, "w", newline="", encoding="utf-8")).writerows(sheet)
    csv.writer(open(KEY, "w", newline="", encoding="utf-8")).writerows(key)
    print("Created", SHEET, "- fill in the member1 and member2 columns, then run again.")

else:
    sheet = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    key = list(csv.DictReader(open(KEY, encoding="utf-8")))
    counts = {"member1": {n: 0 for n in NAMES}, "member2": {n: 0 for n in NAMES}}
    agree = 0
    for s, k in zip(sheet, key):
        pick1 = k[s["member1"].strip().upper()]         # letter -> model name
        pick2 = k[s["member2"].strip().upper()]
        counts["member1"][pick1] += 1
        counts["member2"][pick2] += 1
        agree += pick1 == pick2

    table = "| | SFT picked | PPO picked | DPO picked |\n|---|---|---|---|\n"
    for m in ["member1", "member2"]:
        table += f"| {m} | {counts[m]['sft']} | {counts[m]['ppo']} | {counts[m]['dpo']} |\n"
    table += f"| Agreement between members | {agree} / {len(sheet)} | | |\n"
    print(table)
    open("results/table4_human_eval.md", "w").write(table)
