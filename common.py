# common.py - small helpers used by every stage
import sys
sys.path.append("starter")          # so we can import the starter files

import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader

from config import *
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft, collate_pref, collate_prompts

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def token_logprobs(model, input_ids, attention_mask):
    """log pi(x_t | x_<t) for every position t >= 1.  Shape (B, L-1)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits   # (B, L, V)
    logp = F.log_softmax(logits[:, :-1], dim=-1)     # position t-1 predicts token t
    targets = input_ids[:, 1:]                       # the tokens that were predicted
    return logp.gather(2, targets.unsqueeze(2)).squeeze(2)


def summary_mask(attention_mask, prompt_lens):
    """1 on summary tokens, 0 on prompt and padding.  Shape (B, L-1),
    the same shape as token_logprobs (token 0 is dropped)."""
    L = attention_mask.shape[1]
    positions = torch.arange(L, device=attention_mask.device)        # 0, 1, ..., L-1
    after_prompt = positions[None, :] >= prompt_lens[:, None]        # not a prompt token
    mask = after_prompt & (attention_mask == 1)                      # and not padding
    return mask[:, 1:].float()


def masked_mean(x, mask):
    """Mean of x only where mask is 1."""
    return (x * mask).sum() / mask.sum()


@torch.no_grad()
def generate(model, prompt_ids, prompt_mask, sample=False):
    """Returns only the new tokens, shape (B, T) with T <= 48."""
    if sample:      # PPO rollout: pure sampling
        out = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                             max_new_tokens=MAX_SUMMARY_TOKENS, do_sample=True,
                             temperature=1.0, top_k=0, top_p=1.0,
                             pad_token_id=model.config.eos_token_id)
    else:           # greedy
        out = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                             max_new_tokens=MAX_SUMMARY_TOKENS, do_sample=False,
                             pad_token_id=model.config.eos_token_id)
    return out[:, prompt_ids.shape[1]:]


def generated_mask(new_tokens, eos_id):
    """1 up to and including the first EOS, 0 after it.  Shape (B, T)."""
    is_eos = (new_tokens == eos_id).long()
    eos_before = is_eos.cumsum(1) - is_eos        # how many EOS came before this position
    return (eos_before == 0).float()


def save_plot(x, curves, xlabel, path):
    """curves = {"name": list of values}.  One small plot per curve."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 3.5))
    if len(curves) == 1:
        axes = [axes]
    for ax, name in zip(axes, curves):
        ax.plot(x, curves[name])
        ax.set_xlabel(xlabel)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------- Task 1: data report + mask check ----------------
if __name__ == "__main__":
    tok = load_tokenizer()
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)

    p_lens = [len(ex["prompt_ids"]) for ex in sft_train]
    s_lens = [len(ex["summary_ids"]) for ex in sft_train]

    table = "| | |\n|---|---|\n"
    table += f"| SFT train / validation pairs | {len(sft_train)} / {len(sft_val)} |\n"
    table += f"| Preference train / validation pairs | {len(pref_train)} / {len(pref_val)} |\n"
    table += f"| Evaluation posts | {len(eval_posts)} |\n"
    table += f"| Mean / max prompt length, SFT train (tokens) | {sum(p_lens) / len(p_lens):.1f} / {max(p_lens)} |\n"
    table += f"| Mean / max summary length, SFT train (tokens) | {sum(s_lens) / len(s_lens):.1f} / {max(s_lens)} |\n"
    print(table)
    open("results/table1_data.md", "w").write(table)

    # check: decoding the tokens where summary_mask is 1 must give the summary + EOS
    model = load_causal_lm()
    b = collate_sft(sft_train[:4], tok.pad_token_id)
    ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
    with torch.no_grad():
        logp = token_logprobs(model, ids, mask)
    smask = summary_mask(mask, b["prompt_lens"].to(device))
    print("token_logprobs shape:", tuple(logp.shape), " input shape:", tuple(ids.shape))
    for i in range(4):
        tokens = ids[i, 1:][smask[i] == 1]          # ids[:, 1:] lines up with the mask
        print("MASKED :", repr(tok.decode(tokens)))
        print("SUMMARY:", repr(sft_train[i]["summary"] + tok.eos_token))
