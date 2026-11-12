# app/app.py - Task 8: front end.  Run from the repo root:  python app/app.py
import sys
sys.path.append(".")                 # so we can import common.py from the repo root

import gradio as gr
from common import *
from train_reward import load_reward_model

tok = load_tokenizer()
reward_model = load_reward_model()
models = {name: load_causal_lm("checkpoints/" + name) for name in ["sft", "ppo", "dpo"]}


def summarize(post):
    # build the prompt the same way as the training data: post + "\nTL;DR:"
    post_ids = tok(post.strip(), add_special_tokens=False)["input_ids"][:440]
    tldr_ids = tok("\nTL;DR:", add_special_tokens=False)["input_ids"]
    p_ids = torch.tensor([post_ids + tldr_ids], device=device)
    p_mask = torch.ones_like(p_ids)

    outputs = []
    for name in ["sft", "ppo", "dpo"]:
        new = generate(models[name], p_ids, p_mask)                  # greedy
        g_mask = generated_mask(new, tok.eos_token_id)
        ids = torch.cat([p_ids, new], dim=1)
        mask = torch.cat([p_mask, g_mask.long()], dim=1)
        with torch.no_grad():
            score = reward_model(ids, mask).item()
        outputs.append(tok.decode(new[0], skip_special_tokens=True).strip())
        outputs.append(round(score, 3))
    return outputs


with gr.Blocks(title="Summarize from Human Feedback") as demo:
    gr.Markdown("# Learning to Summarize from Human Feedback\nPaste a Reddit post and compare the SFT, PPO and DPO summaries.")
    post = gr.Textbox(label="Reddit post", lines=10)
    button = gr.Button("Summarize")
    boxes = []
    with gr.Row():
        for name in ["SFT", "PPO", "DPO"]:
            with gr.Column():
                boxes.append(gr.Textbox(label=name + " summary", lines=4))
                boxes.append(gr.Number(label=name + " reward-model score"))
    button.click(summarize, inputs=post, outputs=boxes)

demo.launch()
