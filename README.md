# RLHF from Scratch: Learning to Summarize from Human Feedback

A practice reproduction of *Learning to Summarize from Human Feedback* (Stiennon et al., 2020) on a
135M-parameter model (`HuggingFaceTB/SmolLM2-135M`). The four stages are written directly in PyTorch,
without TRL, PEFT or HF Trainer: **SFT → reward model → PPO → DPO**.


## Results (300 held-out Reddit posts)

| Policy | Mean reward | Win rate vs SFT | KL to SFT | Length (tokens) | ROUGE-L |
|---|---|---|---|---|---|
| SFT | 2.16 | – | 0.00 | 29.5 | 0.256 |
| PPO | 3.27 | 89.0% | 6.17 | 33.4 | 0.240 |
| DPO | **3.35** | **92.0%** | **2.84** | 36.8 | **0.257** |

| Stage | Final train loss | Validation metric | Time (A10G) |
|---|---|---|---|
| SFT | 2.409 | val. loss 2.354 | 20.8 min |
| Reward model | 0.627 | pairwise acc. 64.3% | 38.8 min |
| PPO (200 iters) | -0.029 | final mean reward 3.17 | 20.0 min |
| DPO | 0.672 | implicit reward acc. 61.4% | 65.3 min |

Total cloud cost: about **$3** on Modal.

Figures: `results/fig1_sft.png`, `fig2_reward.png`, `fig3_ppo.png`, `fig4_dpo.png`.
Samples: `results/samples.md`. All summaries: `results/eval_summaries.jsonl`.

## Files

| File | Purpose |
|---|---|
| `starter/` | data loading, cleaning, batching, model loading |
| `common.py` | `token_logprobs`, `summary_mask`, generation helper, data stats |
| `train_sft.py` | supervised fine-tuning (NLL on summary tokens only) |
| `train_reward.py` | reward model (body + linear head) with Bradley–Terry loss |
| `train_ppo.py` | value head, rollouts, KL-penalised rewards, returns, clipped PPO loss |
| `train_dpo.py` | DPO against a frozen SFT reference |
| `evaluate.py` | reward, win rate, KL, length and ROUGE-L for all three policies |
| `human_eval.py` | blind A/B/C sheet for human preference evaluation |
| `app/app.py` | Gradio front end that shows the three summaries and their reward scores side by side |
| `modal_app.py` | runs the whole pipeline on a Modal GPU (`--mode smoke` / `full`, `--start N` to resume) |
| `rlhf_pipeline.ipynb` | the same pipeline as a single notebook |

## How to run

Locally or on Colab/Kaggle (T4 is enough):

```bash
pip install -r requirements.txt
python common.py && python train_sft.py && python train_reward.py
python train_ppo.py && python train_dpo.py && python evaluate.py
python app/app.py
```

On Modal:

```bash
python -m modal run modal_app.py --mode smoke            # 2-minute end-to-end test on tiny data
python -m modal run --detach modal_app.py --mode full    # full run (~2.5 h on A10G)
python -m modal volume get rlhf-vol full/results/ ./results_full/
```

## Correctness checks (from the real run)

- **SFT mask:** decoding the masked tokens gives exactly the summary followed by EOS.
- **Frozen models:** `ref frozen: True`, `reward frozen: True` in PPO and DPO.
- **PPO, iteration 1:** KL to the reference = 0.0.
- **PPO, every iteration:** clip fraction in the first epoch = 0.0, so the ratio is 1.
- **DPO, first step:** loss = 0.6931 = log 2.
- **Padding:** prompts are left-padded for generation; the reward model reads the last real token via `attention_mask`.

## What I learned

- PPO and DPO both beat SFT on reward-model score by about 1.1–1.2 points, and win roughly 90% of head-to-heads.
- DPO got a slightly higher reward while drifting **less than half as far** from SFT (KL 2.8 vs 6.2), and kept ROUGE-L.
- PPO's ROUGE-L fell a little: it optimises the reward model, not the human reference.
- Both methods made summaries longer, a common reward-model length bias.
- PPO keeps 3 models in GPU memory (policy + value head, reference, reward model). DPO needs only 2 and has no sampling loop, so it is simpler.
- Gotcha: recent `transformers` loads SmolLM2 in bfloat16. Full fine-tuning needs `dtype=torch.float32`, or the reward head crashes and small DPO updates vanish.


