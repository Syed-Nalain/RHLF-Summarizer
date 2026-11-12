# Analysis (Task 7)

Places marked **[fill]** need the numbers from your own training run.

## 7.1 Reward model accuracy

- Pairwise accuracy on the 1,000 validation pairs: **[fill]** (chance = 50%).
- Three pairs it gets wrong: see `results/reward_mistakes.md`. **[fill: one line each on why, e.g. both
  summaries are almost the same, or the rejected one is longer / more fluent but factually wrong.]**

## 7.2 PPO curves

- Figure: `results/fig3_ppo.png`.
- Did the mean reward-model score rise? **[fill]**. Did the length change? **[fill]**.
- A higher reward-model score is **not** the same as a better summary. PPO maximises the reward
  model, and the reward model is only an imperfect copy of human taste (accuracy above). The policy
  can find summaries the reward model likes but humans do not. Human evaluation (Table 4): **[fill]**.

## 7.3 Role of beta_KL

The PPO objective is: maximise `E[ r(x, y) ] - beta_KL * KL( pi_theta || pi_ref )`.

- `beta_KL` is the price the policy pays for moving away from the SFT model. It keeps the summaries
  fluent and on-topic, because SFT was trained on human-written summaries.
- With `beta_KL = 0` only the reward term is left. Nothing stops the policy from drifting, so it would
  exploit the weaknesses of the reward model: the score keeps rising, the KL grows without limit, and the
  text can collapse into repeated or meaningless text that the reward model happens to score highly
  (reward hacking). The entropy bonus and the clip only limit the size of each step, not the total drift.

## 7.4 Clip fraction

- In the first of the K epochs it is exactly 0, because the policy has not changed yet (ratio = 1).
- In later epochs it was about **[fill]**.
- The clip fraction is the share of tokens whose probability changed by more than 20%. A small value
  means each policy update is small and stays inside the trust region; a large value means the updates
  are too big (learning rate too high) and the clip is doing the work of limiting them.

## 7.5 Where did the reward go in DPO?

- The KL-constrained objective has a closed-form best policy:
  `pi*(y|x) = pi_ref(y|x) * exp( r(x,y) / beta ) / Z(x)`.
- Turn it around to get the reward: `r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)`.
- So the reward is hidden inside the policy itself: the "implicit reward" is
  `beta * log( pi_theta(y|x) / pi_ref(y|x) )`.
- Put this reward in the Bradley–Terry loss. The loss only uses the **difference**
  `r(x, y_w) - r(x, y_l)`. Both summaries belong to the same post `x`, so `beta * log Z(x)` is the same
  in both and cancels. That is why `Z(x)` never has to be computed.
- Accuracy on the same 1,000 validation pairs: reward model **[fill]** vs DPO implicit reward **[fill]**.

## 7.6 Cost

- PPO: **3 models** in GPU memory (policy + value head, frozen reference, frozen reward model), and it
  first needs the reward model to be trained.
- DPO: **2 models** (policy, frozen reference). No reward model, no sampling.
- Time per stage: SFT **[fill]**, reward **[fill]**, PPO **[fill]**, DPO **[fill]** (Table 2).
- Choice for a new project: **[fill – your opinion]**. (DPO is simpler: a plain supervised loss, fewer
  models, fewer settings, more stable. PPO is more flexible: it can optimise any reward, also on new
  prompts with no preference pairs, but it is harder to tune.)
