# Goal-Pose Prior → FastWAM

Port of the two-stage goal-pose prior from ImageWAM v2 onto FastWAM
(`Wan2.2-TI2V-5B` video DiT + ActionDiT, MoT mixed attention).

**Repo** `/data2/JM/Code/FastWAM` · **venv** `.venv` (py3.10, torch 2.7.1+cu128)
**Reference implementation** `/data2/JM/Code/ImageWAM-v2`

---

## 0. Why FastWAM

| architecture | model | LIBERO-Plus baseline | our gain |
|---|---|---|---|
| VLA | MolmoAct2 (V3) | 58.89 | **+8.38** (w/o Language +10.23) |
| WAM | ImageWAM v2 | 83.01 | +1.44 (w/o Language +3.03) |
| WAM | **FastWAM** | **51.5** (paper Table 3) | — |

The method is established on VLA. On WAM the ImageWAM gain is significant
(p<1e-4) but too small to carry the claim. FastWAM starts low, so there is room.

Our own data motivates the choice: across 27 (suite×axis) cells on ImageWAM,
Pearson r(baseline level, our delta) = **−0.767**; cells with baseline<70% went
6/6 with mean +12.04pp. The same shape repeats inside the MolmoAct2 table.

**The weakness of that argument, stated plainly:** a low baseline is easier to
beat by any means, and regression to the mean produces the same correlation.
Low-start→big-gain is necessary evidence, not sufficient. The real criterion is
that the gain must have the **same shape** as MolmoAct2 — Noise / Layout /
Camera largest — not a uniform lift.

FastWAM's published per-axis scores, i.e. where the headroom is:

| Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|
| 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | **51.5** |

**Expect Language to lose.** It is negative on both architectures already
(MolmoAct2 −1.82, ImageWAM −7.71).

---

## 1. What FastWAM does today

MoT sequence is `[video | action]`, and the mask (`_build_mot_attention_mask`)
says:

```python
mask[:V, :V] = video→video      # first_frame_causal
mask[V:, V:] = True             # action→action, full
mask[V:, :F] = True             # action→video: FIRST FRAME ONLY
```

**The action tower never sees the future.** That is the Fast-WAM thesis — *Do
World Action Models Need Test-time Future Imagination?*, answered no — and
`infer_action` accordingly sets `timestep_video = zeros` and runs a single clean
forward with no denoising loop.

They validated that claim in-distribution (97.6%). Under perturbation it falls
to 51.5, with Camera at 16.4.

**Our claim becomes:** you do not need the full future *video*, but you do need a
compact, gated **goal-pose prior**. This lands directly on their thesis rather
than beside it.

Sizes that matter: `video_size [224,448]` → Wan VAE ÷16 → 14×28, patch `[1,2,2]`
→ **98 tokens per latent frame**, 9 latent frames ≈ **882 video tokens**, against
**32 action tokens**. Text arrives as cross-attention `context` (T5, dim 4096),
with the first-step proprio already appended to it.

---

## 2. Where the goal tokens attach

MoT carries **separate context payloads per expert** (`video_context_payload`,
`action_context_payload`). So the goal tokens ride the action expert's
cross-attention, not the MoT token sequence:

```
action_context = [ text | proprio | goal ]      goal = 8 (Stage 1) or 100 (Stage 2)
video_context  = [ text | proprio ]             unchanged — no leakage
```

Three things fall out of this, and they are the reason to prefer it over adding a
third block to the MoT sequence:

* the video expert is untouched, so nothing leaks into the world model;
* the gate is a **logit bias on those columns**, which is exactly the mechanism
  `syn_gate_bias` already uses in ImageWAM;
* `_build_mot_attention_mask` needs **no syn block at all**.

---

## 3. Stage 1 — vision-free pose-conditioned action prior

**Goal:** learn π(action | language, proprio, goal-pose) with **no visual input
whatsoever**, so the policy cannot memorise appearance and has to use the pose
signal. That is where the domain-invariant steer comes from.

**Topology** `[null(32) | action]` in MoT · `action_context = [text | proprio | goal(8)]`

**Trainable policy** — mirrors `apply_trainable_policy()`:

| module | Stage 1 |
|---|---|
| `action_expert` | train, `requires_grad_(True)` |
| `goal_pose_encoder` | train, `requires_grad_(True)` |
| `stage1_null_tokens` | `nn.Parameter`, **hand-collected** (bare Parameter, not a module) |
| `video_expert` | `.eval()`, `requires_grad_(False)` — **frozen** |

**The video tower is skipped entirely, not merely frozen.** With no video loss
and no gradient, running its forward buys nothing. Skipping it drops the action
expert's self-attention sequence from ~130 tokens to 32 and is what makes a large
batch affordable. The 32 learnable null tokens stand in for that removed span so
the attention shape and statistics do not jump at the Stage 1 → Stage 2 handoff.

**Goal pose = the last observation of the sample window.** ImageWAM does exactly
this — `extract_goal_pose_from_proprio`, *"Split the last observation as goal
before aligning proprio with the action horizon."* On FastWAM with
`num_frames: 33` that is `proprio[:, -1, :]`, an 8-vector (eef_pose 6 + gripper 2).
Note today's `build_inputs` keeps only `proprio[:, 0, :]`; the last frame is
currently discarded and must be threaded through.

**`GoalPoseEncoder`** — pure MLP, no vision:
`Linear(8→512) → GELU → Linear(512→512) → GELU → Linear(512→8×D)` → `[B, 8, D]`.

**Loss** `lambda_action 1.0` + `lambda_pose 0.3`, **`lambda_video = 0`**.
(In ImageWAM the config leaves `lambda_video: 0.5` on, but with the video expert
frozen that gradient has nowhere to go — it is inert there and simply wasteful
here.)

**Schedule** bs **64/GPU** (to be measured, fall back 32) × 8 = 512 global,
lr 2e-4, warmup 2000, **10,000 steps**, `save_every 5000`.

---

## 4. Stage 2 — inferred tokens behind a gate

**Everything opens up:**

| module | Stage 2 |
|---|---|
| `action_expert` | train |
| `video_expert` | train, `requires_grad_(True)` — **unfrozen** |
| `semantic_visual_aggregator` | train |
| `semantic_visual_pose_norm` / `_pose_decoder` | train |
| `goal_pose_encoder`, `stage1_null_tokens` | **not carried over** (`STAGE1_ONLY_CHECKPOINT_PREFIXES`) |

The oracle channel and the placeholder both retire with Stage 1.

**Topology** `[video(first frame) | action]` · `action_context = [text | proprio | syn(100)]`
where 100 = **8 pose-supervised + 92 free context**.

**The aggregator is interleaved into the DiT forward, not a one-shot head.**
`forward_layer(queries, semantic_hidden, visual_hidden, layer_idx, num_layers)` —
the 100 latents are refreshed at every layer, cross-attending the text and image
streams *at that depth*, with parameters shared within each of 5 groups.
`layer_group_index = layer_idx // (num_layers // num_layer_groups)` requires
divisibility: ActionDiT has 30 layers → **5 groups × 6 layers**.

**Why this is still compatible with option B (no test-time video generation):**
`infer_action` already runs one clean video forward (`timestep_video = zeros`, no
denoise loop). The per-layer hidden states the aggregator needs are a by-product
of that single forward. So the cost is one forward, not a denoising loop, and the
190 ms latency claim survives.

**Dual-stream fusion is kept.** The cross-attention block is
`nn.MultiheadAttention(embed_dim=latent_dim, kdim=context_dim, vdim=context_dim)`
— key/value width is already independent of the latent width, so T5 at 4096 plugs
straight in. Functionally it is not optional: the goal tokens have to encode
*which* object and *where*, which is a language×vision binding. Image-only latents
cannot separate "put the **black bowl**" from "put the **plate**" — they would be
instruction-blind, and Language is already the one axis we lose on both
architectures.

`context_dim = 4096` (T5) · `kv_dim = 3072` (Wan hidden) · `latent_dim = 768`
→ one projection into ActionDiT's `hidden_dim 1024` (ImageWAM had 3072→3072 and
needed none).

**Split gate.** One scalar `syn_gate_bias` per layer group, init **−2.0**, and
`gate_pose_tokens: false` so only columns 8..100 are gated. At step 0 the 92
context columns are suppressed ~150×, leaving exactly Stage 1's
`[text | proprio | pose]` interface rather than a pose-conditioned prior with no
pose. The bias is learnable, so the gate can open on its own.
`zero_init_value: false` — `to_value` is shared across all syn tokens, so zeroing
it would silence the pose columns too.

**Regularisers** (training only, off at inference): `context_token_dropout 0.05`
drops whole columns; `context_blackout_prob 0.10` masks the entire context block.
The fallback state is again Stage 1's interface.

**Channel regime.** `action_sees_ref: true`. Sampling uses **fixed counts**, not
independent coins, so every rank sees every regime in every batch (ranks that
disagree about which metric keys exist deadlock the all-gather):

```python
n_ref_only = round(p_ref_only * n);  n_syn_only = round(p_syn_only * n)
n_ref_only = max(1, min(n_ref_only, n-2))
n_syn_only = max(1, min(n_syn_only, n-1-n_ref_only))
n_both     = n - n_ref_only - n_syn_only
```

**Copying ImageWAM's config numbers would not reproduce ImageWAM's behaviour,**
because the batch differs:

| n | config 0.55/0.15/0.30 → effective |
|---|---|
| 10 (ImageWAM) | ref=2, syn=3, both=5 → **0.50 / 0.20 / 0.30** ← what was validated |
| 16 (FastWAM) | ref=2, syn=5, both=9 → 0.5625 / **0.125** / 0.3125 |

`p_ref_only` would fall by 38% relative — and that path is precisely what breaks
the policy's dependence on the synthetic channel. So **FastWAM's config is written
as the effective values `0.50 / 0.20 / 0.30`**, giving ref=3, syn=5, both=8 →
0.50 / 0.1875 / 0.3125 at n=16, within 0.013 of the validated ratio.

**Schedule** bs 16/GPU × 8 = 128 global, lr 1e-4, warmup 5% rule, **17,420 steps
(10 epochs)** — deliberately identical to FastWAM's own published LIBERO recipe,
so Stage 2 takes no extra steps. `save_every 4000`.

---

## 5. Compute

`len(dataset)` is `multi_dataset.num_frames` = **277,713** — LeRobot treats every frame as a
window start and clamps the tail rather than dropping it. The published recipe's 10 epochs
at global batch 128 is therefore **21,700 steps**, not the 17,420 an earlier estimate here
reached by subtracting the window tail.

| | batch/GPU | global | steps | samples | wall clock |
|---|---|---|---|---|---|
| FastWAM published recipe | 16 | 128 | 21,700 | 2,777,600 | — |
| Stage 1 | 64 | 512 | 10,000 | 5,120,000 | 1.5 h @ 1.84 step/s |
| Stage 2 | 16 | 128 | **30,000** | 3,840,000 | 22.5 h @ 0.37 step/s |
| **total** | | | 40,000 | **8,960,000** | **~24 h** |

**Why 30k and not the recipe's 21.7k.** Matching a recipe we never ran ourselves is a weak
form of matching. The FastWAM baseline trained later uses the same 30k, and *that* is
step-matched exactly. The price is saying plainly that Stage 2 runs 1.38x the published
recipe's steps.

**Sample count overstates the cost; GPU-hours are the honest metric.** Stage 1 is 12 of the
~192 GPU-hours — about 6% — because it is vision-free: no VAE, no video expert, 83% of
parameters frozen, so a sample there is ~27x cheaper. It also introduces **no new data**:
the same 1,712 episodes, more passes, and those passes carry no visual channel at all, so
"they extracted more from the images" is not available as an objection.

**Checkpoints every 5,000 steps.** LIBERO-Plus punishes overfitting to the training
distribution and 30k is past the published recipe, so whether the extra steps help or hurt
has to be observed rather than assumed. **The 30k checkpoint is the primary result**; the
intermediates show the trend and are not to be selected from after the fact.

## 5b. The Stage 1 → Stage 2 handoff spike

Stage 1 ends at action loss 0.0097; Stage 2 starts at 11.79 and is back to 0.106 by step
1,420. Worth being precise about the cause, because the obvious explanation is wrong.

It is **not** a topology jump. The action expert's visible span only goes from 64 tokens
(32 action + 32 placeholders) to 130 (98 first-frame video + 32 action) — `mask[V:, :F]`
admits the first frame only, not all 882 video tokens.

It is the **pose channel turning to noise**. In Stage 1 those 8 columns carry the oracle
pose through a trained encoder. In Stage 2 the same 8 columns are produced by a freshly
initialised aggregator — and they are deliberately *ungated*, since gating them would leave
a pose-conditioned prior with no pose at all. So the cold start reproduces Stage 1's
topology but not Stage 1's *signal*: the prior's most load-bearing input is briefly garbage,
and the gate by construction does not cover it.

Recovery is fast — `loss_pose` falls 0.0696 → 0.0055 within 1,420 steps, and the action loss
follows. ImageWAM has the same structure, so this is inherent to the design rather than a
port defect. But if the Stage 1 prior turns out not to pay off, **this is the first place to
look**, and there are two obvious remedies: gate the pose columns briefly as well, or warm
the aggregator against the oracle pose for a few hundred steps before Stage 2 proper.

## 6. Sanity anchors

We are **not** training a FastWAM baseline — the published 51.5 is the reference.
Two consequences, one mitigated and one accepted:

**Accepted:** no matched-pair statistics. Every ImageWAM claim was McNemar over
identical task ids; against a published average we can only compare average to
average — no p-value, no per-task win/loss. If we later want it, evaluating the
*released* checkpoint (`libero_uncond_2cam224.pt`) buys pairing without training,
though 10,030 tasks is not itself cheap.

**Mitigated:** "is the number wrong, or is my setup broken?" The paper reports
**97.6 in-distribution**. Run the 40-task in-dist eval after every training
(~50 min). Landing near 97 means the pipeline is healthy; landing at 60 means the
setup is broken and has nothing to do with the idea. Cheapest available probe —
run it before trusting any LIBERO-Plus number.

Independent evidence the eval pipeline is faithful: the paper's ImageWAM is 83.1
and we measured **83.01** on the same protocol.

---

## 7. Deviations from ImageWAM, and why

| # | ImageWAM | FastWAM | reason |
|---|---|---|---|
| 1 | goal tokens in the token stream | action expert's cross-attn context | MoT already separates per-expert context; leaves the video expert and the MoT mask untouched |
| 2 | Stage 1 image stream is zero-length | video tower **skipped**, action→video cut | no video loss and no gradient ⇒ its forward buys nothing; this is what affords the large batch |
| 3 | `lambda_video 0.5` in Stage 1 | `0` | inert with the tower frozen |
| 4 | 25 layers / 5 groups | 30 layers / 5 groups × 6 | `layer_group_index` needs divisibility; 30 % 5 == 0 |
| 5 | 3072→3072, no projection | 768 → 1024 projection | ActionDiT hidden is 1024 |
| 6 | config 0.55/0.15/0.30 | config 0.50/0.20/0.30 | fixed-count sampling is batch-dependent; see §4 |

---

## 8. Open

* Stage 2 peaks at **97.6%** of an 80 GB card. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is set to limit fragmentation, but a long run has little headroom.
* We are not training a FastWAM baseline yet, so there are no matched pairs and no p-value —
  only average vs the published average. Evaluating the released `libero_uncond_2cam224.pt`
  would buy pairing without training, if the result warrants it.

---

## 9. Silent failures found in review

Every one of these let training run to completion with falling losses and a clean log,
while the mechanism it governs did nothing.

| what | why it was silent |
|---|---|
| **The gate could never move.** Casting the aggregator to the backbone dtype made `syn_gate_bias` a bf16 scalar at −2.0, where the grid is 0.0078 while an AdamW step moves ~1e-4 — every update rounded straight back. | In the optimizer, receiving gradient 1.32, graph intact. Only the *value* betrayed it. Fixed by storing a delta from the init, which starts at 0 where bf16 is dense; the sum is taken in fp32 so the gate moves continuously. Casting the module to fp32 instead fails: autocast hands bf16 activations to fp32 LayerNorms. |
| **`infer_action` had no goal prior at all.** Stage 2 trains with 100 gated latents in the action context; inference ran the stock path without them. | No error — the policy would simply be evaluated without the input it learned to use. Fixed by collecting the latents during the video prefill (the video stream never attends the action stream, so one pass covers every denoise step) and appending them per layer. |
| **The aggregator was fed the wrong context at inference** — the video expert's projected 3072-d copy instead of the raw 4096-d T5 context. | Caught only because the inference test also checked that removing the goal channel changes the output (it moves 112% of scale). |
| **`save_checkpoint` stored none of the new modules.** | Reload silently returns randomly initialised ones. |
| **`warmup_steps` was ignored**; the trainer hardcoded the 5% rule. | A config key that reads as honoured. |
| **`_skip_video_encoding` gated on `self.training`**, which is always False because the trainer runs `model.eval()` then re-enables only `model.dit`. | The optimisation never ran; fixing it took Stage 1 from 0.06 to 1.48 step/s. |

The lesson worth keeping: *in the optimizer* + *receiving gradient* + *loss decreasing* does
not imply *the mechanism works*. Each load-bearing part needs its own check — for the gate,
that its value leaves its initialisation; for inference, that removing the channel changes
the output.
