# LIT on FAST-WAM

The **Latent Interface Training (LIT)** instantiation of *Breaking the Vision–Action Shortcut: Latent
Interface Training for Generalizable Robot Foundation Models* on FAST-WAM.
Hub, project page, checkpoints: https://github.com/jianmanlincjx/LIT · https://jianmanlincjx.github.io/LIT/ ·
https://huggingface.co/linjianman/LIT (public; Stage 1 and Stage 2 for every backbone)

This is a fork of [FastWAM](https://github.com/yuantianyuan01/FastWAM) (original README:
[`README_upstream.md`](./README_upstream.md) — installation, data preparation, the released weights).
Use branch **`feat/goal-pose-prior`**.

**Environment.** `pyproject.toml` pins the model stack (torch 2.7.1+cu128) but not the simulator.
Verified on a fresh machine on 2026-09-10 with:

```bash
git clone -b feat/goal-pose-prior https://github.com/jianmanlincjx/fastwam.git && cd fastwam
uv venv --python 3.12 .venv && source .venv/bin/activate
UV_INDEX_STRATEGY=unsafe-best-match UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 uv pip install -e .

# LIBERO / LIBERO-Plus and their runtime deps (LIBERO-plus's setup.py declares none)
ln -s /path/to/LIBERO-plus third_party/LIBERO-plus
uv pip install -e third_party/LIBERO-plus robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 mujoco==3.8.1 \
  Wand h5py scikit-image future easydict thop einops matplotlib cloudpickle imageio imageio-ffmpeg
```

`Wand` needs the ImageMagick shared library (`libMagickWand`). Do not install `robomimic` (its `egl-probe`
dependency fails to build and is not needed). Backbone weights resolve relative to the repo root:
`checkpoints/` must hold `Wan-AI/` (Wan2.2-TI2V-5B, Wan2.1 tokenizer), the ActionDiT initialisation and
`prompt_cache.pt`, exactly as for the released FAST-WAM — see `README_upstream.md`.

Every LIBERO-Plus number was produced with **`LIBERO_PLUS_FIX_LANG=1`**; Overall is the mean over the seven
perturbation axes.

---

## 1. Evaluate the released checkpoint

```bash
hf download linjianman/LIT --include "fastwam/*" --local-dir ./LIT_ckpt
CK=./LIT_ckpt/fastwam/lit_stage2         # model.pt + config.yaml + dataset_stats.json
export PYTHONPATH=$PWD/third_party/LIBERO-plus:$PWD/src MUJOCO_GL=egl
```

```bash
# LIBERO (in-distribution): 50 episodes per task, 4 suites, 8 GPUs
python experiments/libero/run_libero_manager.py task=sim_libero_goal_prior \
  ckpt=$CK/model.pt EVALUATION.dataset_stats_path=$CK/dataset_stats.json MULTIRUN.num_gpus=8

# LIBERO-Plus (out of distribution): 10,030 tasks, one episode each
LIBERO_PLUS_FIX_LANG=1 python experiments/libero/run_libero_manager.py task=sim_libero_goal_prior \
  ckpt=$CK/model.pt EVALUATION.dataset_stats_path=$CK/dataset_stats.json MULTIRUN.num_gpus=8
```

Quick check (one task, two rollouts, ~30 s). Pick the GPU with `CUDA_VISIBLE_DEVICES` and keep `gpu_id=0`;
`gpu_id` indexes the visible devices:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py --config-name sim_libero_goal_prior \
  ckpt=$CK/model.pt EVALUATION.dataset_stats_path=$CK/dataset_stats.json gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 EVALUATION.num_trials=2
```

Results land under `evaluate_results/…/<suite>/gpu*_task*_results.json` plus `summary.json`; aggregate the
LIBERO-Plus run per axis from `task_success_rates.csv` with `scripts/aggregate.py` in the LIT hub (or the
`task_classification.json` mapping directly).

---

## 2. Train, then evaluate

**What you need** — the released FAST-WAM package and LIBERO data, as in `README_upstream.md`:

```bash
hf download yuanty/fastwam --local-dir ./checkpoints/fastwam_release      # Wan2.2 backbone, ActionDiT init, prompt cache
hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir ./data/libero   # all four suites, no_noops
```

The **baseline** is the released FAST-WAM checkpoint evaluated as-is (`libero_uncond_2cam224.pt`); we do not
retrain it. **LIT** is two runs on `configs/sim_libero_goal_prior.yaml`:

```bash
# Stage 1 — vision-free SE(3)-conditioned action prior
python scripts/train.py task=sim_libero_goal_prior model.goal_prior_stage=stage1 \
  batch_size=64 learning_rate=2e-4 warmup_steps=2000 max_steps=10000

# Stage 2 — pose-supervised latent interface, initialised from Stage 1
python scripts/train.py task=sim_libero_goal_prior model.goal_prior_stage=stage2 \
  resume=<stage1_run>/checkpoints/weights/step_010000.pt \
  batch_size=16 learning_rate=1e-4 max_steps=30000
```

To skip Stage 1, use the released prior: `resume=./LIT_ckpt/fastwam/lit_stage1/model.pt`.

Then evaluate `<stage2_run>/checkpoints/weights/step_030000.pt` with `<stage2_run>/dataset_stats.json`
exactly as in §1 (the released `lit_stage2/model.pt` is that file, renamed).

---

## 3. How LIT is integrated in FAST-WAM

FAST-WAM is a world–action model: a Wan2.2 video DiT backbone produces per-block features and an
ActionDiT generates action chunks conditioned on them (`src/fastwam/models/wan22/fastwam_joint.py`).
LIT touches only the conditioning path:

| Piece | Where | What it does |
| --- | --- | --- |
| Latent interface | `src/fastwam/models/wan22/goal_prior.py` (latent tokens + `GoalPoseEncoder` / `GoalPoseDecoder`), wired in `fastwam_joint.py` / `mot.py` | 100 learnable latents cross-attend to the Wan2.2 block features (and the T5 text tokens) and become the ActionDiT's only visual input |
| Firewall | `configs/model/fastwam_goal_prior_stage2.yaml` (`goal_prior_stage: stage2`) — the joint model closes the direct feature path when the stage is set | the direct feature path from the video backbone into the ActionDiT is closed; conditioning goes through the latents |
| Spatial supervision | `GoalPoseDecoder` in `goal_prior.py`, `lambda_pose = 0.3` | 8 latents decode to the chunk-end SE(3) target (position + axis-angle + gripper, quantile-normalised); MSE added to the flow-matching loss |
| Stage-1 conditioning | `GoalPoseEncoder` in `goal_prior.py`, `configs/model/fastwam_goal_prior_stage1.yaml` | encodes the terminal pose into the ActionDiT's conditioning while the video backbone is off |
| Configs | `configs/sim_libero_goal_prior.yaml`, `configs/task/libero_goal_prior_stage{1,2}.yaml` | `model.goal_prior_stage=stage1|stage2` selects the stage; Stage 2 `resume=`s the Stage-1 weights |

Interface settings match the other three backbones and were not tuned per model: `num_latents=100`,
`num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`. The action representation, chunk
length, horizon and the flow-matching objective are the upstream ones.
