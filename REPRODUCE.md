# Reproducing the LIT results on FAST-WAM

This repository is the FAST-WAM codebase plus a two-stage **Latent Interface Training (LIT)**
recipe. This page covers only what LIT adds; for the base model see `README.md`.

## What LIT changes

The action expert is normally conditioned on backbone visual representations directly.
LIT replaces that path:

- **Stage 1** trains the action expert on language + robot state + each chunk's terminal
  SE(3) end-effector pose, with no image observations. This yields a spatial-goal-conditioned
  action prior that never sees appearance.
- **Stage 2** restores vision, but only through 100 learnable latent tokens that aggregate
  the backbone; raw visual tokens are masked out of the action expert, and 8 of the latents
  are supervised to reconstruct the same terminal pose (`lambda_pose = 0.3`).

Stage 1's SE(3) encoder is training-time scaffolding — Stage 2 drops it and the latents
predict the pose from vision, so no privileged input is needed at inference.

## Baseline

The baseline is the released FAST-WAM checkpoint, evaluated as-is; we do not retrain it.
See the Hugging Face link in `README.md`.

## Training LIT

Both stages use `configs/sim_libero_goal_prior.yaml`, on all four LIBERO suites
(`libero_spatial / object / goal / 10`, `no_noops`).

```bash
# Stage 1 — vision-free SE(3)-conditioned action prior
python scripts/train.py task=sim_libero_goal_prior \
  model.goal_prior_stage=stage1 \
  batch_size=64 learning_rate=2e-4 warmup_steps=2000 max_steps=10000

# Stage 2 — pose-supervised latent interface, initialised from Stage 1
python scripts/train.py task=sim_libero_goal_prior \
  model.goal_prior_stage=stage2 \
  resume=<stage1_run>/checkpoints/weights/step_010000.pt \
  batch_size=16 learning_rate=1e-4 max_steps=30000
```

Interface settings are identical to the other three backbones and were not tuned per model:
`num_latents=100`, `num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`.

## Environment for evaluation

`pyproject.toml` pins the model stack (torch 2.7.1+cu128) but not the simulator. Verified on a fresh
machine on 2026-09-10 with:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
UV_INDEX_STRATEGY=unsafe-best-match \
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 uv pip install -e .

# LIBERO / LIBERO-Plus and their runtime deps (not declared by LIBERO-plus's setup.py)
ln -s /path/to/LIBERO-plus third_party/LIBERO-plus
uv pip install -e third_party/LIBERO-plus \
  robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 mujoco==3.8.1 \
  Wand h5py scikit-image future easydict thop einops matplotlib cloudpickle imageio imageio-ffmpeg
```

`Wand` needs the ImageMagick shared library (`libMagickWand`) on the system. Do not install
`robomimic` (its `egl-probe` dependency fails to build and is not needed).

Backbone weights are resolved relative to the repository root: `checkpoints/` must contain
`Wan-AI/` (Wan2.2-TI2V-5B and the Wan2.1 tokenizer), `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
and `prompt_cache.pt`, as for the released FAST-WAM.

Pick the GPU with `CUDA_VISIBLE_DEVICES=<id>` and pass `gpu_id=0`; `gpu_id` indexes the visible devices.
Single-task check (about 30 s for two rollouts):

```bash
export PYTHONPATH=$PWD/third_party/LIBERO-plus:$PWD/src MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py --config-name sim_libero_goal_prior \
  ckpt=<stage2>/step_030000.pt EVALUATION.dataset_stats_path=<stage2>/dataset_stats.json gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 EVALUATION.num_trials=2
```

## Evaluation

### LIBERO (in-distribution)

```bash
python experiments/libero/run_libero_manager.py \
  task=sim_libero_goal_prior \
  ckpt=<stage2_run>/checkpoints/weights/step_030000.pt \
  EVALUATION.dataset_stats_path=<stage2_run>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

### LIBERO-Plus (out of distribution)

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) is 10,030 perturbed tasks over seven
axes, one episode each. Point the manager at the LIBERO-Plus task set and set:

```bash
LIBERO_PLUS_FIX_LANG=1 python experiments/libero/run_libero_manager.py \
  task=sim_libero_goal_prior \
  ckpt=<stage2_run>/checkpoints/weights/step_030000.pt \
  EVALUATION.dataset_stats_path=<stage2_run>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

**`LIBERO_PLUS_FIX_LANG=1` is required.** Upstream LIBERO-Plus derives the instruction from
the perturbed file name, so on every non-language axis the policy is otherwise fed strings
like `... view 0 0 100 2 352 initstate 0`. Numbers produced without it are not comparable.

## Attention figures

`FASTWAM_ATTN_DUMP=1` captures the latent-to-image cross-attention weights during evaluation
(off by default; the normal forward path keeps `need_weights=False`).

## Checkpoints

`paper_ckpt/fastwam/lit_stage1/step_010000.pt` and `lit_stage2/step_030000.pt`, each with the
`config.yaml` and `dataset_stats.json` of its run.

## The same method on other backbones

π0.5 `jianmanlincjx/pi05` · MolmoAct2 `jianmanlincjx/Molmoact2` · ImageWAM `jianmanlincjx/ImageWAM`
