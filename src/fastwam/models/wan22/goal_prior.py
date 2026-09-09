"""Goal-pose prior for FastWAM.

Stage 1 trains a *vision-free* action prior: the policy sees language, proprio and
an oracle goal pose, and no pixels at all, so it cannot memorise appearance and has
to use the pose signal. Stage 2 keeps that prior and replaces the oracle with tokens
inferred from the observation, behind a gate that starts closed.

Ported from ImageWAM v2 (`imagewam/models/backbones/goal_pose_prior.py`); the
deviations and their reasons are recorded in `PORT_goal_prior.md`.
"""

from typing import Optional

import torch
import torch.nn as nn

# ImageWAM v2 defaults, kept verbatim so the recipe transfers.
GOAL_PRIOR_NUM_GOAL_TOKENS = 8       # Stage 1: oracle pose -> K tokens
GOAL_PRIOR_NUM_LATENTS = 100         # Stage 2: 8 pose-supervised + 92 free context
GOAL_PRIOR_NUM_POSE_TOKENS = 8
GOAL_PRIOR_INNER_DIM = 512
GOAL_PRIOR_LATENT_DIM = 768
GOAL_PRIOR_NUM_GROUPS = 5
GOAL_PRIOR_SYN_GATE_BIAS_INIT = -2.0
GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT = 0.05
GOAL_PRIOR_CONTEXT_BLACKOUT_PROB = 0.10
GOAL_PRIOR_STAGE1_NULL_TOKENS = 32

# Stage 1 owns these; Stage 2 starts from the Stage 1 weights but drops them, because
# the oracle channel and the placeholder both retire once real observations arrive.
STAGE1_ONLY_CHECKPOINT_KEYS = ("goal_pose_encoder", "stage1_null_tokens")


def extract_goal_pose(proprio: torch.Tensor) -> torch.Tensor:
    """The goal is the last observation of the sample window.

    Mirrors ImageWAM's `extract_goal_pose_from_proprio` ("Split the last observation
    as goal"). FastWAM's `build_inputs` keeps only `proprio[:, 0]` for the context, so
    the final frame is otherwise discarded.
    """
    if proprio.ndim != 3:
        raise ValueError(f"`proprio` must be [B, T, d], got {tuple(proprio.shape)}")
    return proprio[:, -1, :]


class GoalPoseEncoder(nn.Module):
    """Maps a normalized goal pose vector to K cross-attention tokens.

    Deliberately an MLP over the pose vector alone: Stage 1 must not have any path
    from pixels to the action expert.
    """

    def __init__(
        self,
        pose_dim: int,
        num_tokens: int = GOAL_PRIOR_NUM_GOAL_TOKENS,
        hidden_size: int = 4096,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.pose_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.num_tokens * self.hidden_size),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 2:
            raise ValueError(f"`pose` must be [B, pose_dim], got {tuple(pose.shape)}")
        if pose.shape[1] != self.pose_dim:
            raise ValueError(f"`pose` last dim must be {self.pose_dim}, got {pose.shape[1]}")
        return self.net(pose).view(pose.shape[0], self.num_tokens, self.hidden_size)


class GoalPoseDecoder(nn.Module):
    """Reconstructs a normalized goal pose from the pose latents (Stage 2 readout)."""

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_POSE_TOKENS,
        hidden_size: int = GOAL_PRIOR_LATENT_DIM,
        pose_dim: int = 8,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.pose_dim = int(pose_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_tokens * self.hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.pose_dim),
        )

    def forward(self, goal_hidden: torch.Tensor) -> torch.Tensor:
        if goal_hidden.ndim != 3:
            raise ValueError(f"`goal_hidden` must be [B, K, D], got {tuple(goal_hidden.shape)}")
        return self.net(goal_hidden.reshape(goal_hidden.shape[0], -1))


def sample_channel_regime(
    batch_size: int,
    device: torch.device,
    p_ref_only: float,
    p_syn_only: float,
    training: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Plan B. Per-sample (ref_keep, syn_keep) booleans, or (None, None).

    Fixed counts rather than independent coins: every rank then sees every regime in
    every batch, so the per-regime training metrics are always defined. Ranks that
    disagree about which metric keys exist deadlock the trainer's per-key all-gather.

    Because the counts are rounded per batch, the *effective* ratio depends on the
    batch size. ImageWAM ran bs=10 with 0.55/0.15/0.30, which lands on 0.50/0.20/0.30;
    FastWAM's config therefore states those effective values directly rather than
    copying the nominal ones, which at bs=16 would cut p_ref_only from 0.20 to 0.125.
    """
    if not training:
        return None, None
    n = int(batch_size)
    if n <= 0:
        return None, None
    n_ref_only = int(round(p_ref_only * n))
    n_syn_only = int(round(p_syn_only * n))
    if n >= 3:  # keep every regime represented
        n_ref_only = max(1, min(n_ref_only, n - 2))
        n_syn_only = max(1, min(n_syn_only, n - 1 - n_ref_only))
    else:
        n_ref_only = min(n_ref_only, n)
        n_syn_only = min(n_syn_only, n - n_ref_only)
    order = torch.randperm(n, device=device)
    ref_keep = torch.ones(n, dtype=torch.bool, device=device)
    syn_keep = torch.ones(n, dtype=torch.bool, device=device)
    ref_keep[order[:n_syn_only]] = False                      # syn-only samples
    syn_keep[order[n_syn_only : n_syn_only + n_ref_only]] = False  # ref-only samples
    return ref_keep, syn_keep


class _SelfAttentionBlock(nn.Module):
    """Lets the 100 latents talk to each other before they read the streams."""

    def __init__(self, latent_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        tokens = tokens + self.dropout(self.attn(h, h, h, need_weights=False)[0])
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class _CrossAttentionBlock(nn.Module):
    """Reads one stream. kdim/vdim are independent of latent_dim, so the text stream
    (T5, 4096) and the visual stream (Wan hidden, 3072) both plug in unchanged."""

    def __init__(self, latent_dim: int, context_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout,
            kdim=context_dim, vdim=context_dim, batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

    ATTN_SINK = []          # filled only while FASTWAM_ATTN_DUMP is set

    def forward(self, queries, context, key_padding_mask=None):
        import os as _os
        _dump = bool(_os.environ.get("FASTWAM_ATTN_DUMP")) and getattr(self, "_stream", None) == "visual"
        _o = self.cross_attn(
            self.query_norm(queries), self.context_norm(context), self.context_norm(context),
            key_padding_mask=key_padding_mask, need_weights=_dump,
            **({"average_attn_weights": True} if _dump else {}),
        )
        out = _o[0]
        if _dump and _o[1] is not None:
            # (B, n_latents, n_visual_tokens), heads already averaged
            _CrossAttentionBlock.ATTN_SINK.append(_o[1].detach().float().cpu())
        queries = queries + self.dropout(out)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class _AggregatorGroup(nn.Module):
    """One depth group: self-attend, then read text, then read vision."""

    def __init__(self, latent_dim, semantic_dim, visual_dim, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.self_attn = _SelfAttentionBlock(latent_dim, num_heads, ffn_ratio, dropout)
        self.semantic = _CrossAttentionBlock(latent_dim, semantic_dim, num_heads, ffn_ratio, dropout)
        self.visual = _CrossAttentionBlock(latent_dim, visual_dim, num_heads, ffn_ratio, dropout)
        self.semantic._stream = "semantic"
        self.visual._stream = "visual"      # only this one maps back onto pixels

    def forward(self, queries, semantic_hidden, visual_hidden, semantic_mask=None):
        queries = self.self_attn(queries)
        queries = self.semantic(queries, semantic_hidden, key_padding_mask=semantic_mask)
        return self.visual(queries, visual_hidden)


class SemanticVisualAggregator(nn.Module):
    """The 100 goal latents, refreshed at every backbone layer.

    Not a one-shot head: `forward_layer` is called inside the DiT loop so the latents
    read the text and visual streams *at that depth*, with parameters shared within each
    of `num_layer_groups` contiguous groups. The first `num_pose_tokens` latents carry
    the goal pose and are read out by GoalPoseDecoder; the rest are free context.
    """

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_LATENTS,
        latent_dim: int = GOAL_PRIOR_LATENT_DIM,
        semantic_dim: int = 4096,
        visual_dim: int = 3072,
        out_dim: int = 1024,
        num_layer_groups: int = GOAL_PRIOR_NUM_GROUPS,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        num_pose_tokens: int = GOAL_PRIOR_NUM_POSE_TOKENS,
        context_token_dropout: float = GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT,
        context_blackout_prob: float = GOAL_PRIOR_CONTEXT_BLACKOUT_PROB,
        gate_bias_init: float = GOAL_PRIOR_SYN_GATE_BIAS_INIT,
        gate_pose_tokens: bool = False,
    ):
        super().__init__()
        if not 1 <= num_pose_tokens <= num_tokens:
            raise ValueError(f"num_pose_tokens must be in [1, {num_tokens}], got {num_pose_tokens}")
        self.num_tokens = int(num_tokens)
        self.num_pose_tokens = int(num_pose_tokens)
        self.num_layer_groups = int(num_layer_groups)
        self.context_token_dropout = float(context_token_dropout)
        self.context_blackout_prob = float(context_blackout_prob)
        self.gate_pose_tokens = bool(gate_pose_tokens)

        self.queries = nn.Parameter(torch.empty(num_tokens, latent_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.groups = nn.ModuleList(
            _AggregatorGroup(latent_dim, semantic_dim, visual_dim, num_heads, ffn_ratio, dropout)
            for _ in range(self.num_layer_groups)
        )
        # ImageWAM's 3072->3072 needed no projection; ActionDiT's hidden is 1024.
        self.to_action = nn.Linear(latent_dim, out_dim)
        # One learnable scalar per group, suppressing the gated columns about 150x at
        # init so step 0 reproduces Stage 1's [text | pose | action] interface instead of
        # running a pose-conditioned prior with no pose.
        #
        # Stored as an offset from the init rather than the absolute value: at -2.0 the
        # bf16 grid is 0.0078 wide and an optimizer step moves ~1e-4, so every update
        # would round back and the gate could never open. Starting the learnable part at
        # zero puts it where bf16 is dense.
        self.register_buffer("syn_gate_init",
                             torch.full((self.num_layer_groups,), float(gate_bias_init)),
                             persistent=True)
        self.syn_gate_delta = nn.Parameter(torch.zeros(self.num_layer_groups))

    def layer_group_index(self, layer_idx: int, num_layers: int) -> int:
        if num_layers % self.num_layer_groups != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by num_layer_groups "
                f"({self.num_layer_groups})"
            )
        return int(layer_idx) // (num_layers // self.num_layer_groups)

    def init_queries(self, batch_size: int, device, dtype) -> torch.Tensor:
        return self.queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def forward_layer(self, queries, semantic_hidden, visual_hidden, *, layer_idx, num_layers,
                      semantic_mask=None):
        group = self.groups[self.layer_group_index(layer_idx, num_layers)]
        return group(queries, semantic_hidden, visual_hidden, semantic_mask=semantic_mask)

    def gate_for_layer(self, layer_idx: int, num_layers: int) -> torch.Tensor:
        idx = self.layer_group_index(layer_idx, num_layers)
        # Summed in fp32: adding a ~1e-4 delta to -2.0 in bf16 rounds straight back, so
        # the delta would accumulate losslessly but the gate it produces would still only
        # move in 0.0078 jumps.
        return self.syn_gate_init[idx].float() + self.syn_gate_delta[idx].float()

    @property
    def syn_gate_bias(self) -> torch.Tensor:
        """The effective gate, for logging and checkpoint inspection."""
        return self.syn_gate_init.float() + self.syn_gate_delta.float()

    def context_keep_mask(self, batch_size: int, device, training: bool) -> torch.Tensor:
        """Per-sample keep mask over the latents. Pose columns are never dropped.

        Drops whole context columns so the action expert cannot lean on a single
        aggregate, and blacks out the entire context block for a fraction of samples --
        whose fallback state is exactly Stage 1's interface.
        """
        keep = torch.ones(batch_size, self.num_tokens, dtype=torch.bool, device=device)
        if not training:
            return keep
        n_ctx = self.num_tokens - self.num_pose_tokens
        if n_ctx > 0 and self.context_token_dropout > 0:
            drop = torch.rand(batch_size, n_ctx, device=device) < self.context_token_dropout
            keep[:, self.num_pose_tokens :] &= ~drop
        if n_ctx > 0 and self.context_blackout_prob > 0:
            black = torch.rand(batch_size, device=device) < self.context_blackout_prob
            keep[black, self.num_pose_tokens :] = False
        return keep

    def gated_span(self) -> tuple[int, int]:
        """Columns the gate applies to: the context latents only, unless configured
        otherwise, so the pose columns stay open from step 0."""
        return (0, self.num_tokens) if self.gate_pose_tokens else (self.num_pose_tokens, self.num_tokens)
