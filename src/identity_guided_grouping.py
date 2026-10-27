from __future__ import annotations

import argparse

import numpy as np
import torch

from . import data
from .models import IdentityGuidedGroupingEncoder
from .transfer_context import CaseFishContext


def build_identity_guided_grouping_encoder(
    reference_ctx: CaseFishContext,
    *,
    d_model: int,
    args: argparse.Namespace,
    normalize_out: bool,
) -> IdentityGuidedGroupingEncoder:
    static_dim = 0 if reference_ctx.static_nf is None else int(reference_ctx.static_nf.shape[1])
    fish = reference_ctx.base.fish
    use_gene = bool(getattr(args, "use_gene", False))
    use_pos = bool(getattr(args, "use_pos", False))
    use_atlas = bool(getattr(args, "use_atlas", False))
    use_region = bool(getattr(args, "use_region", False))
    use_activity = bool(getattr(args, "use_activity", False))
    gene_dim = int(fish.genes_nf.shape[1]) if use_gene and fish.genes_nf is not None else None
    pos_dim = int(fish.coords_nf.shape[1]) if use_pos and fish.coords_nf is not None else None
    atlas_dim = int(fish.regions_nf.shape[1]) if use_atlas and fish.regions_nf is not None else None
    region_dim = int(fish.region_membership_nf.shape[1]) if use_region and fish.region_membership_nf is not None else None
    activity_dim = int(fish.activity_nf.shape[1]) if use_activity and fish.activity_nf is not None else None
    return IdentityGuidedGroupingEncoder(
        static_dim=int(static_dim), d_model=int(d_model), token_mlp_hidden=int(args.token_mlp_hidden), normalize_out=bool(normalize_out),
        static_has_gene=use_gene, static_has_pos=use_pos, static_has_atlas=use_atlas,
        static_has_region=use_region, static_has_activity=use_activity,
        gene_dim=gene_dim, pos_dim=pos_dim, atlas_dim=atlas_dim, region_dim=region_dim, activity_dim=activity_dim,
        window_left=int(args.window_left), window_right=int(args.window_right), internal_dim=int(args.slot_internal_dim),
        static_hidden_dim=int(args.slot_static_hidden_dim), num_slots=int(args.slot_num_slots),
        slot_mixer_hidden=int(args.slot_mixer_hidden), slot_assign_temp=float(args.slot_assign_temp),
        assign_topk=int(args.slot_assign_topk),
    )


def prepare_subset_cache(model: torch.nn.Module, *, ctx: CaseFishContext, neuron_indices: np.ndarray, device: torch.device, static_nf_override: np.ndarray | None = None) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
    n_idx = np.asarray(neuron_indices, dtype=np.int64)
    x_nt = torch.as_tensor(ctx.base.fish.neural_nt[n_idx], dtype=torch.float32, device=device)
    x_nt_ctx = model.apply_temporal_frontend(x_nt) if hasattr(model, "apply_temporal_frontend") else x_nt
    x_nt_ctx = data.precompute_windowed_time_context(model, x_nt_ctx)
    static_t = None
    static_nf = static_nf_override
    if static_nf is None and ctx.static_nf is not None:
        static_nf = np.asarray(ctx.static_nf[n_idx], dtype=np.float32)
    if static_nf is not None:
        static_t = torch.as_tensor(static_nf, dtype=torch.float32, device=device)
    static_cache = data.precompute_model_static_cache(model, static_t, dtype=torch.float32, device=device)
    return x_nt_ctx, static_t, static_cache


def build_time_batch(model: torch.nn.Module, *, x_nt_ctx: torch.Tensor | dict[str, torch.Tensor], time_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    t_idx = np.asarray(time_indices, dtype=np.int64)
    t_b = torch.as_tensor(t_idx, dtype=torch.int64, device=device)
    return data.build_time_batch_input(model, x_nt_ctx, t_b)
