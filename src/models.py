from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _make_static_encoder(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
    )


class IdentityGuidedGroupingEncoder(nn.Module):
    def __init__(
        self,
        static_dim: int,
        d_model: int,
        token_mlp_hidden: int,
        normalize_out: bool,
        static_has_gene: bool = True,
        static_has_pos: bool = True,
        static_has_atlas: bool = False,
        static_has_region: bool = False,
        static_has_activity: bool = False,
        gene_dim: int | None = None,
        pos_dim: int | None = None,
        atlas_dim: int | None = None,
        region_dim: int | None = None,
        activity_dim: int | None = None,
        window_left: int = 3,
        window_right: int = 2,
        internal_dim: int = 128,
        static_hidden_dim: int = 64,
        num_slots: int = 256,
        slot_mixer_hidden: int = 64,
        slot_assign_temp: float = 0.35,
        assign_topk: int = 5,
    ):
        super().__init__()
        if int(window_left) < 0 or int(window_right) < 0:
            raise ValueError(f"window_left/right must be >=0, got ({window_left},{window_right})")
        if int(internal_dim) < 1:
            raise ValueError(f"internal_dim must be >=1, got {internal_dim}")
        if int(static_hidden_dim) < 1:
            raise ValueError(f"static_hidden_dim must be >=1, got {static_hidden_dim}")
        if int(num_slots) < 2:
            raise ValueError(f"num_slots must be >=2, got {num_slots}")
        if int(slot_mixer_hidden) < 1:
            raise ValueError(f"slot_mixer_hidden must be >=1, got {slot_mixer_hidden}")
        if float(slot_assign_temp) <= 0.0:
            raise ValueError(f"slot_assign_temp must be >0, got {slot_assign_temp}")

        self.static_dim = int(static_dim)
        self.static_has_gene = bool(static_has_gene)
        self.static_has_pos = bool(static_has_pos)
        self.static_has_atlas = bool(static_has_atlas)
        self.static_has_region = bool(static_has_region)
        self.static_has_activity = bool(static_has_activity)
        self.normalize_out = bool(normalize_out)
        self.window_left = int(window_left)
        self.window_right = int(window_right)
        self.window_size = self.window_left + self.window_right + 1
        self.internal_dim = int(internal_dim)
        self.static_hidden_dim = int(static_hidden_dim)
        self.num_slots = int(num_slots)
        self.slot_mixer_hidden = int(slot_mixer_hidden)
        self.slot_assign_temp = float(slot_assign_temp)
        self.assign_topk = int(assign_topk)
        if self.assign_topk < 0:
            raise ValueError(f"assign_topk must be >=0, got {assign_topk}")
        if self.assign_topk > 0 and self.assign_topk > self.num_slots:
            raise ValueError(f"assign_topk must be <= num_slots, got {assign_topk}>{self.num_slots}")

        self.pos_dim = int((3 if self.static_has_pos else 0) if pos_dim is None else pos_dim)
        if not self.static_has_pos:
            self.pos_dim = 0
        self.gene_dim = 0
        self.atlas_dim = 0
        self.region_dim = 0
        self.activity_dim = 0
        specified_dims = int(self.pos_dim)
        pending_dims: list[str] = []
        for name, enabled, dim in (
            ("gene", self.static_has_gene, gene_dim),
            ("atlas", self.static_has_atlas, atlas_dim),
            ("region", self.static_has_region, region_dim),
            ("activity", self.static_has_activity, activity_dim),
        ):
            if not enabled:
                continue
            if dim is None:
                pending_dims.append(name)
            else:
                value = int(dim)
                setattr(self, f"{name}_dim", value)
                specified_dims += value
        remaining = int(self.static_dim - specified_dims)
        if remaining < 0:
            raise ValueError(f"static modality dims exceed static_dim={self.static_dim}")
        if len(pending_dims) > 1:
            raise ValueError("Enabled non-position static modalities must provide explicit dimensions.")
        if len(pending_dims) == 1:
            setattr(self, f"{pending_dims[0]}_dim", int(remaining))
        elif self.static_dim > 0 and remaining != 0:
            raise ValueError(f"Static modality dims must sum to static_dim={self.static_dim}, leftover={remaining}")

        total_static = int(self.gene_dim + self.pos_dim + self.atlas_dim + self.region_dim + self.activity_dim)
        if self.static_dim > 0 and total_static != self.static_dim:
            raise ValueError(f"Static modality dims must sum to static_dim={self.static_dim}, got {total_static}")

        t_h = int(token_mlp_hidden)
        d = int(d_model)
        r = int(self.internal_dim)
        s_h = int(self.static_hidden_dim)
        k = int(self.num_slots)
        h_slot = int(self.slot_mixer_hidden)

        self.temporal_value = nn.Sequential(
            nn.Linear(int(self.window_size), t_h),
            nn.GELU(),
            nn.Linear(t_h, r),
        )

        self.gene_encoder = _make_static_encoder(self.gene_dim, s_h) if self.gene_dim > 0 else None
        self.pos_encoder = _make_static_encoder(self.pos_dim, s_h) if self.pos_dim > 0 else None
        self.atlas_encoder = _make_static_encoder(self.atlas_dim, s_h) if self.atlas_dim > 0 else None
        self.region_encoder = _make_static_encoder(self.region_dim, s_h) if self.region_dim > 0 else None
        self.activity_encoder = _make_static_encoder(self.activity_dim, s_h) if self.activity_dim > 0 else None

        self.gene_norm = nn.LayerNorm(s_h) if self.gene_encoder is not None else None
        self.pos_norm = nn.LayerNorm(s_h) if self.pos_encoder is not None else None
        self.atlas_norm = nn.LayerNorm(s_h) if self.atlas_encoder is not None else None
        self.region_norm = nn.LayerNorm(s_h) if self.region_encoder is not None else None
        self.activity_norm = nn.LayerNorm(s_h) if self.activity_encoder is not None else None
        self.num_static_modalities = sum(
            int(module is not None)
            for module in (self.gene_encoder, self.pos_encoder, self.atlas_encoder, self.region_encoder, self.activity_encoder)
        )
        if self.num_static_modalities > 0:
            self.static_fuse = nn.Sequential(
                nn.Linear(max(1, self.num_static_modalities) * s_h, s_h),
                nn.GELU(),
                nn.Linear(s_h, s_h),
                nn.GELU(),
            )
        else:
            self.static_fuse = None
            self.register_parameter("null_static", nn.Parameter(torch.zeros(s_h)))

        self.assign_logits = nn.Linear(s_h, k)
        self.slot_to_hidden = nn.Linear(k, h_slot)
        self.slot_axis_mlp = nn.Sequential(
            nn.Linear(k, h_slot),
            nn.GELU(),
            nn.Linear(h_slot, h_slot),
            nn.GELU(),
        )
        self.readout_backbone = nn.Sequential(
            nn.Linear((r * h_slot) + h_slot, t_h),
            nn.GELU(),
            nn.Linear(t_h, t_h),
            nn.GELU(),
        )
        self.readout = nn.Linear(t_h, d)

    def apply_temporal_frontend(self, x_nt: torch.Tensor) -> torch.Tensor:
        return x_nt

    def _encode_static_base(
        self,
        *,
        n: int,
        dtype: torch.dtype,
        device: torch.device,
        static_nf: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.static_fuse is None:
            return self.null_static.to(device=device, dtype=dtype).expand(n, -1)
        if static_nf is None:
            s = torch.zeros((n, self.static_dim), dtype=dtype, device=device)
        else:
            s = static_nf.to(device=device, dtype=dtype)
        parts = []
        offset = 0
        if self.gene_encoder is not None:
            part = self.gene_norm(self.gene_encoder(s[:, offset:offset + self.gene_dim]))
            parts.append(part)
            offset += self.gene_dim
        if self.pos_encoder is not None:
            part = self.pos_norm(self.pos_encoder(s[:, offset:offset + self.pos_dim]))
            parts.append(part)
            offset += self.pos_dim
        if self.atlas_encoder is not None:
            part = self.atlas_norm(self.atlas_encoder(s[:, offset:offset + self.atlas_dim]))
            parts.append(part)
            offset += self.atlas_dim
        if self.region_encoder is not None:
            part = self.region_norm(self.region_encoder(s[:, offset:offset + self.region_dim]))
            parts.append(part)
            offset += self.region_dim
        if self.activity_encoder is not None:
            part = self.activity_norm(self.activity_encoder(s[:, offset:offset + self.activity_dim]))
            parts.append(part)
        return self.static_fuse(torch.cat(parts, dim=-1))

    def _apply_assign_topk(self, logits: torch.Tensor) -> torch.Tensor:
        if self.assign_topk <= 0 or self.assign_topk >= logits.shape[-1]:
            return logits
        topk_min = torch.topk(logits, k=self.assign_topk, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < topk_min, -1e9)

    def slot_assignments(self, static_nf: torch.Tensor | None) -> torch.Tensor:
        cache = self.precompute_static_cache(
            static_nf,
            dtype=self.readout.weight.dtype,
            device=self.readout.weight.device,
        )
        return cache["assign"]

    def precompute_static_cache(
        self,
        static_nf: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        n = int(static_nf.shape[0]) if static_nf is not None else 1
        s_base = self._encode_static_base(n=n, dtype=dtype, device=device, static_nf=static_nf)
        logits = self.assign_logits(s_base) / self.slot_assign_temp
        logits = self._apply_assign_topk(logits)
        return {"s_base": s_base, "assign": torch.softmax(logits, dim=-1)}

    def forward(
        self,
        x_btn: torch.Tensor | dict[str, torch.Tensor],
        static_nf: torch.Tensor | None,
        static_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        raw_btn = x_btn["raw"] if isinstance(x_btn, dict) else x_btn
        if raw_btn.ndim != 3:
            raise ValueError(f"IdentityGuidedGrouping expects 3D input (B,N,W), got shape={tuple(raw_btn.shape)}")

        values_bnr = self.temporal_value(raw_btn)
        if static_cache is None:
            static_cache = self.precompute_static_cache(static_nf, dtype=raw_btn.dtype, device=raw_btn.device)
        assign_nk = static_cache["assign"]

        slot_tokens = torch.einsum("nk,bnr->bkr", assign_nk, values_bnr)
        slot_mass = assign_nk.sum(dim=0).view(1, -1, 1).clamp_min(1e-6)
        slot_tokens = slot_tokens / torch.sqrt(slot_mass)

        mixed_linear = F.gelu(self.slot_to_hidden(slot_tokens.transpose(1, 2))).reshape(slot_tokens.shape[0], -1)
        mixed_slot = self.slot_axis_mlp(slot_tokens.transpose(1, 2)).mean(dim=1)
        hidden = self.readout_backbone(torch.cat([mixed_slot, mixed_linear], dim=-1))
        z = self.readout(hidden)
        if self.normalize_out:
            z = F.normalize(z, dim=-1)
        return z

    def forward_with_aux(
        self,
        x_btn: torch.Tensor | dict[str, torch.Tensor],
        static_nf: torch.Tensor | None,
        static_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward(x_btn, static_nf, static_cache=static_cache), {}
