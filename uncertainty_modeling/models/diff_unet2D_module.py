import math
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions.kl import kl_divergence

from .diffusion import ContinuousGaussianDiffusion

# MC dropout wrappers: always apply dropout with training=True so MC Dropout works in eval()
class MC_Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = float(p) if p is not None else 0.0
        self.inplace = bool(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


class MC_Dropout2d(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p) if p is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        return F.dropout2d(x, p=self.p, training=True)

def timestep_embedding(timesteps, dim, max_period=10):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class GroupNorm32(nn.GroupNorm):
    """GroupNorm that normalizes in float32 but allows small channel counts.

    Uses ``num_groups = min(32, num_channels)`` so it works for channels < 32.
    """

    def __init__(self, num_channels: int):
        if num_channels % 32 == 0:
            num_groups = 32
        else:
            num_groups = num_channels
        super().__init__(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)
    
class DiffUnet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention__init__
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        image_size=32,
        out_channels=1,
        in_channels=4,
        model_channels=32,
        num_res_blocks=[1,2,3,4],
        num_middle_res_blocks=4,
        attention_resolutions=[-1],
        dropout=0,
        channel_mult=(1, 1, 2, 4),
        conv_resample=True,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        diffusion: bool = False,
        final_act="none",
        one_skip_per_reso=False,
        new_upsample_method=False,
        mlp_attn=False,
        act="silu",
        name="unet",
        ssn=False,
        ssn_rank=10,
        ssn_eps=1e-5,
        diffusion_kwargs=None,
        diffusion_num_steps: int = 50,
        diffusion_sampler_type: str = "ddpm",
        swag_enabled: bool = False,
        encoder_only: bool = False,
    ):
        super().__init__()
        if isinstance(act, str):
            assert act.lower() in ["silu","relu","gelu"], f"Unsupported activation function: {act}"
            act_dict = {"silu": nn.SiLU,
                        "relu": nn.ReLU,
                        "gelu": nn.GELU}
            act = act_dict[act.lower()]
        self.name = name
        self.act = act
        self.mlp_attn = mlp_attn
        self.ssn = bool(ssn)
        self.ssn_rank = ssn_rank
        self.ssn_eps = ssn_eps
        self.swag_enabled = bool(swag_enabled)
        self.encoder_only = bool(encoder_only)
        self.new_upsample_method = new_upsample_method
        self.one_skip_per_reso = one_skip_per_reso
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        time_embed_dim = model_channels*4
        if isinstance(num_res_blocks,int):
            num_res_blocks = [num_res_blocks]*len(channel_mult)
        assert len(num_res_blocks) == len(channel_mult), f"len(num_res_blocks): {len(num_res_blocks)} must be equal to len(channel_mult): {len(channel_mult)}"

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        self.attention_resolutions = []
        for ar in attention_resolutions:
            if ar < 0:
                ar = len(channel_mult) + ar
            self.attention_resolutions.append(ar)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample

        self.class_dict = {}

        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # if diffusion is False, we will ignore timestep embeddings and
        # avoid creating unused parameters where possible.
        self.diffusion = diffusion
        self.fp16_attrs = ["input_blocks", "output_blocks"]
        if num_middle_res_blocks >= 1:
            self.fp16_attrs.append("middle_block")

        # Only create timestep embedding network when diffusion is enabled.
        if self.diffusion:
            assert not ssn, "SSN with diffusion is not supported."
            self.fp16_attrs.append("time_embed")
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                self.act(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            diffusion_kwargs = diffusion_kwargs or {}
            self.diffusion_process = ContinuousGaussianDiffusion(**diffusion_kwargs)
        else:
            self.time_embed = None
            self.diffusion_process = None
        self.diffusion_num_steps = int(diffusion_num_steps) if diffusion_num_steps is not None else 50
        self.diffusion_sampler_type = diffusion_sampler_type or "ddpm"
        self.in_channels = in_channels

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.input_skip = [False]
        input_block_chans = [model_channels]
        ch = model_channels
        # 2D-only implementation for now: no generic dims handling.
        res_block_kwargs = {
            "emb_channels": time_embed_dim,
            "dropout": dropout,
            "use_scale_shift_norm": use_scale_shift_norm,
            "act": act,
        }
        attn_kwargs = {
            "num_heads": num_heads,
            "with_xattn": False,
            "xattn_channels": None,
        }
        resolution = 0
        
        assert channel_mult[0] == 1, "channel_mult[0] must be 1"
        for level, (mult, n_res_blocks) in enumerate(
            zip(channel_mult, num_res_blocks)
        ):
            for _ in range(n_res_blocks):
                if self.new_upsample_method:
                    ch = mult * model_channels
                    ch_in = ch
                else:
                    ch_in = ch
                    ch = mult * model_channels
                layers = []
                if resolution in self.attention_resolutions:
                    if self.mlp_attn:
                        layers = [
                            MLPBlock(ch, **res_block_kwargs),
                            AttentionBlock(ch, **attn_kwargs),
                        ]
                    else:
                        layers = [
                            ResBlock(ch_in, out_channels=ch, **res_block_kwargs),
                            AttentionBlock(ch, **attn_kwargs),
                        ]
                else:
                    layers = [ResBlock(ch_in, out_channels=ch, **res_block_kwargs)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_skip.append(False)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                resolution += 1
                ch_out = (
                    channel_mult[resolution] * model_channels
                    if self.new_upsample_method
                    else None
                )
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, channels_out=ch_out)
                    )
                )
                self.input_skip[-1] = True
                self.input_skip.append(False)
                input_block_chans.append(ch)
        if resolution in self.attention_resolutions:
            if self.mlp_attn:
                middle_layers = (
                    sum(
                        [
                            [
                                MLPBlock(ch, **res_block_kwargs),
                                AttentionBlock(ch, **attn_kwargs),
                            ]
                            for _ in range(num_middle_res_blocks - 1)
                        ],
                        [],
                    )
                    + [MLPBlock(ch, **res_block_kwargs)]
                )
            else:
                middle_layers = (
                    sum(
                        [
                            [
                                ResBlock(ch, **res_block_kwargs),
                                AttentionBlock(ch, **attn_kwargs),
                            ]
                            for _ in range(num_middle_res_blocks - 1)
                        ],
                        [],
                    )
                    + [ResBlock(ch, **res_block_kwargs)]
                )
        else:
            middle_layers = [
                ResBlock(ch, **res_block_kwargs) for _ in range(num_middle_res_blocks)
            ]

        self.middle_block = TimestepEmbedSequential(*middle_layers)

        if not self.encoder_only:
            attn_kwargs["num_heads"] = num_heads_upsample
            self.output_blocks = nn.ModuleList([])
            for level, mult, n_res_blocks in zip(
                reversed(list(range(len(channel_mult)))),
                channel_mult[::-1],
                num_res_blocks[::-1],
            ):
                for i in range(n_res_blocks + 1):
                    if self.new_upsample_method:
                        ch = model_channels * mult
                        ch_in = ch
                    else:
                        ch_in = ch + input_block_chans.pop()
                        ch = model_channels * mult
                    if resolution in self.attention_resolutions:
                        if self.mlp_attn:
                            layers = [
                                MLPBlock(ch, **res_block_kwargs),
                                AttentionBlock(ch, **attn_kwargs),
                            ]
                        else:
                            layers = [
                                ResBlock(ch_in, out_channels=ch, **res_block_kwargs),
                                AttentionBlock(ch, **attn_kwargs),
                            ]
                    else:
                        layers = [ResBlock(ch_in, out_channels=ch, **res_block_kwargs)]
                    if level and i == n_res_blocks:
                        resolution -= 1
                        ch_out = (
                            channel_mult[resolution] * model_channels
                            if self.new_upsample_method
                            else None
                        )
                        layers.append(
                            Upsample(
                                ch,
                                conv_resample,
                                channels_out=ch_out,
                                mode="bilinear" if self.new_upsample_method else "nearest",
                            )
                        )

                    self.output_blocks.append(TimestepEmbedSequential(*layers))

            if self.one_skip_per_reso:
                assert self.new_upsample_method, "one_skip_per_reso only works with new_upsample_method"
            else:
                self.input_skip = [True for _ in self.input_skip]
            assert final_act.lower() in [
                "none",
                "softmax",
                "tanh",
                "sigmoid",
            ], f"Unsupported final activation: {final_act}"
            final_act_dict = {
                "none": nn.Identity(),
                "softmax": nn.Softmax(dim=1),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
            }
            self.out = nn.Sequential(
                nn.Identity(),  # unnecessary, but kept for key consistency
                GroupNorm32(ch),
                self.act(),
                zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
                final_act_dict[final_act.lower()]
            )
            if self.ssn:
                if self.ssn_rank <= 0:
                    raise ValueError("ssn_rank must be positive when ssn=True")

                def _make_ssn_head(out_ch: int) -> nn.Sequential:
                    return nn.Sequential(
                        nn.Identity(),
                        GroupNorm32(ch),
                        self.act(),
                        zero_module(nn.Conv2d(ch, out_ch, 3, padding=1)),
                    )

                self.ssn_cov_head = _make_ssn_head(out_channels)
                self.ssn_factor_head = _make_ssn_head(out_channels * self.ssn_rank)
        else:
            if self.one_skip_per_reso:
                raise ValueError("encoder_only=True is incompatible with one_skip_per_reso")
            self.output_blocks = nn.ModuleList()
            self.out = None
            if self.ssn:
                raise ValueError("SSN head is not supported when encoder_only=True")
        self.out_channels = out_channels
        # expose common attributes expected elsewhere in the repo
        self.num_classes = out_channels

    def _prepare_time_embedding(
        self, x: torch.Tensor, timesteps: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.diffusion:
            if timesteps is None:
                raise ValueError("timesteps must be provided when diffusion=True")
            if timesteps.numel() == 1:
                timesteps = timesteps.expand(x.shape[0])
            return self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if timesteps is not None:
            raise ValueError("timesteps must be None when diffusion=False")
        return None

    def _forward_backbone(
        self, x: torch.Tensor, timesteps: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb = self._prepare_time_embedding(x, timesteps)
        h = x
        hs = []
        for module, skip in zip(self.input_blocks, self.input_skip):
            h = module(h, emb)
            if skip:
                hs.append(h)
            else:
                hs.append(0)
        h = self.middle_block(h, emb)
        if self.encoder_only:
            return h, emb
        for module in self.output_blocks:
            if self.new_upsample_method:
                cat_in = h + hs.pop()
            else:
                cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return h, emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        mean_only: bool = False,
    ) -> torch.Tensor | tuple[td.LowRankMultivariateNormal, bool]:
        """Apply the model to an input batch."""
        h, _ = self._forward_backbone(x, timesteps)
        h = h.type(x.dtype)
        if self.encoder_only:
            return h
        if not self.ssn:
            return self.out(h)

        mean_logits = self.out(h)
        distribution, cov_failed_flag = self._build_ssn_distribution(
            features=h, mean_logits=mean_logits, mean_only=mean_only
        )
        return distribution, cov_failed_flag

    def forward_features(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the network and return the decoder features before the final head."""
        h, _ = self._forward_backbone(x, timesteps)
        return h.type(x.dtype)

    def _build_ssn_distribution(
        self,
        features: torch.Tensor,
        mean_logits: torch.Tensor,
        mean_only: bool,
    ) -> tuple[td.LowRankMultivariateNormal, bool]:
        batch_size = mean_logits.shape[0]
        spatial_dims = mean_logits.shape[2:]
        mean = mean_logits.view(batch_size, -1)

        cov_logits = self.ssn_cov_head(features)
        cov_diag = F.softplus(cov_logits) + self.ssn_eps
        cov_diag = torch.nan_to_num(
            cov_diag, nan=1.0, posinf=1e6, neginf=self.ssn_eps
        ).clamp(min=self.ssn_eps)
        cov_diag = cov_diag.view(batch_size, -1)

        if mean_only:
            cov_factor = torch.zeros(
                (batch_size, mean.shape[1], self.ssn_rank),
                device=mean_logits.device,
                dtype=mean_logits.dtype,
            )
        else:
            cov_factor = self.ssn_factor_head(features)
            cov_factor = cov_factor.view(
                batch_size, self.ssn_rank, self.num_classes, *spatial_dims
            )
            cov_factor = cov_factor.view(batch_size, self.ssn_rank, -1)
            cov_factor = cov_factor.transpose(1, 2)

        try:
            distribution = td.LowRankMultivariateNormal(
                loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
            )
            cov_failed_flag = False
        except Exception:
            cov_failed_flag = True
            safe_diag = torch.nan_to_num(
                cov_diag, nan=1.0, posinf=1e6, neginf=self.ssn_eps
            ).clamp(min=self.ssn_eps)
            scale = torch.sqrt(safe_diag).clamp(min=self.ssn_eps)
            distribution = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return distribution, cov_failed_flag

    def diffusion_train_loss_step(
        self,
        x,
        im,
        loss_mask=None,
        eps=None,
        t=None,
        self_cond=False,
    ):
        """Proxy ContinuousGaussianDiffusion.train_loss_step so callers can use this module directly."""
        if self.diffusion_process is None:
            raise RuntimeError("diffusion_train_loss_step called but diffusion is disabled for this model")
        return self.diffusion_process.train_loss_step(
            model=self,
            x=x,
            im=im,
            loss_mask=loss_mask,
            eps=eps,
            t=t,
            self_cond=self_cond,
        )

    def diffusion_sample_loop(
        self,
        x_init,
        im,
        num_steps,
        sampler_type="ddpm",
        clip_x=False,
        guidance_weight=0.0,
        progress_bar=False,
        self_cond=False,
    ):
        """Proxy ContinuousGaussianDiffusion.sample_loop for downstream sampling utilities."""
        if self.diffusion_process is None:
            raise RuntimeError("diffusion_sample_loop called but diffusion is disabled for this model")
        return self.diffusion_process.sample_loop(
            model=self,
            x_init=x_init,
            im=im,
            num_steps=num_steps,
            sampler_type=sampler_type,
            clip_x=clip_x,
            guidance_weight=guidance_weight,
            progress_bar=progress_bar,
            self_cond=self_cond,
        )

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential container that optionally passes timestep embeddings.

    When ``emb`` is ``None``, any :class:`TimestepBlock` layers are called with
    their default (unconditioned) behavior.
    """

    def forward(self, x, emb: torch.Tensor | None = None, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x, y=x_attn)
            elif isinstance(layer, TimestepBlock):
                # Allow ``emb`` to be ``None`` (no conditioning).
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class MLPBlock(TimestepBlock):
    """
    Based on the MLP block from SiD (simple diffusion) pseudo code.

    def mlp_block(x, emb, expansion_factor=4):
    B, HW, C = x.shape
    x = Normalize(x)
    mlp_h = Dense(x, expansion_factor * C)
    scale = DenseGeneral(emb, mlp_h.shape [2:])
    shift = DenseGeneral(emb, mlp_h.shape [2:])
    mlp_h = swish(mlp_h)
    mlp_h = mlp_h * (1. + scale [:, None ]) + shift [:, None]
    if config.transformer_dropout > 0.:
        mlp_h = Dropout(mlp_h, config.transformer_dropout)
    out = Dense(mlp_h, C, kernel_init = zeros)
    return out"""
    def __init__(
        self,
        channels,
        emb_channels,
        expansion_factor=4,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
        act=nn.SiLU
    ):
        super().__init__()
        self.act = act
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        c = expansion_factor * channels
        self.in_layers = nn.Sequential(
            GroupNorm32(channels),
            nn.Conv2d(channels, c, 1),
            self.act(),
        )
        self.emb_layers = nn.Linear(
                emb_channels,
                2 * c if use_scale_shift_norm else c,
            )
        self.out_layers = nn.Sequential(
            MC_Dropout2d(p=dropout),
            nn.Conv2d(c, self.out_channels, 1),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb: torch.Tensor | None = None):
        h = self.in_layers(x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = h * (1 + scale) + shift
            else:
                h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, with_xattn=False, xattn_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.with_xattn = with_xattn
        if self.with_xattn:
            if xattn_channels is None:
                xattn_channels = channels
            self.xattn_channels = xattn_channels
            self.qk_x = nn.Conv1d(xattn_channels, 2*channels, 1) 
            self.v_x = nn.Conv1d(channels, channels, 1)
        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, y=None):
        b, c, *spatial = x.shape
        #assert c==self.channels, f"expected {self.channels} channels, got {c} channels"
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        if y is not None:
            assert self.with_xattn, "y is is only supported as an input for AttentionBlocks with cross attention"
            b, cx, *spatial2 = y.shape
            assert cx==self.xattn_channels, f"expected {self.xattn_channels} channels, got {cx} channels"
            y = y.reshape(b, cx, -1)
            qk = self.qk_x(self.norm(y))
            v = self.v_x(self.norm(h))
            qkv_x = torch.cat([qk,v],dim=-1).reshape(b * self.num_heads, -1, qk.shape[2])
            h = self.attention(qkv_x)+h
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x,y=x_attn)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims: int = 2, mode: str = "nearest"):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.mode = mode
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        if channels_out != channels:
            self.channel_mapper = nn.Conv2d(channels, channels_out, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims: int = 2):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(stride)
        if channels_out != channels:
            self.channel_mapper = nn.Conv2d(channels, channels_out, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.op(x)
        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        return x

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        act=nn.SiLU
    ):
        super().__init__()
        self.act = act
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(channels),
            self.act(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            self.act(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(self.out_channels),
            self.act(),
            MC_Dropout2d(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb: torch.Tensor | None = None):
        h = self.in_layers(x)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            # No timestep conditioning; just apply the output layers.
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ProbUnetFcomb(nn.Module):
    """1x1 convolutional combiner that fuses UNet features with latent samples."""

    def __init__(
        self,
        feature_channels: int,
        latent_dim: int,
        num_classes: int,
        hidden_channels: int | None = None,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive for ProbUnetFcomb")
        hidden_channels = hidden_channels or feature_channels
        in_channels = feature_channels + latent_dim
        body_layers: list[nn.Module] = []
        for _ in range(max(0, num_layers - 1)):
            body_layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
            body_layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        self.body = nn.Sequential(*body_layers) if body_layers else None
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feature_map: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        latent = z.unsqueeze(-1).unsqueeze(-1)
        latent = latent.expand(-1, -1, feature_map.shape[2], feature_map.shape[3])
        fused = torch.cat([feature_map, latent], dim=1)
        if self.body is not None:
            fused = self.body(fused)
        return self.head(fused)


class ProbUnetLatentEncoder(nn.Module):
    """Wraps an encoder-only DiffUnet to parameterize an axis-aligned Gaussian."""

    def __init__(self, encoder: DiffUnet, latent_dim: int) -> None:
        super().__init__()
        if not getattr(encoder, "encoder_only", False):
            raise ValueError("ProbUnetLatentEncoder requires encoder_only DiffUnet instance")
        self.encoder = encoder
        self.latent_dim = int(latent_dim)
        encoder_channels = int(encoder.channel_mult[-1] * encoder.model_channels)
        self.param_head = nn.Conv2d(encoder_channels, 2 * self.latent_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.param_head.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.param_head.bias, mean=0.0, std=1e-2)

    def forward(self, x: torch.Tensor) -> td.Independent:
        encoding = self.encoder.forward_features(x)
        pooled = encoding.mean(dim=(2, 3), keepdim=True)
        params = self.param_head(pooled).squeeze(-1).squeeze(-1)
        mu, log_sigma = torch.split(params, self.latent_dim, dim=1)
        scale = torch.exp(log_sigma)
        return td.Independent(td.Normal(loc=mu, scale=scale), 1)


class ProbabilisticUnetModel(nn.Module):
    """Probabilistic UNet wrapper built on top of DiffUnet backbones."""

    def __init__(
        self,
        base_unet: DiffUnet,
        prior_encoder: ProbUnetLatentEncoder,
        posterior_encoder: ProbUnetLatentEncoder,
        fcomb: ProbUnetFcomb,
        latent_dim: int,
        beta: float,
        regularizer_coeff: float,
        beta_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.unet = base_unet
        self.prior_encoder = prior_encoder
        self.posterior_encoder = posterior_encoder
        self.fcomb = fcomb
        self.latent_dim = int(latent_dim)
        self._target_beta = float(beta)
        self.beta = float(beta)
        self.beta_warmup_epochs = max(0, int(beta_warmup_epochs))
        self.regularizer_scale = float(regularizer_coeff)
        self.prior_latent_space: td.Distribution | None = None
        self.posterior_latent_space: td.Distribution | None = None
        self._feature_map: torch.Tensor | None = None
        self.prob_unet = True
        self.diffusion = False
        self.ssn = False
        self.swag_enabled = getattr(base_unet, "swag_enabled", False)
        self.num_classes = base_unet.num_classes
        self.out_channels = base_unet.out_channels
        self.diffusion_num_steps = getattr(base_unet, "diffusion_num_steps", 0)
        self.diffusion_sampler_type = getattr(base_unet, "diffusion_sampler_type", "ddpm") or "ddpm"

    def forward(
        self,
        patch: torch.Tensor,
        segm: torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        if training and segm is None:
            raise ValueError("Posterior segmentation mask is required during training")
        self._feature_map = self.unet.forward_features(patch)
        self.prior_latent_space = self.prior_encoder(patch)
        if training:
            assert segm is not None
            if segm.shape[0] != patch.shape[0]:
                raise ValueError("Segmentation mask batch size must match inputs")
            if segm.shape[2:] != patch.shape[2:]:
                raise ValueError("Segmentation mask spatial shape must match inputs")
            posterior_input = torch.cat([patch, segm], dim=1)
            self.posterior_latent_space = self.posterior_encoder(posterior_input)
        else:
            self.posterior_latent_space = None
        return self._feature_map

    def sample(self, from_prior: bool = True, testing: bool = False) -> torch.Tensor:
        if self._feature_map is None:
            raise RuntimeError("Call forward before sampling from the Probabilistic UNet")
        latent_dist = self.prior_latent_space if from_prior else self.posterior_latent_space
        if latent_dist is None:
            raise RuntimeError("Latent distribution is not available for sampling")
        z = latent_dist.sample() if testing else latent_dist.rsample()
        return self.fcomb(self._feature_map, z)

    def sample_multiple(
        self,
        num_samples: int,
        from_prior: bool = True,
        testing: bool = True,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        samples = [self.sample(from_prior=from_prior, testing=testing) for _ in range(num_samples)]
        return torch.stack(samples, dim=0)

    def apply_beta_warmup(self, epoch: int) -> float:
        """Update beta based on a linear warmup schedule."""
        if self.beta_warmup_epochs <= 0:
            self.beta = self._target_beta
            return self.beta
        progress = float(epoch + 1) / float(self.beta_warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        self.beta = self._target_beta * progress
        return self.beta

    def elbo(
        self,
        target: torch.Tensor,
        ignore_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._feature_map is None or self.posterior_latent_space is None:
            raise RuntimeError("Call forward with training=True before computing the ELBO")
        if self.prior_latent_space is None:
            raise RuntimeError("Prior distribution is not initialized")
        z_post = self.posterior_latent_space.rsample()
        reconstruction_logits = self.fcomb(self._feature_map, z_post)
        ce_kwargs = {}
        if isinstance(ignore_index, int) and ignore_index >= 0:
            ce_kwargs["ignore_index"] = ignore_index
        recon_loss = F.cross_entropy(
            reconstruction_logits,
            target,
            reduction="mean",
            **ce_kwargs,
        )
        self.mean_reconstruction_loss = recon_loss
        kl = kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        kl = torch.mean(kl)
        self.kl = kl
        elbo = -(recon_loss + self.beta * kl)
        return elbo, recon_loss, kl

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for module in (self.posterior_encoder, self.prior_encoder, self.fcomb):
            module_sum = torch.zeros((), device=device)
            for param in module.parameters():
                module_sum = module_sum + torch.sum(param ** 2)
            total = total + module_sum
        return total

    def get_pred(
        self,
        patch: torch.Tensor,
        num_samples: int = 1,
        apply_softmax: bool = True,
    ) -> torch.Tensor:
        self.forward(patch, segm=None, training=False)
        logits = self.sample_multiple(num_samples=num_samples, from_prior=True, testing=True)
        if apply_softmax:
            return torch.softmax(logits, dim=2)
        return logits

def _normalize_sampling_config(config: Any) -> dict[str, Any]:
    """Return a lower-cased plain dict for diffusion sampling overrides."""
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError(
            f"diffusion_sampling must be a mapping when provided, got {type(config)!r}."
        )
    normalized = {}
    for key, value in config.items():
        normalized[str(key).lower()] = value
    return normalized


def _scale_channel_multipliers(channel_mult: list[int], scale: float | None) -> list[int]:
    if scale is None or abs(scale - 1.0) < 1e-6:
        return list(channel_mult)
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("Channel multiplier scaling factors must be > 0.0")
    scaled: list[int] = []
    for idx, value in enumerate(channel_mult):
        scaled_value = max(1, int(round(value * factor)))
        if idx == 0:
            scaled_value = 1
        scaled.append(scaled_value)
    return scaled


def _build_prob_unet_model(
    cfg_dict: dict[str, Any],
    prob_cfg_raw: Mapping[str, Any],
    diffusion_kwargs: dict[str, Any] | None,
    sanitized_kwargs: dict[str, Any],
) -> ProbabilisticUnetModel:
    if cfg_dict.get("diffusion", False):
        raise ValueError("Probabilistic UNet does not support diffusion training")
    base_cfg = deepcopy(cfg_dict)
    base_channel_mult = list(base_cfg.get("channel_mult", []))
    if not base_channel_mult:
        raise ValueError("CHANNEL_MULT must be specified to instantiate the Probabilistic UNet")
    prob_cfg = {str(k).lower(): v for k, v in prob_cfg_raw.items()}
    latent_dim = int(prob_cfg.get("latent_dim", 6))
    beta = float(prob_cfg.get("beta", 10.0))
    beta_warmup_epochs = int(prob_cfg.get("beta_warmup_epochs", 0))
    reg_coeff = float(prob_cfg.get("regularizer_coeff", 1e-5))
    num_fcomb = int(prob_cfg.get("num_fcomb_convs", 4))
    unet_scale = float(prob_cfg.get("unet_channel_mult", 0.75))
    prior_scale = float(prob_cfg.get("prior_channel_mult", 0.5))
    posterior_scale = float(prob_cfg.get("posterior_channel_mult", 0.5))

    def _prepare_cfg(scale: float, in_channels: int | None = None, encoder_only: bool = False) -> dict[str, Any]:
        cfg = deepcopy(base_cfg)
        cfg["channel_mult"] = _scale_channel_multipliers(base_channel_mult, scale)
        cfg["diffusion"] = False
        cfg["ssn"] = False
        cfg["encoder_only"] = encoder_only
        if in_channels is not None:
            cfg["in_channels"] = in_channels
        return cfg

    in_channels = int(base_cfg["in_channels"])
    out_channels = int(base_cfg["out_channels"])
    unet_cfg = _prepare_cfg(unet_scale, encoder_only=False)
    prior_cfg = _prepare_cfg(prior_scale, encoder_only=True)
    posterior_cfg = _prepare_cfg(
        posterior_scale,
        in_channels=in_channels + out_channels,
        encoder_only=True,
    )

    base_unet = DiffUnet(**unet_cfg, diffusion_kwargs=diffusion_kwargs, **sanitized_kwargs)
    prior_unet = DiffUnet(**prior_cfg, diffusion_kwargs=diffusion_kwargs, **sanitized_kwargs)
    posterior_unet = DiffUnet(**posterior_cfg, diffusion_kwargs=diffusion_kwargs, **sanitized_kwargs)

    prior_encoder = ProbUnetLatentEncoder(prior_unet, latent_dim)
    posterior_encoder = ProbUnetLatentEncoder(posterior_unet, latent_dim)
    fcomb = ProbUnetFcomb(
        feature_channels=base_unet.model_channels,
        latent_dim=latent_dim,
        num_classes=base_unet.num_classes,
        hidden_channels=base_unet.model_channels,
        num_layers=max(1, num_fcomb),
    )
    return ProbabilisticUnetModel(
        base_unet=base_unet,
        prior_encoder=prior_encoder,
        posterior_encoder=posterior_encoder,
        fcomb=fcomb,
        latent_dim=latent_dim,
        beta=beta,
        regularizer_coeff=reg_coeff,
        beta_warmup_epochs=beta_warmup_epochs,
    )


def get_seg_model(cfg, **kwargs):
    """Factory that forwards config as kwargs, assuming keys match DiffUnet.__init__."""
    cfg_dict = OmegaConf.to_container(cfg.MODEL, resolve=True)
    #map keys to lower
    cfg_dict = {k.lower(): v for k, v in cfg_dict.items()}
    swag_requested = bool(cfg_dict.pop("swag", False))
    dropout_rate_override = cfg_dict.pop("dropout_rate", None)
    diffusion_kwargs_cfg = cfg_dict.pop("diffusion_kwargs", None)
    diffusion_sampling_cfg = cfg_dict.pop("diffusion_sampling", None)
    prob_unet_cfg = cfg_dict.pop("prob_unet", None)
    if dropout_rate_override is not None:
        prob = float(dropout_rate_override)
        if prob is not None:
            cfg_dict["dropout"] = prob
    # Remove keys that DiffUnet.__init__ does not accept but might be present in shared configs.
    for stray_key in ("pretrained", "pretrained_weights"):
        cfg_dict.pop(stray_key, None)
    if cfg_dict.get("diffusion", False):
        cfg_dict["in_channels"] += cfg_dict["out_channels"]
    diffusion_kwargs_override = kwargs.pop("diffusion_kwargs", None)
    if diffusion_kwargs_override is not None:
        diffusion_kwargs = diffusion_kwargs_override
    else:
        diffusion_kwargs = diffusion_kwargs_cfg
    diffusion_sampling_override = kwargs.pop("diffusion_sampling", None)
    if diffusion_sampling_override is not None:
        diffusion_sampling = _normalize_sampling_config(diffusion_sampling_override)
    else:
        diffusion_sampling = _normalize_sampling_config(diffusion_sampling_cfg)
    num_steps = diffusion_sampling.get("num_steps")
    sampler = diffusion_sampling.get("sampler")
    if num_steps is not None:
        cfg_dict["diffusion_num_steps"] = num_steps
    if sampler is not None:
        cfg_dict["diffusion_sampler_type"] = sampler
    cfg_dict["swag_enabled"] = swag_requested
    # Hydra can pass extra metadata (e.g., nickname) that DiffUnet does not accept.
    meta_keys = {"nickname"}
    sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}
    if prob_unet_cfg is not None:
        if isinstance(prob_unet_cfg, Mapping):
            prob_mapping = prob_unet_cfg
        elif isinstance(prob_unet_cfg, bool):
            prob_mapping = {}
        else:
            raise TypeError("PROB_UNET configuration must be a mapping or boolean flag")
        return _build_prob_unet_model(cfg_dict, prob_mapping, diffusion_kwargs, sanitized_kwargs)
    model = DiffUnet(**cfg_dict, diffusion_kwargs=diffusion_kwargs, **sanitized_kwargs)
    return model