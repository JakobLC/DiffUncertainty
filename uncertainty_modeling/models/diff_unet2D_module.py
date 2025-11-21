import math
import warnings
from abc import abstractmethod

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

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
        num_groups = min(32, num_channels)
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
            self.fp16_attrs.append("time_embed")
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                self.act(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed = None
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
        self.out_channels = out_channels
        # expose common attributes expected elsewhere in the repo
        self.num_classes = out_channels

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        mean_only: bool = False,
    ) -> torch.Tensor | tuple[td.LowRankMultivariateNormal, bool]:
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] input image tensor.
        :param timesteps: a 1-D batch of timesteps or a 0-D single timestep to repeat.
        :return: an [N x C x ...] Tensor of predicted masks.
        """
        h = x
        emb: torch.Tensor | None
        if self.diffusion:
            if timesteps is None:
                raise ValueError("timesteps must be provided when diffusion=True")
            if timesteps.numel() == 1:
                timesteps = timesteps.expand(h.shape[0])
            emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels)
            )
        else:
            # Non-diffusion mode: ignore timestep conditioning.
            emb = None

        hs = []
        depth = 0
        for module, skip in zip(self.input_blocks, self.input_skip):
            h = module(h, emb)
            if skip:
                hs.append(h)
            else:
                hs.append(0)
            depth += 1
        h = self.middle_block(h, emb)
        depth += 1
        for module in self.output_blocks:
            if self.new_upsample_method:
                cat_in = h + hs.pop()
            else:
                cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            depth += 1
        h = h.type(x.dtype)
        if not self.ssn:
            return self.out(h)

        mean_logits = self.out(h)
        distribution, cov_failed_flag = self._build_ssn_distribution(
            features=h, mean_logits=mean_logits, mean_only=mean_only
        )
        return distribution, cov_failed_flag

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
            if not torch.all(torch.isfinite(safe_diag)):
                warnings.warn(
                    "Non-finite values encountered in covariance diagonal.",
                    RuntimeWarning,
                )
            scale = torch.sqrt(safe_diag).clamp(min=self.ssn_eps)
            distribution = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return distribution, cov_failed_flag

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
            nn.Dropout(p=dropout),
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
            nn.Dropout(p=dropout),
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


def get_seg_model(cfg, **kwargs):
    """Factory that forwards config as kwargs, assuming keys match DiffUnet.__init__."""
    cfg_dict = OmegaConf.to_container(cfg.MODEL, resolve=True)
    #map keys to lower
    cfg_dict = {k.lower(): v for k, v in cfg_dict.items()}
    # Hydra can pass extra metadata (e.g., nickname) that DiffUnet does not accept.
    meta_keys = {"nickname"}
    sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}
    model = DiffUnet(**cfg_dict, **sanitized_kwargs)
    return model