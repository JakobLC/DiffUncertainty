import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

import math
import torch

import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from torch import nn
from omegaconf import OmegaConf

from models.nn import (timestep_embedding,checkpoint,identity_module,total_model_norm)
from models.fp16 import (convert_module_to_f16,
                                         convert_module_to_f32)

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

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

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def identity_module(module,raise_error=True):
    """
    Make valid layers do an identity mapping by setting parameters appropriately.
    Asserts that the number of input and output channels are the same.
    """
    try:
        valid_layers = [nn.Linear,nn.Conv2d]
        assert isinstance(module, tuple(valid_layers)), f"module should be one of the valid layers: {valid_layers}, found: {module}"
        if isinstance(module, nn.Linear):
            assert module.in_features == module.out_features, f"module should have same number of input and output features, found: {module.in_features} and {module.out_features}"
            module.weight.data.copy_(torch.eye(module.in_features))
            module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            assert module.in_channels == module.out_channels, f"module should have same number of input and output channels, found: {module.in_channels} and {module.out_channels}"
            module.weight.data.zero_()
            module.bias.data.zero_()
            #set central pixels to 1s such that no change in input
            for i in range(module.out_channels):
                module.weight.data[i,i,module.kernel_size[0]//2,module.kernel_size[1]//2] = 1
        success = 1
    except AssertionError as e:
        if raise_error:
            raise e
        success = 0
    return success

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class UNetModel(nn.Module):
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
        dims=2,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        no_diffusion=False,
        final_act="none",
        one_skip_per_reso=False,
        new_upsample_method=False,
        mlp_attn=False,
        act=nn.SiLU
    ):
        super().__init__()
        self.act = act
        self.mlp_attn = mlp_attn
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

        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.no_diffusion = no_diffusion
        self.fp16_attrs = ["input_blocks","output_blocks"]
        if num_middle_res_blocks>=1:
            self.fp16_attrs.append("middle_block")

        self.fp16_attrs.append("time_embed")
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            self.act(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.in_channels = in_channels

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(self.in_channels, model_channels, 3, padding=1))
            ])
        self.input_skip = [False]
        input_block_chans = [model_channels]
        ch = model_channels
        res_block_kwargs = {"emb_channels": time_embed_dim,
                            "dropout": dropout,
                            "dims": dims,
                            "use_checkpoint": use_checkpoint,
                            "use_scale_shift_norm": use_scale_shift_norm,
                            "act": act}
        attn_kwargs = {"use_checkpoint": use_checkpoint,
                        "num_heads": num_heads,
                        "with_xattn": False,
                        "xattn_channels": None}
        resolution = 0
        
        assert channel_mult[0]==1, "channel_mult[0] must be 1"
        for level, (mult, n_res_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            for _ in range(n_res_blocks):
                if self.new_upsample_method:
                    ch = mult*model_channels
                    ch_in = ch
                else:
                    ch_in = ch
                    ch = mult*model_channels
                layers = [
                    
                ]
                
                if resolution in self.attention_resolutions:
                    if self.mlp_attn:
                        layers = [MLPBlock(ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                    else:
                        layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_skip.append(False)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                resolution += 1
                ch_out = channel_mult[resolution]*model_channels if self.new_upsample_method else None
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, channels_out=ch_out))
                )
                self.input_skip[-1] = True
                self.input_skip.append(False)
                input_block_chans.append(ch)
        if resolution in self.attention_resolutions:
            if self.mlp_attn:
                middle_layers = (sum([[MLPBlock(ch,**res_block_kwargs),
                               AttentionBlock(ch,**attn_kwargs)] 
                               for _ in range(num_middle_res_blocks-1)],[])+
                            [MLPBlock(ch,**res_block_kwargs)])
            else:
                middle_layers = (sum([[ResBlock(ch,**res_block_kwargs),
                               AttentionBlock(ch,**attn_kwargs)] 
                               for _ in range(num_middle_res_blocks-1)],[])+
                            [ResBlock(ch,**res_block_kwargs)])
        else:
            middle_layers = [ResBlock(ch,**res_block_kwargs) for _ in range(num_middle_res_blocks)]

        self.middle_block = TimestepEmbedSequential(*middle_layers)

        attn_kwargs["num_heads"] = num_heads_upsample
        self.output_blocks = nn.ModuleList([])
        for level, mult, n_res_blocks in zip(reversed(list(range(len(channel_mult)))),channel_mult[::-1],num_res_blocks[::-1]):
            for i in range(n_res_blocks + 1):
                if self.new_upsample_method:
                    ch = model_channels * mult
                    ch_in = ch
                else:
                    ch_in = ch+input_block_chans.pop()
                    ch = model_channels * mult
                if resolution in self.attention_resolutions:
                    if self.mlp_attn:
                        layers = [MLPBlock(ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                    else:
                        layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                if level and i == n_res_blocks:
                    resolution -= 1
                    ch_out = channel_mult[resolution]*model_channels if self.new_upsample_method else None
                    layers.append(Upsample(ch, conv_resample, dims=dims, channels_out=ch_out,
                                           mode="bilinear" if self.new_upsample_method else "nearest"))
                    
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        if self.one_skip_per_reso:
            assert self.new_upsample_method, "one_skip_per_reso only works with new_upsample_method"
        else:
            self.input_skip = [True for _ in self.input_skip]

        final_act_dict = {"none": nn.Identity(),
                       "softmax": nn.Softmax(dim=1),
                          "tanh": nn.Tanh()}
        self.out = nn.Sequential(
            nn.Identity(),#unnecessary, but kept for key consistency
            nn.GroupNorm32(ch),
            self.act(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
            final_act_dict[final_act.lower()]
        )
        self.out_channels = out_channels

    def initialize_as_identity(self,verbose=False):
        """Initializes parameters in all modules such that the model behaves as an identity function. 
        Convolutions are initialized with zeros, except for a 1 in their central pixel. Bias terms are set to zero.
        BatchNorm layers are initialized such that their output is zero. 
        """
        start_names = ['input_blocks', 'middle_block', 'output_blocks', 'out']
        success_params = 0
        total_params = 0
        t_before = total_model_norm(self)
        total_params = sum([p.numel() for p in self.parameters()])
        for name,m in self.named_modules():            
            if isinstance(m,(nn.Linear,nn.Conv2d)) and any([name.startswith(n) for n in start_names]):
                success = identity_module(m,raise_error=False)
                if success:
                    success_params += sum([p.numel() for p in m.parameters()])
        t_after = total_model_norm(self)
        if verbose:
            print(f"Initialized {success_params}/{total_params} ({100*success_params/total_params:.2f}%) parameters as identity")
            print(f"Model norm before: {t_before:.2f}, after: {t_after:.2f}, relative change: {100*(t_after-t_before)/t_before:.2f}%")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def to_xattn(self, vit_output, depth):
        if self.vit_injection_type!="xattn":
            out = None
        else:
            if self.block_info.iloc[depth]["has_attention"]:
                out = vit_output
            else:
                out = None
        return out

    def apply_class_emb(self, classes):
        emb = 0
        for i,k in enumerate(self.class_dict.keys()):
            emb += self.class_emb[k](classes[:,i])
        return emb

    def forward(self, image, sample, timesteps):
        """
        Apply the model to an input batch.

        :param sample: an [N x C x ...] Diffusion sample tensor.
        :param timesteps: a 1-D batch of timesteps or a 0-D single timestep to repeat.
        :param image: an [N x C x ...] image tensor.
        :return: an [N x C x ...] Tensor of predicted masks.
        """
        bs = sample.shape[0]
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(bs)
        assert image.shape[1]==3 and sample.shape[1]==1, f"image shape: {image.shape}, sample shape: {sample.shape}" #TODO remove
        h = torch.cat([sample, image], dim=1).type(self.inner_dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        depth = 0
        for module,skip in zip(self.input_blocks,self.input_skip):
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
        h = h.type(sample.dtype)
        h = self.out(h)
        return h

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
        use_checkpoint=False,
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
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            nn.GroupNorm32(channels),
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
        
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    
    def _forward(self, x, emb):
        #b, c, *spatial = x.shape
        #x = x.reshape(b, c, -1)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return (self.skip_connection(x) + h)#.reshape(b, c, *spatial)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False, with_xattn=False, xattn_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.with_xattn = with_xattn
        if self.with_xattn:
            if xattn_channels is None:
                xattn_channels = channels
            self.xattn_channels = xattn_channels
            self.qk_x = nn.Conv1d(xattn_channels, 2*channels, 1) 
            self.v_x = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm32(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x, y=None):
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

    def __init__(self, channels, use_conv, channels_out=None, dims=2, mode="nearest"):
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

    def __init__(self, channels, use_conv, channels_out=None, dims=2):
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
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        act=nn.SiLU
    ):
        super().__init__()
        self.act = act
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm32(channels),
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
            nn.GroupNorm32(self.out_channels),
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

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        
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
        return self.skip_connection(x) + h


class UnetModel(nn.Module):
    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config.MODEL.EXTRA
        super(UnetModel, self).__init__()
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS
        self.num_classes = config.DATASET.NUM_CLASSES

        if self.ssn:
            self.cov_factor_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=config.DATASET.NUM_CLASSES * self.rank,
                    kernel_size=extra.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
                ),
            )

    def unet_ssn(self, x, x_size, mean_only):
        mean = self.last_layer(x)
        mean = F.interpolate(
            mean, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
        )
        batch_size = mean.shape[0]
        mean = mean.view((batch_size, -1))
        # if mean_only:
        #     return mean
        cov_diag = self.last_layer(x).exp() + self.epsilon
        cov_diag = F.interpolate(
            cov_diag, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
        )
        cov_diag = cov_diag.clamp(min=self.epsilon)
        cov_diag = cov_diag.view((batch_size, -1))
        if mean_only:
            cov_factor = torch.zeros([*cov_diag.shape, self.rank])
        else:
            cov_factor = self.cov_factor_conv(x)
            cov_factor = F.interpolate(
                cov_factor, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
            )
            cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
            cov_factor = cov_factor.flatten(2, 3)
            cov_factor = cov_factor.transpose(1, 2)
        try:
            distribution = td.LowRankMultivariateNormal(
                loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
            )
            cov_failed_flag = False
        except:
            cov_failed_flag = True
            distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )

        return distribution, cov_failed_flag

    def forward(self, x, mean_only=False):

        if self.ssn:
            x, cov_failed_flag = self.unet_ssn(x, x_size, mean_only)
            return x, cov_failed_flag
        else:
            x = self.last_layer(x)

            x = F.interpolate(
                x, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
            )
            return x

    def init_weights(self):
        print("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, pretrained):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={"cuda:0": "cpu"}, weights_only=True)
            print("Loading pretrained weights {}".format(pretrained))

            # some preprocessing
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = pretrained_dict["state_dict"]
            if any([k.startswith("ema_model.") for k in pretrained_dict.keys()]):
                raise ValueError("Unexpected EMA weights in pretrained model, probably missing code to handle this case.")
            pretrained_dict = {
                k.replace("model.", "")
                .replace("module.", "")
                .replace("backbone.", ""): v
                for k, v in pretrained_dict.items()
            }

            model_dict = self.state_dict()

            # find weights which match to the model
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            no_match = set(model_dict) - set(pretrained_dict)

            # check if shape of pretrained weights match to the model
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if v.shape == model_dict[k].shape
            }
            shape_mismatch = (set(model_dict) - set(pretrained_dict)) - no_match
            total = len(model_dict)
            # log info about weights which are not found and weights which have a shape mismatch
            if len(no_match):
                num = len(no_match)
                if num >= 5:
                    no_match = list(no_match)[:5]
                    no_match.append("...")
                print(f"No pretrained Weights found for {num}/{total} layers: {no_match}")
            if len(shape_mismatch):
                num = len(shape_mismatch)
                if num >= 5:
                    shape_mismatch = list(shape_mismatch)[:5]
                    shape_mismatch.append("...")
                print(f"Shape Mismatch for {num}/{total} layers: {shape_mismatch}")

            # load weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            del model_dict, pretrained_dict
            print("Weights successfully loaded")
        else:
            raise NotImplementedError(f"No Pretrained Weights found for {pretrained}")


def get_seg_model(cfg, **kwargs):
    model = UnetModel(cfg)
    if cfg.MODEL.PRETRAINED:
        model.load_weights(cfg.MODEL.PRETRAINED_WEIGHTS)
    return model


if __name__ == "__main__":
    args = OmegaConf.load("./configs/config.yaml")
    unet = create_unet_from_args(args["unet"])
    print(unet)
    im = torch.randn(8, 3, 128, 128)
    mask = torch.randn(8, 1, 128, 128)
    t = torch.zeros([8])
    out = unet(im, mask, t)
    print(out.shape)