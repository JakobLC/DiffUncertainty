from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SeededDropoutConfig:
    share_across_batch: bool = True


class _SeededDropoutBase(nn.Module):
    def __init__(
        self,
        p: float,
        base_seed: int,
        layer_offset: int,
        share_across_batch: bool = True,
    ) -> None:
        super().__init__()
        self.p = float(p) if p is not None else 0.0
        self.base_seed = int(base_seed)
        self.layer_offset = int(layer_offset)
        self.share_across_batch = bool(share_across_batch)

    def set_base_seed(self, seed: int) -> None:
        self.base_seed = int(seed)

    def _make_generator(self, device: torch.device) -> torch.Generator:
        # torch.Generator supports per-device generators for CUDA on modern PyTorch,
        # but fall back to CPU if unavailable.
        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator()
        gen.manual_seed(int(self.base_seed) + int(self.layer_offset))
        return gen


class SeededMCDropout(_SeededDropoutBase):
    """Deterministic MC Dropout (elementwise) controlled by an explicit seed.

    Designed to replace the project's MC_Dropout modules during evaluation when
    `--same_dropout` is enabled.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        base_seed: int = 0,
        layer_offset: int = 0,
        share_across_batch: bool = True,
    ) -> None:
        super().__init__(
            p=p,
            base_seed=base_seed,
            layer_offset=layer_offset,
            share_across_batch=share_across_batch,
        )
        self.inplace = bool(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x

        keep_prob = 1.0 - float(self.p)
        if keep_prob <= 0.0:
            return torch.zeros_like(x)

        gen = self._make_generator(x.device)

        if self.share_across_batch and x.dim() >= 1:
            mask_shape = (1, *x.shape[1:])
        else:
            mask_shape = tuple(x.shape)

        rand = torch.rand(mask_shape, device=x.device, dtype=torch.float32, generator=gen)
        mask = (rand >= float(self.p)).to(dtype=x.dtype)
        if self.share_across_batch and x.dim() >= 1:
            mask = mask.expand_as(x)

        if self.inplace:
            x.mul_(mask).div_(keep_prob)
            return x
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p:g}, base_seed={self.base_seed}, layer_offset={self.layer_offset}, share_across_batch={self.share_across_batch}"


class SeededMCDropout2d(_SeededDropoutBase):
    """Deterministic MC Dropout2d controlled by an explicit seed."""

    def __init__(
        self,
        p: float = 0.5,
        base_seed: int = 0,
        layer_offset: int = 0,
        share_across_batch: bool = True,
    ) -> None:
        super().__init__(
            p=p,
            base_seed=base_seed,
            layer_offset=layer_offset,
            share_across_batch=share_across_batch,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x
        if x.dim() != 4:
            raise ValueError(f"SeededMCDropout2d expects NCHW tensor, got shape={tuple(x.shape)}")

        keep_prob = 1.0 - float(self.p)
        if keep_prob <= 0.0:
            return torch.zeros_like(x)

        gen = self._make_generator(x.device)

        # Channel-wise dropout mask (like Dropout2d) with optional sharing across batch.
        if self.share_across_batch:
            mask_shape = (1, x.shape[1], 1, 1)
        else:
            mask_shape = (x.shape[0], x.shape[1], 1, 1)

        rand = torch.rand(mask_shape, device=x.device, dtype=torch.float32, generator=gen)
        mask = (rand >= float(self.p)).to(dtype=x.dtype)
        if self.share_across_batch:
            mask = mask.expand(x.shape[0], -1, 1, 1)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p:g}, base_seed={self.base_seed}, layer_offset={self.layer_offset}, share_across_batch={self.share_across_batch}"


def _replace_named_child(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    # setattr is enough for standard Module attribute children.
    setattr(parent, name, new_module)


def enable_seeded_dropout(
    model: nn.Module,
    *,
    base_seed: int,
    config: SeededDropoutConfig | None = None,
) -> int:
    """Replace project MC dropout modules with seeded deterministic versions.

    Returns the number of dropout modules replaced.
    """
    cfg = config or SeededDropoutConfig()
    replaced = 0
    layer_offset = 0

    def _visit(module: nn.Module) -> None:
        nonlocal replaced, layer_offset
        for child_name, child in list(module.named_children()):
            # Avoid double-wrapping.
            if isinstance(child, _SeededDropoutBase):
                layer_offset += 1
                continue

            cls_name = type(child).__name__
            if cls_name == "MC_Dropout":
                p = float(getattr(child, "p", 0.0) or 0.0)
                inplace = bool(getattr(child, "inplace", False))
                new = SeededMCDropout(
                    p=p,
                    inplace=inplace,
                    base_seed=base_seed,
                    layer_offset=layer_offset,
                    share_across_batch=cfg.share_across_batch,
                )
                _replace_named_child(module, child_name, new)
                replaced += 1
                layer_offset += 1
                continue
            if cls_name == "MC_Dropout2d":
                p = float(getattr(child, "p", 0.0) or 0.0)
                new = SeededMCDropout2d(
                    p=p,
                    base_seed=base_seed,
                    layer_offset=layer_offset,
                    share_across_batch=cfg.share_across_batch,
                )
                _replace_named_child(module, child_name, new)
                replaced += 1
                layer_offset += 1
                continue

            _visit(child)

    _visit(model)
    return replaced


def set_seeded_dropout_seed(model: nn.Module, seed: int) -> int:
    """Set base seed on all seeded dropout modules in the model.

    Returns the number of modules updated.
    """
    updated = 0
    for module in model.modules():
        if isinstance(module, _SeededDropoutBase):
            module.set_base_seed(seed)
            updated += 1
    return updated
