from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MaskedLayerBase(nn.Module):
    """Base class for mask-parameterized weight layers."""

    def __init__(self, num_masks: int) -> None:
        super().__init__()
        if int(num_masks) <= 0:
            raise ValueError("num_masks must be positive")
        self.num_masks = int(num_masks)
        self.active_mask_idx: int | None = 0
        self.temp: float = 1.0
        self.rows_only: bool = False
        self.normalize: bool = False

    def set_mask_context(self, mask_idx: int | None, temp: float) -> None:
        self.active_mask_idx = mask_idx
        self.temp = float(temp)

    def set_rows_only(self, rows_only: bool) -> None:
        self.rows_only = bool(rows_only)

    def set_normalize(self, normalize: bool) -> None:
        self.normalize = bool(normalize)

    def _mask_rescale_factor(self, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        keep_ratio = mask.to(torch.float32).mean().clamp_min(float(eps)).to(mask.dtype)
        return torch.reciprocal(keep_ratio)

    def _sample_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            binary = F.gumbel_softmax(
                torch.stack([logits, -logits], dim=-1),
                tau=float(self.temp),
                hard=False,
                dim=-1,
            )
            return binary[..., 0]
        return (logits >= 0.0).to(logits.dtype)

    def _selected_logits(self, logits: torch.Tensor, mask_idx: int | None) -> torch.Tensor:
        if mask_idx is None:
            if self.active_mask_idx is None:
                raise RuntimeError(
                    "mask_idx is None and no active_mask_idx is set; call set_mask_context first"
                )
            mask_idx = self.active_mask_idx
        if mask_idx < 0 or mask_idx >= self.num_masks:
            raise IndexError(
                f"mask_idx={mask_idx} is out of range for num_masks={self.num_masks}"
            )
        return logits[mask_idx]


class MaskedLinear(_MaskedLayerBase):
    """Linear layer with independent input/output channel masks per submodel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_masks: int = 10,
    ) -> None:
        super().__init__(num_masks=num_masks)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.mask_logits_inputs = nn.Parameter(torch.zeros(self.num_masks, self.in_features))
        self.mask_logits_outputs = nn.Parameter(torch.zeros(self.num_masks, self.out_features))

    @classmethod
    def from_linear(cls, layer: nn.Linear, num_masks: int) -> "MaskedLinear":
        masked = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            num_masks=num_masks,
        )
        masked = masked.to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            masked.weight.copy_(layer.weight)
            if layer.bias is not None and masked.bias is not None:
                masked.bias.copy_(layer.bias)
        return masked

    def freeze_base_parameters(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def _mask_vectors(self, mask_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        in_logits = self._selected_logits(self.mask_logits_inputs, mask_idx)
        out_logits = self._selected_logits(self.mask_logits_outputs, mask_idx)
        if self.rows_only:
            in_mask = torch.ones_like(in_logits)
        else:
            in_mask = self._sample_binary_probs(in_logits)
        out_mask = self._sample_binary_probs(out_logits)
        return in_mask, out_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_mask, out_mask = self._mask_vectors(mask_idx=None)
        masked_weight = self.weight * out_mask.unsqueeze(1) * in_mask.unsqueeze(0)
        if self.normalize:
            scale = self._mask_rescale_factor(in_mask) * self._mask_rescale_factor(out_mask)
            masked_weight = masked_weight * scale
        if self.bias is not None:
            masked_bias = self.bias * out_mask
            if self.normalize:
                masked_bias = masked_bias * self._mask_rescale_factor(out_mask)
        else:
            masked_bias = None
        return F.linear(x, masked_weight, masked_bias)

    def hard_binary_weights(self) -> torch.Tensor:
        if self.rows_only:
            in_mask = torch.ones_like(self.mask_logits_inputs, dtype=torch.bool)
        else:
            in_mask = self.mask_logits_inputs >= 0.0
        out_mask = self.mask_logits_outputs >= 0.0
        return (out_mask.unsqueeze(-1) & in_mask.unsqueeze(-2)).reshape(self.num_masks, -1)

    def soft_binary_weights(self) -> torch.Tensor:
        if self.rows_only:
            in_prob = torch.ones_like(self.mask_logits_inputs)
        else:
            in_prob = torch.sigmoid(self.mask_logits_inputs * 2.0)
        out_prob = torch.sigmoid(self.mask_logits_outputs * 2.0)
        return (out_prob.unsqueeze(-1) * in_prob.unsqueeze(-2)).reshape(self.num_masks, -1)

    def expected_active_weights(self) -> torch.Tensor:
        return self.soft_binary_weights().sum(dim=1)

    def total_weight_count(self) -> int:
        return int(self.weight.numel())

    def get_combined_mask_logits(self) -> torch.Tensor:
        if self.rows_only:
            return self.mask_logits_outputs
        return torch.cat([self.mask_logits_inputs, self.mask_logits_outputs], dim=1)

    def to_linear(self, mask_idx: int) -> nn.Linear:
        if self.rows_only:
            in_mask = torch.ones(self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        else:
            in_mask = (self.mask_logits_inputs[mask_idx] >= 0.0).to(self.weight.dtype)
        out_mask = (self.mask_logits_outputs[mask_idx] >= 0.0).to(self.weight.dtype)
        layer = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            materialized_weight = self.weight * out_mask.unsqueeze(1) * in_mask.unsqueeze(0)
            if self.normalize:
                materialized_weight = materialized_weight * (
                    self._mask_rescale_factor(in_mask) * self._mask_rescale_factor(out_mask)
                )
            layer.weight.copy_(materialized_weight)
            if self.bias is not None and layer.bias is not None:
                materialized_bias = self.bias * out_mask
                if self.normalize:
                    materialized_bias = materialized_bias * self._mask_rescale_factor(out_mask)
                layer.bias.copy_(materialized_bias)
        return layer


class MaskedConv2d(_MaskedLayerBase):
    """Conv2d wrapper with channel-wise input/output masks per submodel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_masks: int = 10,
    ) -> None:
        super().__init__(num_masks=num_masks)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.mask_logits_inputs = nn.Parameter(torch.zeros(self.num_masks, self.in_channels))
        self.mask_logits_outputs = nn.Parameter(torch.zeros(self.num_masks, self.out_channels))

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, num_masks: int) -> "MaskedConv2d":
        masked = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            num_masks=num_masks,
        )
        masked = masked.to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            masked.weight.copy_(layer.weight)
            if layer.bias is not None and masked.bias is not None:
                masked.bias.copy_(layer.bias)
        return masked

    def freeze_base_parameters(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def _expanded_input_mask(self, in_mask: torch.Tensor) -> torch.Tensor:
        # Returns per-output-channel input masks with grouped convolution support.
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        expanded = torch.empty(
            self.out_channels,
            in_per_group,
            device=in_mask.device,
            dtype=in_mask.dtype,
        )
        for group_idx in range(self.groups):
            in_slice = in_mask[group_idx * in_per_group : (group_idx + 1) * in_per_group]
            out_start = group_idx * out_per_group
            out_end = (group_idx + 1) * out_per_group
            expanded[out_start:out_end] = in_slice.unsqueeze(0).expand(out_per_group, -1)
        return expanded

    def _mask_vectors(self, mask_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        in_logits = self._selected_logits(self.mask_logits_inputs, mask_idx)
        out_logits = self._selected_logits(self.mask_logits_outputs, mask_idx)
        if self.rows_only:
            in_mask = torch.ones_like(in_logits)
        else:
            in_mask = self._sample_binary_probs(in_logits)
        out_mask = self._sample_binary_probs(out_logits)
        return in_mask, out_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_mask, out_mask = self._mask_vectors(mask_idx=None)
        expanded_in = self._expanded_input_mask(in_mask)
        channel_mask = out_mask.unsqueeze(1) * expanded_in
        masked_weight = self.weight * channel_mask.unsqueeze(-1).unsqueeze(-1)
        if self.normalize:
            scale = self._mask_rescale_factor(in_mask) * self._mask_rescale_factor(out_mask)
            masked_weight = masked_weight * scale
        if self.bias is not None:
            masked_bias = self.bias * out_mask
            if self.normalize:
                masked_bias = masked_bias * self._mask_rescale_factor(out_mask)
        else:
            masked_bias = None
        return F.conv2d(
            x,
            masked_weight,
            masked_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def hard_binary_weights(self) -> torch.Tensor:
        if self.rows_only:
            in_mask = torch.ones_like(self.mask_logits_inputs, dtype=torch.bool)
        else:
            in_mask = self.mask_logits_inputs >= 0.0
        out_mask = self.mask_logits_outputs >= 0.0
        return self._flatten_binary_channels(in_mask, out_mask)

    def soft_binary_weights(self) -> torch.Tensor:
        if self.rows_only:
            in_prob = torch.ones_like(self.mask_logits_inputs)
        else:
            in_prob = torch.sigmoid(self.mask_logits_inputs * 2.0)
        out_prob = torch.sigmoid(self.mask_logits_outputs * 2.0)
        return self._flatten_binary_channels(in_prob, out_prob)

    def _flatten_binary_channels(self, in_tensor: torch.Tensor, out_tensor: torch.Tensor) -> torch.Tensor:
        # Returns channel-pair masks with shape (num_masks, out_channels * in_channels_per_group).
        m = in_tensor.shape[0]
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        flat = torch.empty(
            m,
            self.out_channels * in_per_group,
            device=in_tensor.device,
            dtype=in_tensor.dtype,
        )
        for group_idx in range(self.groups):
            in_slice = in_tensor[:, group_idx * in_per_group : (group_idx + 1) * in_per_group]
            out_slice = out_tensor[:, group_idx * out_per_group : (group_idx + 1) * out_per_group]
            block = out_slice.unsqueeze(-1) * in_slice.unsqueeze(-2)
            flat[:, group_idx * out_per_group * in_per_group : (group_idx + 1) * out_per_group * in_per_group] = block.reshape(m, -1)
        return flat

    def expected_active_weights(self) -> torch.Tensor:
        kernel_elems = int(self.kernel_size[0] * self.kernel_size[1])
        return self.soft_binary_weights().sum(dim=1) * kernel_elems

    def total_weight_count(self) -> int:
        return int(self.weight.numel())

    def get_combined_mask_logits(self) -> torch.Tensor:
        if self.rows_only:
            return self.mask_logits_outputs
        return torch.cat([self.mask_logits_inputs, self.mask_logits_outputs], dim=1)

    def to_conv2d(self, mask_idx: int) -> nn.Conv2d:
        if self.rows_only:
            in_mask = torch.ones(self.in_channels, device=self.weight.device, dtype=self.weight.dtype)
        else:
            in_mask = (self.mask_logits_inputs[mask_idx] >= 0.0).to(self.weight.dtype)
        out_mask = (self.mask_logits_outputs[mask_idx] >= 0.0).to(self.weight.dtype)
        expanded_in = self._expanded_input_mask(in_mask)
        channel_mask = out_mask.unsqueeze(1) * expanded_in
        layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        with torch.no_grad():
            materialized_weight = self.weight * channel_mask.unsqueeze(-1).unsqueeze(-1)
            if self.normalize:
                materialized_weight = materialized_weight * (
                    self._mask_rescale_factor(in_mask) * self._mask_rescale_factor(out_mask)
                )
            layer.weight.copy_(materialized_weight)
            if self.bias is not None and layer.bias is not None:
                materialized_bias = self.bias * out_mask
                if self.normalize:
                    materialized_bias = materialized_bias * self._mask_rescale_factor(out_mask)
                layer.bias.copy_(materialized_bias)
        return layer


MaskedLayer = MaskedLinear | MaskedConv2d


@dataclass
class ReplaceSummary:
    linear_replaced: int
    conv_replaced: int

    @property
    def total_replaced(self) -> int:
        return self.linear_replaced + self.conv_replaced


LayerPredicate = Callable[[str, nn.Module], bool]


def replace_with_masked_layers(
    root: nn.Module,
    num_masks: int,
    predicate: LayerPredicate | None = None,
    prefix: str = "",
) -> ReplaceSummary:
    """Recursively replace Linear/Conv2d modules with mask wrappers."""
    linear_count = 0
    conv_count = 0
    for name, child in list(root.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (predicate is None or predicate(qualified, child)):
            setattr(root, name, MaskedLinear.from_linear(child, num_masks=num_masks))
            linear_count += 1
            continue
        if isinstance(child, nn.Conv2d) and (predicate is None or predicate(qualified, child)):
            setattr(root, name, MaskedConv2d.from_conv2d(child, num_masks=num_masks))
            conv_count += 1
            continue
        child_summary = replace_with_masked_layers(
            child,
            num_masks=num_masks,
            predicate=predicate,
            prefix=qualified,
        )
        linear_count += child_summary.linear_replaced
        conv_count += child_summary.conv_replaced
    return ReplaceSummary(linear_replaced=linear_count, conv_replaced=conv_count)


def iter_masked_layers(root: nn.Module) -> Iterable[MaskedLayer]:
    for module in root.modules():
        if isinstance(module, (MaskedLinear, MaskedConv2d)):
            yield module


def set_active_submodel(root: nn.Module, mask_idx: int | None, temp: float) -> None:
    for module in iter_masked_layers(root):
        module.set_mask_context(mask_idx=mask_idx, temp=temp)


def freeze_unmasked_parameters(root: nn.Module) -> None:
    for param in root.parameters():
        param.requires_grad = False
    for module in iter_masked_layers(root):
        module.mask_logits_inputs.requires_grad = not module.rows_only
        module.mask_logits_outputs.requires_grad = True


def set_rows_only_mode(root: nn.Module, rows_only: bool) -> None:
    for module in iter_masked_layers(root):
        module.set_rows_only(rows_only)


def set_normalize_mode(root: nn.Module, normalize: bool) -> None:
    for module in iter_masked_layers(root):
        module.set_normalize(normalize)


def initialize_mask_logits_for_target_fraction(
    root: nn.Module,
    target_fraction: float,
    use_zero_init: bool = True,
    eps: float = 1e-6,
) -> None:
    """Initialize mask logits to match an expected fill ratio at step 0.

    - rows_only=True: output probs ~= target_fraction
    - rows_only=False: input/output probs ~= sqrt(target_fraction)
      so their product ~= target_fraction
    """
    if use_zero_init:
        return
    if target_fraction <= 0.0 or target_fraction > 1.0:
        raise ValueError("target_fraction must be in (0, 1]")

    target = float(min(max(target_fraction, eps), 1.0 - eps))
    sqrt_target = float(min(max(target**0.5, eps), 1.0 - eps))

    def _logit_for_softprob(p: float) -> float:
        # soft_binary_weights uses sigmoid(mask_logits * 2.0), so invert that mapping.
        return 0.5 * float(torch.logit(torch.tensor(p, dtype=torch.float32)).item())

    out_logit = _logit_for_softprob(target)
    in_out_logit = _logit_for_softprob(sqrt_target)

    with torch.no_grad():
        for module in iter_masked_layers(root):
            if module.rows_only:
                module.mask_logits_outputs.fill_(out_logit)
                # Input logits are ignored in rows_only mode but keep deterministic values.
                module.mask_logits_inputs.zero_()
            else:
                module.mask_logits_inputs.fill_(in_out_logit)
                module.mask_logits_outputs.fill_(in_out_logit)


def combined_mask_logits(root: nn.Module) -> torch.Tensor:
    parts = [module.get_combined_mask_logits() for module in iter_masked_layers(root)]
    if not parts:
        raise RuntimeError("No masked layers found in model")
    return torch.cat(parts, dim=1)


def mean_pairwise_iou(root: nn.Module, hard: bool) -> torch.Tensor:
    inter_acc: torch.Tensor | None = None
    union_acc: torch.Tensor | None = None
    for module in iter_masked_layers(root):
        per_model = module.hard_binary_weights().to(torch.float32) if hard else module.soft_binary_weights()
        if per_model.shape[0] <= 1:
            continue
        inter = per_model @ per_model.transpose(0, 1)
        sums = per_model.sum(dim=1, keepdim=True)
        union = sums + sums.transpose(0, 1) - inter
        if inter_acc is None:
            inter_acc = inter
            union_acc = union
        else:
            inter_acc = inter_acc + inter
            union_acc = union_acc + union

    if inter_acc is None or union_acc is None:
        device = next(root.parameters()).device
        return torch.zeros((), device=device)
    i, j = torch.triu_indices(inter_acc.shape[0], inter_acc.shape[1], offset=1)
    if i.numel() == 0:
        return torch.zeros((), device=inter_acc.device)
    pairwise = inter_acc[i, j] / (union_acc[i, j] + 1e-8)
    return pairwise.mean()


def submodel_size_penalty(
    root: nn.Module,
    target_fraction: float,
    global_fraction_penalty: bool = False,
) -> torch.Tensor:
    if target_fraction <= 0.0 or target_fraction > 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    penalties = []
    expected_active_total: torch.Tensor | None = None
    total_weight_count = 0.0
    for module in iter_masked_layers(root):
        expected_active = module.expected_active_weights()
        module_total = float(module.total_weight_count())
        if global_fraction_penalty:
            expected_active_total = (
                expected_active
                if expected_active_total is None
                else expected_active_total + expected_active
            )
            total_weight_count += module_total
            continue
        expected = expected_active / module_total
        penalties.append(F.relu(expected - target_fraction).mean())
    if not penalties:
        if global_fraction_penalty and expected_active_total is not None and total_weight_count > 0.0:
            expected_global = expected_active_total / total_weight_count
            return F.relu(expected_global - target_fraction).mean()
        device = next(root.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def materialize_submodel(root: nn.Module, mask_idx: int) -> nn.Module:
    """Return a deep-copied model with masked layers converted back to plain modules."""
    import copy

    model = copy.deepcopy(root)

    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, MaskedLinear):
                setattr(module, name, child.to_linear(mask_idx))
                continue
            if isinstance(child, MaskedConv2d):
                setattr(module, name, child.to_conv2d(mask_idx))
                continue
            _replace(child)

    _replace(model)
    return model


def count_unwrapped_dense_layers(root: nn.Module) -> tuple[int, int]:
    linear_count = 0
    conv_count = 0
    for module in root.modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
        elif isinstance(module, nn.Conv2d):
            conv_count += 1
    return linear_count, conv_count
