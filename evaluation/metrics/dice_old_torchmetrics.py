# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor

"""This is a copy of torchmetrics.functional.classification.dice and (almost) all its dependencies 
from torchmetrics==0.11.4, as this version is not compatible with our 1.8.2
"""

# INCLUDED WITH ALL THEIR DEPENDENCIES TO AVOID VERSION CONFLICTS
# from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
# from torchmetrics.utilities.checks import _input_format_classification
#from torchmetrics.utilities.data import DataType
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod, DataType
from torch import tensor

def _dice_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: Optional[str],
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Computes dice from the stat scores: true positives, false positives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)
    """
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


def dice(
    preds: Tensor,
    target: Tensor,
    zero_division: int = 0,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Computes `Dice`_:

    .. math:: \text{Dice} = \frac{\text{2 * TP}}{\text{2 * TP} + \text{FP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    It is recommend set `ignore_index` to index of background class.

    The reduction method (how the recall scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        zero_division: The value to use for the score if denominator equals zero
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number of classes

    Raises:
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"`` or ``None``
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torchmetrics.functional import dice
        >>> preds = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> dice(preds, target, average='micro')
        tensor(0.2500)
    """
    allowed_average = ("micro", "macro", "weighted", "samples", "none", None)
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    preds, target = _input_squeeze(preds, target)
    reduce = "macro" if average in ("weighted", "none", None) else average
    tp, fp, _, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_average,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _dice_compute(tp, fp, fn, average, mdmc_average, zero_division)


def _reduce_stat_scores(
    numerator: Tensor,
    denominator: Tensor,
    weights: Optional[Tensor],
    average: Optional[str],
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Reduces scores of type ``numerator/denominator`` or.

    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights: A tensor of weights to be used if ``average='weighted'``.
        average: The method to average the scores
        mdmc_average: The method to average the scores if inputs were multi-dimensional multi-class (MDMC)
        zero_division: The value to use for the score if denominator equals zero.
    """
    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    if weights is None:
        weights = torch.ones_like(denominator)
    else:
        weights = weights.float()

    numerator = torch.where(
        zero_div_mask, tensor(zero_division, dtype=numerator.dtype, device=numerator.device), numerator
    )
    denominator = torch.where(
        zero_div_mask | ignore_mask, tensor(1.0, dtype=denominator.dtype, device=denominator.device), denominator
    )
    weights = torch.where(ignore_mask, tensor(0.0, dtype=weights.dtype, device=weights.device), weights)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)

    scores = weights * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = torch.where(torch.isnan(scores), tensor(zero_division, dtype=scores.dtype, device=scores.device), scores)

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()

    if average in (AverageMethod.NONE, None):
        scores = torch.where(ignore_mask, tensor(float("nan"), device=scores.device), scores)
    else:
        scores = scores.sum()

    return scores


def _stat_scores_update(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = 1,
    threshold: float = 0.5,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    mode: DataType = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns the number of true positives, false positives, true negatives, false negatives. Raises
    ValueError if:

        - The `ignore_index` is not valid
        - When `ignore_index` is used with binary data
        - When inputs are multi-dimensional multi-class, and the ``mdmc_reduce`` parameter is not set

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied
        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handled
        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors
    """

    _negative_index_dropped = False

    if ignore_index is not None and ignore_index < 0 and mode is not None:
        preds, target = _drop_negative_ignored_indices(preds, target, ignore_index, mode)
        _negative_index_dropped = True
    preds, target, _ = _input_format_classification(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
        ignore_index=ignore_index,
    )
    if ignore_index is not None and preds.shape[1] == 1:
        raise ValueError("You can not use `ignore_index` with binary data.")

    if ignore_index is not None and ignore_index >= preds.shape[1]:
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {preds.shape[1]} classes")
    
    if preds.ndim == 3:
        if not mdmc_reduce:
            raise ValueError(
                "When your inputs are multi-dimensional multi-class, you have to set the `mdmc_reduce` parameter"
            )
        if mdmc_reduce == "global":
            preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
            target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

    # Delete what is in ignore_index, if applicable (and classes don't matter):
    if ignore_index is not None and reduce != "macro" and not _negative_index_dropped:
        preds = _del_column(preds, ignore_index)
        target = _del_column(target, ignore_index)

    tp, fp, tn, fn = _stat_scores(preds, target, reduce=reduce)

    # Take care of ignore_index
    if ignore_index is not None and reduce == "macro" and not _negative_index_dropped:
        tp[..., ignore_index] = -1
        fp[..., ignore_index] = -1
        tn[..., ignore_index] = -1
        fn[..., ignore_index] = -1

    return tp, fp, tn, fn


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds: An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target: An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce: One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depends on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then:

        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then:

        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors
    """
    dim: Union[int, List[int]] = 1  # for "samples"
    if reduce == "micro":
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)

    return tp.long(), fp.long(), tn.long(), fn.long()

def _del_column(data: Tensor, idx: int) -> Tensor:
    """Delete the column at index."""
    return torch.cat([data[:, :idx], data[:, (idx + 1) :]], 1)


def _drop_negative_ignored_indices(
    preds: Tensor, target: Tensor, ignore_index: int, mode: DataType
) -> Tuple[Tensor, Tensor]:
    """Remove negative ignored indices.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors

    Return:
        Tensors of preds and target without negative ignore target values.
    """
    if mode == mode.MULTIDIM_MULTICLASS and preds.dtype == torch.float:
        # In case or multi-dimensional multi-class with logits
        n_dims = len(preds.shape)
        num_classes = preds.shape[1]
        # move class dim to last so that we can flatten the additional dimensions into N: [N, C, ...] -> [N, ..., C]
        preds = preds.transpose(1, n_dims - 1)

        # flatten: [N, ..., C] -> [N', C]
        preds = preds.reshape(-1, num_classes)
        target = target.reshape(-1)

    if mode in [mode.MULTICLASS, mode.MULTIDIM_MULTICLASS]:
        preds = preds[target != ignore_index]
        target = target[target != ignore_index]

    return preds, target

def _input_squeeze(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Remove excess dimensions."""
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()
    return preds, target


def _input_format_classification(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, DataType]:
    """Convert preds and target tensors into common format.

    Preds and targets are supposed to fall into one of these categories (and are
    validated to make sure this is the case):

    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``, and target is binary, while preds
      are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and
      is integer (multi-class)
    * preds and target are of shape ``(N, ...)``, target is binary and preds is a float
      (multi-label)
    * preds are of shape ``(N, C, ...)`` and are floats, target is of shape ``(N, ...)``
      and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
      multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``case`` string, which describes which of the above cases the inputs belonged to - regardless
    of whether this was "overridden" by other settings (like ``multiclass``).

    In binary case, targets are normally returned as ``(N,1)`` tensor, while preds are transformed
    into a binary tensor (elements become 1 if the probability is greater than or equal to
    ``threshold`` or 0 otherwise). If ``multiclass=True``, then both targets are preds
    become ``(N, 2)`` tensors by a one-hot transformation; with the thresholding being applied to
    preds first.

    In multi-class case, normally both preds and targets become ``(N, C)`` binary tensors; targets
    by a one-hot transformation and preds by selecting ``top_k`` largest entries (if their original
    shape was ``(N,C)``). However, if ``multiclass=False``, then targets and preds will be
    returned as ``(N,1)`` tensor.

    In multi-label case, normally targets and preds are returned as ``(N, C)`` binary tensors, with
    preds being binarized as in the binary case. Here the ``C`` dimension is obtained by flattening
    all dimensions after the first one. However, if ``multiclass=True``, then both are returned as
    ``(N, 2, C)``, by an equivalent transformation as in the binary case.

    In multi-dimensional multi-class case, normally both target and preds are returned as
    ``(N, C, X)`` tensors, with ``X`` resulting from flattening of all dimensions except ``N`` and
    ``C``. The transformations performed here are equivalent to the multi-class case. However, if
    ``multiclass=False`` (and there are up to two classes), then the data is returned as
    ``(N, X)`` binary tensors (multi-label).

    Note:
        Where a one-hot transformation needs to be performed and the number of classes
        is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
        equal to ``num_classes``, if it is given, or the maximum label value in preds and
        target.

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0 or 1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left unset (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Returns:
        preds: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        target: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        case: The case the inputs fall in, one of ``'binary'``, ``'multi-class'``, ``'multi-label'`` or
            ``'multi-dim multi-class'``
    """
    # Remove excess dimensions
    preds, target = _input_squeeze(preds, target)

    # Convert half precision tensors to full precision, as not all ops are supported
    # for example, min() is not supported
    if preds.dtype == torch.float16:
        preds = preds.float()
    case = _check_classification_inputs(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
        ignore_index=ignore_index,
    )
    if case in (DataType.BINARY, DataType.MULTILABEL) and not top_k:
        preds = (preds >= threshold).int()
        num_classes = num_classes if not multiclass else 2

    if case == DataType.MULTILABEL and top_k:
        preds = select_topk(preds, top_k)
    if case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS) or multiclass:
        if preds.is_floating_point():
            num_classes = preds.shape[1]
            preds = select_topk(preds, top_k or 1)
        else:
            num_classes = num_classes if num_classes else max(preds.max(), target.max()) + 1
            preds = to_onehot(preds, max(2, num_classes))

        target = to_onehot(target, max(2, num_classes))  # type: ignore

        if multiclass is False:
            preds, target = preds[:, 1, ...], target[:, 1, ...]
    if not _check_for_empty_tensors(preds, target):
        if (case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS) and multiclass is not False) or multiclass:
            target = target.reshape(target.shape[0], target.shape[1], -1)
            preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        else:
            target = target.reshape(target.shape[0], -1)
            preds = preds.reshape(preds.shape[0], -1)
    # Some operations above create an extra dimension for MC/binary case - this removes it
    if preds.ndim > 2:
        preds, target = preds.squeeze(-1), target.squeeze(-1)
    return preds.int(), target.int(), case

def _check_for_empty_tensors(preds: Tensor, target: Tensor) -> bool:
    if preds.numel() == target.numel() == 0:
        return True
    return False

def _check_classification_inputs(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    num_classes: Optional[int],
    multiclass: Optional[bool],
    top_k: Optional[int],
    ignore_index: Optional[int] = None,
) -> DataType:
    """Performs error checking on inputs for classification.

    This ensures that preds and target take one of the shape/type combinations that are
    specified in ``_input_format_classification`` docstring. It also checks the cases of
    over-rides with ``multiclass`` by checking (for multi-class and multi-dim multi-class
    cases) that there are only up to 2 distinct labels.

    In case where preds are floats (probabilities), it is checked whether they are in ``[0,1]`` interval.

    When ``num_classes`` is given, it is checked that it is consistent with input cases (binary,
    multi-label, ...), and that, if available, the implied number of classes in the ``C``
    dimension is consistent with it (as well as that max label in target is smaller than it).

    When ``num_classes`` is not specified in these cases, consistency of the highest target
    value against ``C`` dimension is checked for (multi-dimensional) multi-class cases.

    If ``top_k`` is set (not None) for inputs that do not have probability predictions (and
    are not binary), an error is raised. Similarly, if ``top_k`` is set to a number that
    is higher than or equal to the ``C`` dimension of ``preds``, an error is raised.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (``N``).

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of the highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. The default value (``None``) will be
            interpreted as 1 for these inputs. If this parameter is set for multi-label inputs,
            it will take precedence over threshold.

            Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.


    Return:
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'
    """

    # Basic validation (that does not need case/type information)
    _basic_input_validation(preds, target, threshold, multiclass, ignore_index)

    # Check that shape/types fall into one of the cases
    case, implied_classes = _check_shape_and_type_consistency(preds, target)

    # Check consistency with the `C` dimension in case of multi-class data
    if preds.shape != target.shape:
        if multiclass is False and implied_classes != 2:
            raise ValueError(
                "You have set `multiclass=False`, but have more than 2 classes in your data,"
                " based on the C dimension of `preds`."
            )
        if target.max() >= implied_classes:
            raise ValueError(
                "The highest label in `target` should be smaller than the size of the `C` dimension of `preds`."
            )

    # Check that num_classes is consistent
    if num_classes:
        if case == DataType.BINARY:
            _check_num_classes_binary(num_classes, multiclass)
        elif case in (DataType.MULTICLASS, DataType.MULTIDIM_MULTICLASS):
            _check_num_classes_mc(preds, target, num_classes, multiclass, implied_classes)
        elif case.MULTILABEL:
            _check_num_classes_ml(num_classes, multiclass, implied_classes)

    # Check that top_k is consistent
    if top_k is not None:
        _check_top_k(top_k, case, implied_classes, multiclass, preds.is_floating_point())

    return case


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Converts a dense label tensor to one-hot format.

    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Returns:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = torch.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def _basic_input_validation(
    preds: Tensor, target: Tensor, threshold: float, multiclass: Optional[bool], ignore_index: Optional[int]
) -> None:
    """Perform basic validation of inputs that does not require deducing any information of the type of inputs."""
    # Skip all other checks if both preds and target are empty tensors
    if _check_for_empty_tensors(preds, target):
        return

    if target.is_floating_point():
        raise ValueError("The `target` has to be an integer tensor.")

    if ignore_index is None and target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")
    elif ignore_index is not None and ignore_index >= 0 and target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")

    preds_float = preds.is_floating_point()
    if not preds_float and preds.min() < 0:
        raise ValueError("If `preds` are integers, they have to be non-negative.")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError("The `preds` and `target` should have the same first dimension.")

    if multiclass is False and target.max() > 1:
        raise ValueError("If you set `multiclass=False`, then `target` should not exceed 1.")

    if multiclass is False and not preds_float and preds.max() > 1:
        raise ValueError("If you set `multiclass=False` and `preds` are integers, then `preds` should not exceed 1.")

def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k the highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of the highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type ``torch.int32``

    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)
    """
    zeros = torch.zeros_like(prob_tensor)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor = zeros.scatter(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()



def _check_shape_and_type_consistency(preds: Tensor, target: Tensor) -> Tuple[DataType, int]:
    """This checks that the shape and type of inputs are consistent with each other and fall into one of the
    allowed input types (see the documentation of docstring of ``_input_format_classification``). It does not check
    for consistency of number of classes, other functions take care of that.

    It returns the name of the case in which the inputs fall, and the implied number of classes (from the ``C`` dim for
    multi-class data, or extra dim(s) for multi-label data).
    """

    preds_float = preds.is_floating_point()

    if preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        if preds_float and target.numel() > 0 and target.max() > 1:
            raise ValueError(
                "If `preds` and `target` are of shape (N, ...) and `preds` are floats, `target` should be binary."
            )

        # Get the case
        if preds.ndim == 1 and preds_float:
            case = DataType.BINARY
        elif preds.ndim == 1 and not preds_float:
            case = DataType.MULTICLASS
        elif preds.ndim > 1 and preds_float:
            case = DataType.MULTILABEL
        else:
            case = DataType.MULTIDIM_MULTICLASS
        implied_classes = preds[0].numel() if preds.numel() > 0 else 0

    elif preds.ndim == target.ndim + 1:
        if not preds_float:
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )

        implied_classes = preds.shape[1] if preds.numel() > 0 else 0

        if preds.ndim == 2:
            case = DataType.MULTICLASS
        else:
            case = DataType.MULTIDIM_MULTICLASS
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    return case, implied_classes


def _check_num_classes_mc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multiclass: Optional[bool],
    implied_classes: int,
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for (multi-
    dimensional) multi-class data."""

    if num_classes == 1 and multiclass is not False:
        raise ValueError(
            "You have set `num_classes=1`, but predictions are integers."
            " If you want to convert (multi-dimensional) multi-class data with 2 classes"
            " to binary/multi-label, set `multiclass=False`."
        )
    if num_classes > 1:
        if multiclass is False and implied_classes != num_classes:
            raise ValueError(
                "You have set `multiclass=False`, but the implied number of classes "
                " (from shape of inputs) does not match `num_classes`. If you are trying to"
                " transform multi-dim multi-class data with 2 classes to multi-label, `num_classes`"
                " should be either None or the product of the size of extra dimensions (...)."
                " See Input Types in Metrics documentation."
            )
        if target.numel() > 0 and num_classes <= target.max():
            raise ValueError("The highest label in `target` should be smaller than `num_classes`.")
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError("The size of C dimension of `preds` does not match `num_classes`.")


def _check_num_classes_binary(num_classes: int, multiclass: Optional[bool]) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for binary data."""

    if num_classes > 2:
        raise ValueError("Your data is binary, but `num_classes` is larger than 2.")
    if num_classes == 2 and not multiclass:
        raise ValueError(
            "Your data is binary and `num_classes=2`, but `multiclass` is not True."
            " Set it to True if you want to transform binary data to multi-class format."
        )
    if num_classes == 1 and multiclass:
        raise ValueError(
            "You have binary data and have set `multiclass=True`, but `num_classes` is 1."
            " Either set `multiclass=None`(default) or set `num_classes=2`"
            " to transform binary data to multi-class format."
        )

def _check_num_classes_mc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multiclass: Optional[bool],
    implied_classes: int,
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for (multi-
    dimensional) multi-class data."""

    if num_classes == 1 and multiclass is not False:
        raise ValueError(
            "You have set `num_classes=1`, but predictions are integers."
            " If you want to convert (multi-dimensional) multi-class data with 2 classes"
            " to binary/multi-label, set `multiclass=False`."
        )
    if num_classes > 1:
        if multiclass is False and implied_classes != num_classes:
            raise ValueError(
                "You have set `multiclass=False`, but the implied number of classes "
                " (from shape of inputs) does not match `num_classes`. If you are trying to"
                " transform multi-dim multi-class data with 2 classes to multi-label, `num_classes`"
                " should be either None or the product of the size of extra dimensions (...)."
                " See Input Types in Metrics documentation."
            )
        if target.numel() > 0 and num_classes <= target.max():
            raise ValueError(f"The highest label in `target` should be smaller than `num_classes`, found {target.max()}")
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError("The size of C dimension of `preds` does not match `num_classes`.")


def _check_num_classes_ml(num_classes: int, multiclass: Optional[bool], implied_classes: int) -> None:
    """This checks that the consistency of ``num_classes`` with the data and ``multiclass`` param for multi-label
    data."""

    if multiclass and num_classes != 2:
        raise ValueError(
            "Your have set `multiclass=True`, but `num_classes` is not equal to 2."
            " If you are trying to transform multi-label data to 2 class multi-dimensional"
            " multi-class, you should set `num_classes` to either 2 or None."
        )
    if not multiclass and num_classes != implied_classes:
        raise ValueError("The implied number of classes (from shape of inputs) does not match num_classes.")


def _check_top_k(top_k: int, case: str, implied_classes: int, multiclass: Optional[bool], preds_float: bool) -> None:
    if case == DataType.BINARY:
        raise ValueError("You can not use `top_k` parameter with binary data.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("The `top_k` has to be an integer larger than 0.")
    if not preds_float:
        raise ValueError("You have set `top_k`, but you do not have probability predictions.")
    if multiclass is False:
        raise ValueError("If you set `multiclass=False`, you can not set `top_k`.")
    if case == DataType.MULTILABEL and multiclass:
        raise ValueError(
            "If you want to transform multi-label data to 2 class multi-dimensional"
            "multi-class data using `multiclass=True`, you can not use `top_k`."
        )
    if top_k >= implied_classes:
        raise ValueError("The `top_k` has to be strictly smaller than the `C` dimension of `preds`.")

