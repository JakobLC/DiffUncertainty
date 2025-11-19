"""Utilities for instantiating models with configuration overrides."""

from __future__ import annotations

from typing import Any, Mapping

import hydra.utils
from omegaconf import DictConfig, OmegaConf


def _clone_config(cfg: DictConfig | Mapping[str, Any]) -> DictConfig:
    """Create a mutable copy of the provided OmegaConf config without resolving."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if isinstance(cfg, Mapping):
        return OmegaConf.create(dict(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def instantiate_network(
    target: str,
    cfg: DictConfig | Mapping[str, Any],
    overrides: DictConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a model network with optional configuration overrides.

    Args:
        target: Dotted path to the callable used for instantiation.
        cfg: Base configuration from the network definition.
        overrides: Optional overrides applied on top of the base configuration.
        **kwargs: Additional keyword arguments forwarded to ``hydra.utils.instantiate``.

    Returns:
        Instantiated model object.
    """
    if not target:
        raise ValueError("'target' must be a non-empty string")

    merged_cfg = _clone_config(cfg)

    if overrides is not None:
        override_cfg = _clone_config(overrides)
        merged_cfg = OmegaConf.merge(merged_cfg, override_cfg)

    instantiate_conf = {"_target_": target, "cfg": merged_cfg}
    if kwargs:
        instantiate_conf.update(kwargs)

    return hydra.utils.instantiate(instantiate_conf)
