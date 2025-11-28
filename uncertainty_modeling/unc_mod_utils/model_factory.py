"""Utilities for instantiating models with configuration overrides."""

from __future__ import annotations

from typing import Any, Mapping

import hydra.utils
from omegaconf import DictConfig, OmegaConf, open_dict


def _clone_config(cfg: DictConfig | Mapping[str, Any]) -> DictConfig:
    """Create a mutable copy of the provided OmegaConf config without resolving."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if isinstance(cfg, Mapping):
        return OmegaConf.create(dict(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def _normalize_dropout_cfg(dropout_cfg: DictConfig | Mapping[str, Any] | None) -> tuple[bool, float]:
    if dropout_cfg is None:
        return False, 0.0
    if isinstance(dropout_cfg, DictConfig):
        dropout_cfg = OmegaConf.to_container(dropout_cfg, resolve=True)
    if not isinstance(dropout_cfg, Mapping):
        raise TypeError(
            f"dropout_cfg must be a mapping when provided, got {type(dropout_cfg).__name__}."
        )
    enabled = bool(dropout_cfg.get("enabled", True))
    value = dropout_cfg.get("probability")
    if value is None:
        value = dropout_cfg.get("p", dropout_cfg.get("rate", 0.0))
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.0
    if probability < 0:
        probability = 0.0
    if probability > 1:
        probability = 1.0
    return enabled and probability > 0.0, probability if probability > 0.0 else 0.0


def _apply_dropout_overrides(cfg: DictConfig, dropout_cfg: DictConfig | Mapping[str, Any] | None) -> None:
    enabled, probability = _normalize_dropout_cfg(dropout_cfg)
    if not enabled:
        return
    model_section = cfg.get("MODEL")
    if model_section is None or not isinstance(model_section, DictConfig):
        with open_dict(cfg):
            cfg["MODEL"] = OmegaConf.create({
                "DROPOUT_RATE": probability,
                "DROPOUT": probability,
            })
        return
    with open_dict(model_section):
        model_section["DROPOUT_RATE"] = probability
        model_section["DROPOUT"] = probability


ALLOWED_AU_TYPES = {"softmax", "ssn", "diffusion"}


def _normalize_au_type(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in ALLOWED_AU_TYPES:
        return token
    return None


def _infer_model_au_type(model: Any, nickname: Any | None) -> str:
    explicit = _normalize_au_type(nickname)
    if explicit is not None:
        return explicit
    if bool(getattr(model, "diffusion", False)):
        return "diffusion"
    if bool(getattr(model, "ssn", False)):
        return "ssn"
    return "softmax"


def _attach_au_metadata(model: Any, nickname: Any | None) -> None:
    au_type = _infer_model_au_type(model, nickname)
    setattr(model, "AU_type", au_type)
    setattr(model, "is_generative", au_type != "softmax")


def instantiate_network(
    target: str,
    cfg: DictConfig | Mapping[str, Any],
    overrides: DictConfig | Mapping[str, Any] | None = None,
    nickname: str | None = None,
    dropout_cfg: DictConfig | Mapping[str, Any] | None = None,
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

    _apply_dropout_overrides(merged_cfg, dropout_cfg)

    instantiate_conf = {"_target_": target, "cfg": merged_cfg}
    if kwargs:
        instantiate_conf.update(kwargs)

    model = hydra.utils.instantiate(instantiate_conf)
    _attach_au_metadata(model, nickname)
    return model
