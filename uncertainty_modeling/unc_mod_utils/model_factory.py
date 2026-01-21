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
    if isinstance(dropout_cfg, (DictConfig, Mapping)):
        probability = float(dropout_cfg["probability"])
        if not 0.0 < probability <= 1.0:
            raise ValueError("dropout probability must lie in (0, 1].")
        return bool(dropout_cfg.get("enabled", True)), probability
    raise TypeError(f"dropout_cfg must be a mapping, got {type(dropout_cfg).__name__}.")


def _apply_dropout_overrides(cfg: DictConfig, dropout_cfg: DictConfig | Mapping[str, Any] | None) -> None:
    enabled, probability = _normalize_dropout_cfg(dropout_cfg)
    if not enabled:
        return
    model_section = cfg["MODEL"]
    with open_dict(model_section):
        model_section["DROPOUT_RATE"] = probability
        model_section["DROPOUT"] = probability


def _infer_model_au_type(model: Any) -> str:
    candidates = []
    if getattr(model, "diffusion", False):
        candidates.append("diffusion")
    if getattr(model, "ssn", False):
        candidates.append("ssn")
    if getattr(model, "prob_unet", False):
        candidates.append("prob_unet")
    if len(candidates) > 1:
        raise ValueError(f"Conflicting AU indicators: {candidates}")
    return candidates[0] if candidates else "softmax"


def _attach_au_metadata(model: Any) -> None:
    au_type = _infer_model_au_type(model)
    setattr(model, "AU_type", au_type)
    setattr(model, "is_generative", au_type != "softmax")


def _infer_model_eu_type(model: Any, cfg: Any | None) -> str:
    """Infer ensembling/epistemic uncertainty method for a model.

    Order of precedence:
    - explicit 'eu_method' / 'EU_METHOD' key in provided cfg (if present and valid)
    - model attributes (e.g., 'swag_enabled') and cfg.swag.diag_only when available
    - presence of a positive dropout rate on the model
    - fallback to 'none'
    """
    allowed = {"none", "dropout", "swag", "swag_diag"}
    cfg_map = cfg if isinstance(cfg, (DictConfig, Mapping)) else None
    candidates: set[str] = set()
    explicit = None
    if cfg_map is not None:
        explicit = cfg_map.get("EU_METHOD") or cfg_map.get("eu_method")
    if explicit is not None:
        token = str(explicit).strip().lower()
        if token not in allowed:
            raise ValueError(f"Unsupported EU method '{explicit}'.")
        if token != "none":
            candidates.add(token)
    if getattr(model, "swag_enabled", False):
        swag_type = "swag"
        if cfg_map is not None:
            swag_cfg = cfg_map.get("swag") or cfg_map.get("SWAG")
            if swag_cfg and swag_cfg.get("diag_only"):
                swag_type = "swag_diag"
        candidates.add(swag_type)
    for attr in ("_global_dropout_rate", "_dropout_rate", "dropout"):
        rate = getattr(model, attr, None)
        if rate is not None and float(rate) > 0.0:
            candidates.add("dropout")
            break
    if len(candidates) > 1:
        raise ValueError(f"Conflicting EU indicators: {sorted(candidates)}")
    if candidates:
        return candidates.pop()
    return "none"


def _attach_eu_metadata(model: Any, cfg: Any | None) -> None:
    eu_type = _infer_model_eu_type(model, cfg)
    setattr(model, "EU_type", eu_type)


def instantiate_network(
    target: str,
    cfg: DictConfig | Mapping[str, Any],
    overrides: DictConfig | Mapping[str, Any] | None = None,
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
    _attach_au_metadata(model)
    _attach_eu_metadata(model, merged_cfg)
    return model
