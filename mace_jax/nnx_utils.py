"""NNX helper utilities for MACE-JAX."""

from __future__ import annotations

from typing import Any

from flax import nnx
from flax.nnx import VariableState

from mace_jax.nnx_config import ConfigDict, ConfigVar


def state_to_pure_dict(state: nnx.State) -> dict[str, Any]:
    """Convert an NNX State into a pure dict of arrays."""

    def _extract(value):
        # Flax NNX 0.10+ produces VariableState (not Variable subclass)
        # after nnx.split(); handle it before the Variable checks.
        if isinstance(value, VariableState):
            if issubclass(value.type, ConfigVar):
                config_val = value.value
                if isinstance(config_val, dict) and not isinstance(config_val, ConfigDict):
                    return ConfigDict(config_val)
                return config_val
            return value.value
        if isinstance(value, ConfigVar):
            config_val = value.get_value()
            if isinstance(config_val, dict) and not isinstance(config_val, ConfigDict):
                return ConfigDict(config_val)
            return config_val
        if isinstance(value, nnx.Variable):
            return value.get_value()
        return value

    return nnx.to_pure_dict(state, extract_fn=_extract)


def state_to_serializable_dict(state: nnx.State) -> dict[str, Any]:
    """Convert an NNX State into a pure dict safe for serialization."""

    def _convert(obj):
        if isinstance(obj, ConfigDict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_convert(v) for v in obj)
        return obj

    return _convert(state_to_pure_dict(state))


def replace_state_from_pure_dict(state: nnx.State, values: dict[str, Any]) -> nnx.State:
    """Replace values in an NNX State from a pure dict."""
    nnx.replace_by_pure_dict(state, values)
    return state


def wrap_bare_arrays(module: nnx.Module) -> None:
    """Wrap bare ndarray/jax-array attributes on NNX modules in nnx.Variable.

    Flax NNX >= 0.10 raises 'Arrays leaves are not supported' when
    nnx.split() encounters a raw array stored directly on a module.
    This walks the module tree and wraps any such arrays.
    """
    import jax.numpy as jnp
    import numpy as np

    visited: set[int] = set()

    def _walk(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if not isinstance(obj, nnx.Module):
            return

        for attr_name in list(vars(obj)):
            val = getattr(obj, attr_name)
            if isinstance(val, (np.ndarray, jnp.ndarray)):
                setattr(obj, attr_name, nnx.Variable(val))
            elif isinstance(val, nnx.Module):
                _walk(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, nnx.Module):
                        _walk(item)

    _walk(module)
