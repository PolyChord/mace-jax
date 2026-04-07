"""Helpers for loading serialized MACE-JAX bundles."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from e3nn_jax import Irrep, Irreps
from flax import nnx, serialization

from mace_jax.nnx_config import ConfigVar
from mace_jax.nnx_utils import state_to_pure_dict, state_to_serializable_dict, wrap_bare_arrays
from mace_jax.tools import model_builder

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_PARAMS_NAME = 'params.msgpack'


@dataclass(frozen=True)
class ModelBundle:
    config: dict
    params: dict
    graphdef: object


def resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    path = Path(model_arg).expanduser().resolve()
    if path.is_dir():
        config_path = path / DEFAULT_CONFIG_NAME
        params_path = path / DEFAULT_PARAMS_NAME
    elif path.suffix == '.json':
        config_path = path
        params_path = path.with_suffix('.msgpack')
    else:
        params_path = path
        config_path = path.with_suffix('.json')
    if not config_path.exists():
        raise FileNotFoundError(
            f'Unable to locate JAX model configuration at {config_path}'
        )
    if not params_path.exists():
        # Fall back to .npz (older mace-jax serialized msgpack with .npz extension)
        npz_fallback = params_path.with_suffix('.npz')
        if npz_fallback.exists():
            params_path = npz_fallback
        else:
            raise FileNotFoundError(
                f'Unable to locate serialized JAX parameters at {params_path}'
            )
    return config_path, params_path


def _maybe_set_dtype(dtype: str | None) -> None:
    if dtype and dtype.lower() == 'float64':
        jax.config.update('jax_enable_x64', True)
    elif dtype:
        jax.config.update('jax_enable_x64', False)


def _load_checkpoint_bundle(path: Path, dtype: str) -> ModelBundle:
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'Checkpoint {path} did not contain a dict payload.')
    model_config = payload.get('model_config')
    if model_config is None:
        raise ValueError(
            f'Checkpoint {path} does not include model_config; '
            're-run training with a newer mace-jax or export a model bundle.'
        )
    state_payload = (
        payload.get('eval_state')
        or payload.get('state')
        or payload.get('eval_params')
        or payload.get('params')
    )
    if state_payload is None:
        raise ValueError(f'Checkpoint {path} is missing state/params payload.')
    model_config, _, _ = model_builder._normalize_atomic_config(model_config)
    _maybe_set_dtype(dtype)
    module = model_builder._build_jax_model(model_config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(module)
    state_template = state_to_serializable_dict(state)
    if isinstance(state_payload, (bytes, bytearray)):
        state_pure = serialization.from_bytes(state_template, state_payload)
    else:
        state_pure = state_payload
    _replace_state_with_specials(state, state_pure)
    state_pure = state_to_pure_dict(state)
    _validate_config_matches_params(model_config, state_pure, context=str(path))
    return ModelBundle(config=model_config, params=state_pure, graphdef=graphdef)


def load_model_bundle(
    model_arg: str,
    dtype: str,
    *,
    wrapper: str | None = None,
) -> ModelBundle:
    path = Path(model_arg).expanduser().resolve()
    if path.suffix == '.ckpt':
        if wrapper not in (None, '', 'mace'):
            raise ValueError('mace-jax only supports the built-in MACE wrapper.')
        return _load_checkpoint_bundle(path, dtype)
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    _maybe_set_dtype(dtype)

    if wrapper not in (None, '', 'mace'):
        raise ValueError('mace-jax only supports the built-in MACE wrapper.')

    config, _, _ = model_builder._normalize_atomic_config(config)
    module = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    wrap_bare_arrays(module)
    graphdef, state = nnx.split(module)
    state_template = state_to_serializable_dict(state)
    state_pure = serialization.from_bytes(state_template, params_path.read_bytes())
    # Extract normalize2mom consts before they conflict with state replacement.
    normalize2mom = None
    if isinstance(state_pure, dict) and '_normalize2mom_consts_var' in state_pure:
        normalize2mom = state_pure.pop('_normalize2mom_consts_var')
    # Flax NNX 0.10+ replace_by_pure_dict is strict about key matching.
    # Instead of fighting it, update the state directly: replace values
    # in the flat state from the deserialized pure dict.
    try:
        nnx.replace_by_pure_dict(state, state_pure)
    except ValueError:
        # Key mismatch — fall back to manual leaf-by-leaf replacement.
        _manual_replace(state, state_pure)
    if normalize2mom is not None:
        cfg = state.get('_normalize2mom_consts_var', None)
        if isinstance(cfg, ConfigVar):
            cfg.set_value(normalize2mom)
    state_pure = state_to_pure_dict(state)
    if normalize2mom is not None:
        state_pure['_normalize2mom_consts_var'] = normalize2mom
    _validate_config_matches_params(config, state_pure, context=str(params_path))
    return ModelBundle(config=config, params=state_pure, graphdef=graphdef)


def _find_param_leaf(tree, keys: Sequence[str]):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    if isinstance(tree, (np.ndarray, getattr(jax, 'Array', ()))):
        return tree
    return None


def _infer_num_elements_from_params(config: dict, params) -> int | None:
    weight = _find_param_leaf(params, ('params', 'node_embedding', 'linear', 'weight'))
    if weight is None:
        weight = _find_param_leaf(params, ('node_embedding', 'linear', 'weight'))
    if weight is None:
        return None
    try:
        hidden_irreps = Irreps(config['hidden_irreps'])
        scalar_mul = int(hidden_irreps.count(Irrep(0, 1)))
    except Exception:
        return None
    if scalar_mul <= 0:
        return None
    weight_numel = int(np.asarray(weight).shape[-1])
    if weight_numel % scalar_mul != 0:
        return None
    return weight_numel // scalar_mul


def _validate_config_matches_params(
    config: dict, params, *, context: str | None = None
) -> None:
    inferred = _infer_num_elements_from_params(config, params)
    if inferred is None:
        return
    atomic_numbers = [int(z) for z in config.get('atomic_numbers', [])]
    current = len(atomic_numbers)
    if current == inferred:
        return
    location = f' in {context}' if context else ''
    raise ValueError(
        'Model config atomic_numbers length does not match parameter shapes'
        f'{location} ({current} vs {inferred}). '
        'Re-export the bundle or checkpoint with matching atomic_numbers.'
    )


__all__ = ['ModelBundle', 'load_model_bundle', 'resolve_model_paths']


def _manual_replace(state: nnx.State, pure_dict: dict) -> None:
    """Leaf-by-leaf state replacement that tolerates key mismatches."""
    from flax.nnx import VariableState

    flat = state.flat_state()
    keys = flat._keys
    values = flat._values
    for idx, kp in enumerate(keys):
        # Walk the pure dict along this key path
        node = pure_dict
        try:
            for k in kp:
                node = node[k]
        except (KeyError, TypeError, IndexError):
            continue
        # node is the replacement value; update the VariableState in-place
        vs = values[idx]
        if isinstance(vs, VariableState):
            vs.value = node


def _replace_state_with_specials(state: nnx.State, state_pure: dict) -> None:
    """Replace state while handling ConfigVar leaves that store dicts."""
    normalize2mom = None
    if isinstance(state_pure, dict) and '_normalize2mom_consts_var' in state_pure:
        normalize2mom = state_pure.pop('_normalize2mom_consts_var')

    # Flax NNX 0.10+ replace_by_pure_dict rejects keys present in the
    # pure dict but absent from the state.  Prune any extras.
    state_keys = set(state.flat_state()._keys)

    def _prune(d, prefix=()):
        pruned = {}
        for k, v in d.items():
            kp = prefix + (k,)
            if isinstance(v, dict):
                sub = _prune(v, kp)
                if sub:
                    pruned[k] = sub
            else:
                if kp in state_keys:
                    pruned[k] = v
        return pruned

    state_pure = _prune(state_pure)
    nnx.replace_by_pure_dict(state, state_pure)

    if normalize2mom is not None:
        cfg = state.get('_normalize2mom_consts_var', None)
        if isinstance(cfg, ConfigVar):
            cfg.set_value(normalize2mom)
