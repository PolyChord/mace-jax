"""Focus-set effective-energy decomposition for MACE-jax.

Given a complex of ``M`` molecules (or, more generally, fragments) labelled by a
per-atom ``fragment_index`` array, and a *focus set* ``F`` of fragment ids, this
module evaluates

    E_eff(F)   = E(C) - E(C \\ F)
    E_eff^0(F) = E(C) - E(C \\ F) - E(F)

where ``C = {0, ..., M-1}`` is the full complex and ``E(S)`` is the model
energy of the sub-graph that contains only the atoms whose fragment id lies in
``S``. By the Moebius inversion of the many-body expansion (see
``notes/energy_decomposition.tex``), ``E_eff(F)`` is exactly the sum of all MBE
terms whose subset intersects ``F``, while every interaction confined to the
spectator fragments ``C \\ F`` cancels.

In the typical use case (``F = {0}``, "molecule 0 is the one we care about"),
the resulting energy surface contains every minimum and saddle that involves
molecule 0, and is flat in directions that move spectator molecules around each
other while keeping them outside molecule 0's receptive field.

Caveats:
- Receptive field is ``L * r_max`` where ``L`` is ``num_interactions``. Pairs
  separated by more than this distance contribute zero to ``E_eff`` by
  construction, regardless of whether long-range physics should be present.
- Stresses are not returned; spectator-only stress contributions would also
  cancel and the implementation only handles energy + forces.
- The neighbour list is rebuilt for each sub-system, so edges that would have
  crossed the focus/spectator boundary are simply absent from the spectator
  graph; the cancellation is therefore not an algebraic identity at the level
  of edges, only at the level of energies and their gradients.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import ase
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import traverse_util

from mace_jax import data as data_module
from mace_jax.data.utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    graph_from_configuration,
)


__all__ = [
    'EffectiveEnergy',
    'EffectiveEnergyResult',
    'effective_energy_calculator',
]


def _stop_grad(variables):
    flat = traverse_util.flatten_dict(variables)
    return traverse_util.unflatten_dict(
        {k: jax.lax.stop_gradient(v) for k, v in flat.items()}
    )


@dataclass
class EffectiveEnergyResult:
    """Output of an effective-energy evaluation.

    ``energy`` is ``E_eff`` (or ``E_eff^0`` if ``baseline_subtract`` was used).
    ``forces`` has shape ``(N_full, 3)`` indexed in the original ``atoms``
    layout. ``components`` keeps the three constituent energies for inspection.
    """

    energy: float
    forces: np.ndarray
    components: dict[str, float]


class EffectiveEnergy:
    """Wrap a fitted MACE-jax model with the focus-set decomposition.

    Parameters
    ----------
    model :
        A callable ``model(params, graph, *, compute_force, compute_stress)``
        returning a dict with at least ``'energy'`` and ``'forces'`` (the same
        signature used by :class:`mace_jax.calculators.MACEJAXCalculator`).
    params :
        Model parameters; will be ``stop_gradient``'d so they are not
        differentiated through.
    r_max :
        Cutoff used to build neighbour lists. Must match the model's ``r_max``.
    atomic_numbers :
        Optional list of atomic numbers used during training; required if the
        model expects species indices into a fixed table rather than raw Z.
    pad_n_graph :
        Padding budget for ``jraph.pad_with_graphs``; ``2`` adds one dummy
        graph (matches the existing calculator).
    """

    def __init__(
        self,
        model: Callable,
        params,
        r_max: float,
        *,
        atomic_numbers: Iterable[int] | None = None,
        pad_n_graph: int = 2,
    ) -> None:
        self.model = model
        self.params = _stop_grad(params)
        self.r_max = float(r_max)
        self.pad_n_graph = int(pad_n_graph)

        if atomic_numbers is not None:
            self._z_table = AtomicNumberTable([int(z) for z in atomic_numbers])
        else:
            self._z_table = None

        self._predict = jax.jit(
            lambda w, g: self.model(w, g, compute_force=True, compute_stress=False)
        )
        # Track largest edge budget per shape-key to amortise jit recompiles.
        self._edge_budgets: dict[tuple, int] = {}

    # ---- public API -----------------------------------------------------

    def __call__(
        self,
        atoms: ase.Atoms,
        fragment_index: np.ndarray,
        focus: Iterable[int],
        *,
        baseline_subtract: bool = False,
    ) -> EffectiveEnergyResult:
        """Evaluate the focus-set effective energy.

        Parameters
        ----------
        atoms :
            Full complex.
        fragment_index :
            Integer array of length ``len(atoms)`` assigning each atom to a
            fragment.
        focus :
            Iterable of fragment ids that constitute ``F``.
        baseline_subtract :
            If ``True``, also subtract ``E(F)`` so the surface vanishes when
            ``F`` is removed to infinity from the spectators (returns
            ``E_eff^0``).
        """
        fragment_index = np.asarray(fragment_index, dtype=np.int64)
        if fragment_index.shape != (len(atoms),):
            raise ValueError(
                f'fragment_index has shape {fragment_index.shape}, '
                f'expected ({len(atoms)},).'
            )
        focus_set = set(int(f) for f in focus)
        if not focus_set:
            raise ValueError('focus set must be non-empty.')

        focus_mask = np.isin(fragment_index, list(focus_set))
        spectator_mask = ~focus_mask
        if not focus_mask.any():
            raise ValueError(
                f'No atoms belong to any focus fragment {sorted(focus_set)}.'
            )
        if not spectator_mask.any():
            raise ValueError(
                'Spectator set is empty; effective energy reduces to E(C). '
                'Use the model directly instead.'
            )

        full_e, full_f = self._evaluate(atoms, np.ones(len(atoms), dtype=bool))
        spec_e, spec_f = self._evaluate(atoms, spectator_mask)

        forces = full_f.copy()
        forces[spectator_mask] -= spec_f
        components = {'E_full': full_e, 'E_complement': spec_e}
        energy = full_e - spec_e

        if baseline_subtract:
            foc_e, foc_f = self._evaluate(atoms, focus_mask)
            forces[focus_mask] -= foc_f
            energy -= foc_e
            components['E_focus'] = foc_e

        return EffectiveEnergyResult(
            energy=float(energy),
            forces=forces,
            components=components,
        )

    # ---- internals ------------------------------------------------------

    def _evaluate(
        self, atoms: ase.Atoms, keep_mask: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Run the model on the sub-system defined by ``keep_mask``.

        Returns ``(energy, forces)`` with forces re-expanded to the full-atoms
        layout (zero on dropped atoms).
        """
        sub_atoms = atoms[keep_mask]
        config = data_module.config_from_atoms(sub_atoms)
        if self._z_table is not None:
            config.atomic_numbers = atomic_numbers_to_indices(
                config.atomic_numbers, self._z_table
            )

        graph = graph_from_configuration(config, cutoff=self.r_max)
        graph = self._pad(graph)

        out = self._predict(self.params, graph)
        energy = float(np.asarray(jax.lax.stop_gradient(out['energy']))[0])
        sub_forces = np.asarray(jax.lax.stop_gradient(out['forces']))[: len(sub_atoms)]

        full_forces = np.zeros((len(atoms), 3), dtype=sub_forces.dtype)
        full_forces[keep_mask] = sub_forces
        return energy, full_forces

    def _pad(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        n_node = int(np.asarray(graph.n_node).item()) + 1
        n_edge_real = int(np.asarray(graph.n_edge).item())
        # Round the edge budget up so the same jit cache entry serves nearby
        # geometries. Key by node count + n_graph so different sub-systems
        # share buckets where compatible.
        key = (n_node, self.pad_n_graph)
        budget = self._edge_budgets.get(key, 0)
        if budget < n_edge_real:
            budget = n_edge_real + max(n_edge_real // 10, 10)
            self._edge_budgets[key] = budget
        return jraph.pad_with_graphs(
            graph,
            n_node=n_node,
            n_edge=budget,
            n_graph=self.pad_n_graph,
        )


def effective_energy_calculator(
    model: Callable,
    params,
    r_max: float,
    *,
    atomic_numbers: Iterable[int] | None = None,
) -> EffectiveEnergy:
    """Functional shortcut to build an :class:`EffectiveEnergy` instance."""
    return EffectiveEnergy(
        model=model, params=params, r_max=r_max, atomic_numbers=atomic_numbers
    )
