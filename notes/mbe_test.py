"""Many-body decomposition test for the (4 NH3 + Cu+) complex.

Builds the complex from the PolyChem-sisnf YAML config, generates several
random geometries, runs the full Moebius-inversion MBE on a MACE-jax model,
and writes a figure per geometry showing the structure (3D scatter) alongside
all 1-body, 2-body, 3-body, 4-body and 5-body energy terms.

Run:
    /Users/dprelogo/Docs/code/polychem/PolyChem-sisnf/.venv/bin/python \\
        /Users/dprelogo/Docs/code/polychem/mace-jax/notes/mbe_test.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers as Z_OF
from ase.data import covalent_radii
from flax import nnx
from mace_jax.nnx_utils import wrap_bare_arrays
from mace_jax.tools import model_builder
from mace_jax.tools.bundle import _manual_replace, load_model_bundle

POLYCHEM = Path('/Users/dprelogo/Docs/code/polychem/PolyChem-sisnf')
sys.path.insert(0, str(POLYCHEM))
from polychem.config import ConfigLoader  # noqa: E402
from polychem.molecule import Complex  # noqa: E402

YAML_FILE = POLYCHEM / 'yaml/4_ammonia_Cu1.yaml'
MODEL_PATH = POLYCHEM / 'mace-models/mace-omol-0-jax.json'
OUT_DIR = Path(__file__).parent / 'mbe_figures'
N_GEOM = 4
SEED = 1234


# ------------------------------------------------------------------
# 1. Build the complex
# ------------------------------------------------------------------
def build_complex():
    loader = ConfigLoader(str(YAML_FILE))
    return Complex(
        [m for m in loader.create_complex()],
        coordinate_system='spherical',
        r_max=5.0,
        min_atom_distance=1.5,
    )


# ------------------------------------------------------------------
# 2. MACE-jax energy on an arbitrary atomic subsystem
# ------------------------------------------------------------------
class MaceEnergy:
    """Compute MACE-jax single-point energies for arbitrary subsystems.

    JIT compiles once per distinct atom count, so the 31 subsets of a
    5-molecule complex incur ~9 compiles total (one per distinct n_atoms).
    """

    def __init__(self, model_path: Path, dtype: str = 'float32'):
        self.dtype = jnp.float32 if dtype == 'float32' else jnp.float64
        np_dtype = np.float32 if dtype == 'float32' else np.float64
        self._np_dtype = np_dtype

        bundle = load_model_bundle(str(model_path), dtype)
        self.config = bundle.config
        module = model_builder._build_jax_model(self.config, rngs=nnx.Rngs(0))
        wrap_bare_arrays(module)
        graphdef, state = nnx.split(module)
        try:
            nnx.replace_by_pure_dict(state, bundle.params)
        except (ValueError, KeyError):
            _manual_replace(state, bundle.params)
        self._graphdef = graphdef
        self._state = state

        atomic_numbers_list = [int(z) for z in self.config['atomic_numbers']]
        self._z_to_idx = {z: i for i, z in enumerate(atomic_numbers_list)}
        self._num_species = len(atomic_numbers_list)
        self._has_cs = 'embedding_specs' in self.config

        # Cache of jit-compiled energy functions, keyed by (n_atoms, has_cs).
        self._cache: dict[tuple[int, bool], callable] = {}

    @staticmethod
    def _dense_edge_index(n: int) -> jnp.ndarray:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = ii != jj
        return jnp.array(np.stack([ii[mask], jj[mask]], axis=0), dtype=jnp.int32)

    def _get_energy_fn(self, n_atoms: int):
        key = (n_atoms, self._has_cs)
        if key in self._cache:
            return self._cache[key]

        graphdef = self._graphdef
        state = self._state
        edge_index = self._dense_edge_index(n_atoms)
        n_edges = int(edge_index.shape[1])

        def energy_fn(positions, species_idx, total_charge, total_spin):
            node_attrs = jax.nn.one_hot(species_idx, self._num_species).astype(
                self.dtype
            )
            data = {
                'positions': positions,
                'node_attrs': node_attrs,
                'node_attrs_index': species_idx,
                'edge_index': edge_index,
                'shifts': jnp.zeros((n_edges, 3), dtype=self.dtype),
                'unit_shifts': jnp.zeros((n_edges, 3), dtype=self.dtype),
                'batch': jnp.zeros(n_atoms, dtype=jnp.int32),
                'ptr': jnp.array([0, n_atoms], dtype=jnp.int32),
                'cell': jnp.eye(3, dtype=self.dtype).reshape(1, 3, 3),
            }
            if self._has_cs:
                data['total_charge'] = total_charge
                data['total_spin'] = total_spin
            out, _ = graphdef.apply(state)(data)
            return out['energy'][0]

        jit_fn = jax.jit(energy_fn)
        self._cache[key] = jit_fn
        return jit_fn

    def __call__(
        self, elements: list[str], positions: np.ndarray, total_charge: float
    ) -> float:
        n = len(elements)
        species_idx = jnp.array(
            [self._z_to_idx[Z_OF[el]] for el in elements], dtype=jnp.int32
        )
        positions = jnp.asarray(positions, dtype=self.dtype)
        tc = jnp.array([float(total_charge)], dtype=jnp.float32)
        ts = jnp.array([1.0], dtype=jnp.float32)
        fn = self._get_energy_fn(n)
        return float(fn(positions, species_idx, tc, ts))


# ------------------------------------------------------------------
# 3. Many-body expansion via Moebius inversion
# ------------------------------------------------------------------
def all_subsets(M: int):
    """Non-empty subsets of {0,...,M-1}, ordered by size then lexicographically."""
    return [
        tuple(s)
        for k in range(1, M + 1)
        for s in itertools.combinations(range(M), k)
    ]


def mbe_terms(energies: dict[tuple, float]):
    """Apply the Moebius identity to convert subset energies to n-body terms."""
    terms = {}
    for S in energies:
        n = len(S)
        total = 0.0
        for k in range(1, n + 1):
            sign = (-1) ** (n - k)
            for T in itertools.combinations(S, k):
                total += sign * energies[T]
        terms[S] = total
    return terms


# ------------------------------------------------------------------
# 4. Plotting
# ------------------------------------------------------------------
ELEMENT_COLORS = {
    'H': '#bbbbbb',
    'N': '#1f77b4',
    'Cu': '#d97f00',
}


def _short_label(S: tuple, mol_short: list[str]) -> str:
    return '+'.join(mol_short[i] for i in S)


def plot_geometry_and_decomposition(
    atoms_positions: np.ndarray,
    elements: list[str],
    fragment_index: np.ndarray,
    energies: dict[tuple, float],
    terms: dict[tuple, float],
    mol_short: list[str],
    M: int,
    title: str,
    savepath: Path,
    focus_set: set | None = None,
):
    """If focus_set is given, MBE bars whose subset does NOT intersect the
    focus set are greyed out (they would be cancelled in
    E_eff^F = E(C) - E(C\\F)) and the surviving sum is reported in the
    suptitle. With focus_set=None the figure is identical to the original.
    """
    focus = None if focus_set is None else set(int(f) for f in focus_set)
    DROPPED_COLOR = '#d0d0d0'
    DROPPED_TEXT = '#888888'

    def is_kept(S):
        if focus is None:
            return True
        return bool(set(S) & focus)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.6],
        height_ratios=[1.0, 2.4],
        wspace=0.20,
        hspace=0.25,
    )
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_one = fig.add_subplot(gs[0, 1])
    ax_high = fig.add_subplot(gs[1, 1])

    full = tuple(range(M))
    e_full = energies[full]
    sum_terms = sum(terms.values())

    # ---- 3D structure ----
    fragment_cmap = plt.get_cmap('tab10')
    for i, (pos, el) in enumerate(zip(atoms_positions, elements)):
        r = covalent_radii[Z_OF[el]]
        size = 350 * r * r
        ax3d.scatter(
            *pos,
            s=size,
            c=[fragment_cmap(fragment_index[i] % 10)],
            edgecolors='k',
            linewidths=0.6,
            depthshade=True,
        )
    # Bonds within each fragment (covalent-radius heuristic).
    for fid in range(M):
        mask = fragment_index == fid
        idxs = np.where(mask)[0]
        for a, b in itertools.combinations(idxs, 2):
            d = np.linalg.norm(atoms_positions[a] - atoms_positions[b])
            cutoff = 1.3 * (
                covalent_radii[Z_OF[elements[a]]]
                + covalent_radii[Z_OF[elements[b]]]
            )
            if d < cutoff:
                xs, ys, zs = zip(atoms_positions[a], atoms_positions[b])
                ax3d.plot(xs, ys, zs, color='k', lw=1.0, alpha=0.5)
    # Per-fragment legend
    for fid in range(M):
        ax3d.scatter(
            [], [], [], c=[fragment_cmap(fid % 10)],
            label=mol_short[fid], edgecolors='k', linewidths=0.6,
        )
    ax3d.legend(loc='upper left', fontsize=8, frameon=True)
    ax3d.set_title(f'{title}\n$E_\\mathrm{{full}}$ = {e_full:.4f} eV', fontsize=11)
    ax3d.set_xlabel('x [Å]')
    ax3d.set_ylabel('y [Å]')
    ax3d.set_zlabel('z [Å]')
    pos = atoms_positions
    ctr = pos.mean(axis=0)
    span = max(pos.max(axis=0) - pos.min(axis=0)) / 2 + 1.0
    ax3d.set_xlim(ctr[0] - span, ctr[0] + span)
    ax3d.set_ylim(ctr[1] - span, ctr[1] + span)
    ax3d.set_zlim(ctr[2] - span, ctr[2] + span)

    # ---- MBE bar charts: split 1-body and >=2-body for readability ----
    subsets = list(terms.keys())
    cmap = plt.get_cmap('viridis', M + 1)

    one_body = [S for S in subsets if len(S) == 1]
    high_body = [S for S in subsets if len(S) >= 2]

    # 1-body panel (atomic-ref scale). Values are large negative numbers
    # (atomic refs); use symlog so NH3 monomers and Cu+ both stay readable.
    labels1 = [_short_label(S, mol_short) for S in one_body]
    vals1 = [terms[S] for S in one_body]
    y1 = np.arange(len(labels1))
    kept_1 = [is_kept(S) for S in one_body]
    colors_1 = [cmap(1) if k else DROPPED_COLOR for k in kept_1]
    ax_one.barh(y1, vals1, color=colors_1, edgecolor='k', linewidth=0.4)
    ax_one.set_yticks(y1)
    ax_one.set_yticklabels(labels1, fontsize=9, family='monospace')
    for tick, k in zip(ax_one.get_yticklabels(), kept_1):
        if not k:
            tick.set_color(DROPPED_TEXT)
    ax_one.invert_yaxis()
    ax_one.set_xscale('symlog', linthresh=1.0)
    ax_one.axvline(0, color='k', lw=0.6)
    ax_one.set_xlabel(
        r'$\Delta E^{(1)}_S$  [eV]   (isolated-monomer energies, symlog)'
    )
    ax_one.set_title(
        f'1-body terms     $\\sum_\\alpha E^{{(\\mathrm{{iso}})}}_\\alpha$ = '
        f'{sum(vals1):.4f} eV',
        fontsize=10,
    )
    for yi, v, k in zip(y1, vals1, kept_1):
        ax_one.text(
            0, yi, f'  {v:,.2f} eV', va='center', ha='left',
            fontsize=8, color='black' if k else DROPPED_TEXT,
        )

    # >=2-body panel
    labels_h = [_short_label(S, mol_short) for S in high_body]
    vals_h = [terms[S] for S in high_body]
    sizes_h = [len(S) for S in high_body]
    kept_h = [is_kept(S) for S in high_body]
    colors_h = [cmap(s) if k else DROPPED_COLOR for s, k in zip(sizes_h, kept_h)]
    yh = np.arange(len(labels_h))
    ax_high.barh(yh, vals_h, color=colors_h, edgecolor='k', linewidth=0.4)
    ax_high.set_yticks(yh)
    ax_high.set_yticklabels(labels_h, fontsize=7, family='monospace')
    for tick, k in zip(ax_high.get_yticklabels(), kept_h):
        if not k:
            tick.set_color(DROPPED_TEXT)
    ax_high.invert_yaxis()
    ax_high.axvline(0, color='k', lw=0.6)
    ax_high.set_xlabel(r'$\Delta E^{(n)}_S$  [eV]   (interaction terms)')
    # Horizontal separators between orders
    cum = 0
    for n in range(2, M + 1):
        count = sum(1 for s in sizes_h if s == n)
        if cum > 0:
            ax_high.axhline(cum - 0.5, color='k', lw=0.5, ls='--', alpha=0.5)
        cum += count
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(n), label=f'{n}-body')
        for n in range(2, M + 1)
    ]
    sum_high = sum(vals_h)
    ax_high.set_title(
        rf'$n\geq 2$ terms     $\sum_{{n\geq 2}}\Delta E^{{(n)}}$ = '
        f'{sum_high:.4f} eV    (= '
        rf'$E_\mathrm{{full}}-\sum_\alpha E^{{(\mathrm{{iso}})}}_\alpha$ = '
        f'{e_full - sum(vals1):.4f} eV)',
        fontsize=10,
    )
    ax_high.legend(handles=handles, loc='lower right', fontsize=8, frameon=True)
    if vals_h:
        x_pad = 0.04 * max(abs(min(vals_h)), abs(max(vals_h)), 1e-3)
        for yi, v, k in zip(yh, vals_h, kept_h):
            ax_high.text(
                v + (x_pad if v >= 0 else -x_pad),
                yi,
                f'{v:+.3f}',
                va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=7,
                color='black' if k else DROPPED_TEXT,
            )
        # Pad x-limits so text labels are visible.
        lo, hi = min(vals_h), max(vals_h)
        span = hi - lo
        ax_high.set_xlim(lo - 0.18 * span, hi + 0.18 * span)

    if focus is None:
        suptitle = (
            f'Möbius MBE consistency: $\\sum_S \\Delta E^{{(|S|)}}_S$ = '
            f'{sum_terms:.4f} eV = $E_\\mathrm{{full}}$'
        )
    else:
        focus_label = '+'.join(mol_short[i] for i in sorted(focus))
        sum_kept = sum(terms[S] for S in subsets if is_kept(S))
        sum_dropped = sum(terms[S] for S in subsets if not is_kept(S))
        suptitle = (
            f'Focus $F=\\{{${focus_label}$\\}}$: \\;'
            f'$E_\\mathrm{{eff}}^F = \\sum_{{S\\cap F\\neq\\varnothing}}'
            f'\\Delta E^{{(|S|)}}_S = {sum_kept:.4f}$ eV    '
            f'(dropped spectator-only sum = {sum_dropped:.4f} eV)'
        )
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# 5. Driver
# ------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Building complex...')
    cmplx = build_complex()
    M = len(cmplx)
    mol_names = [m.name for m in cmplx]
    mol_short = [n.replace('ammonia_', 'NH3#').replace('copper', 'Cu') for n in mol_names]
    n_per_mol = [len(m.atoms) for m in cmplx]
    fragment_index = np.concatenate(
        [np.full(n, i, dtype=int) for i, n in enumerate(n_per_mol)]
    )
    print(f'  M={M} molecules, total atoms={fragment_index.size}, DOF={cmplx.size}')

    print('Loading MACE-jax model (this triggers initial JIT trace)...')
    energy_calc = MaceEnergy(MODEL_PATH, dtype='float32')

    ref_rot = {i: np.eye(3) for i in range(M)}
    rng = np.random.default_rng(SEED)
    subsets = all_subsets(M)

    print(f'Running MBE on {N_GEOM} random geometries...')
    for k in range(N_GEOM):
        x = rng.uniform(0.0, 1.0, cmplx.size)
        df = cmplx.geometry(
            x,
            fixation_type='com_frame',
            reference_rotation_dict=ref_rot,
            geometry_backend='numpy',
        )
        positions = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        elements = df['element'].tolist()
        atom_charges = df['initial charge'].to_numpy(dtype=np.float64)

        # Compute E(S) for every non-empty subset.
        energies = {}
        for S in subsets:
            atom_mask = np.isin(fragment_index, list(S))
            sub_elems = [elements[i] for i in np.where(atom_mask)[0]]
            sub_pos = positions[atom_mask]
            sub_charge = float(atom_charges[atom_mask].sum())
            energies[S] = energy_calc(sub_elems, sub_pos, sub_charge)

        terms = mbe_terms(energies)

        full = tuple(range(M))
        sum_terms = sum(terms.values())
        residual = energies[full] - sum_terms
        print(
            f'  geom {k}: E_full={energies[full]:.4f} eV, '
            f'sum(terms)={sum_terms:.4f} eV, '
            f'residual={residual:+.2e} eV'
        )

        savepath = OUT_DIR / f'geom_{k:02d}.png'
        plot_geometry_and_decomposition(
            positions,
            elements,
            fragment_index,
            energies,
            terms,
            mol_short,
            M,
            title=f'random geom #{k}  (seed={SEED})',
            savepath=savepath,
        )
        print(f'    -> {savepath}')

    print(f'\nFigures: {OUT_DIR}')


if __name__ == '__main__':
    main()
