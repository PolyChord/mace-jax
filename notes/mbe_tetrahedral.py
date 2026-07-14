"""[Cu(NH3)4]+ structured-geometry MBE tests.

(1) Tetrahedral: Cu at origin, four NH3 at the vertices of a regular
    tetrahedron with their lone pairs pointing at Cu. 1D scan over
    Cu-N distance, full MBE at the minimum.

(2) Two close + two far: two NH3 coordinated to Cu (axial, opposite sides)
    at the tetrahedral-optimum Cu-N distance; two NH3 forming an H-bonded
    dimer placed well away from Cu so they interact mostly with each other.

Outputs:
    notes/mbe_figures/geom_tet_scan.png   (E vs d_CuN scan)
    notes/mbe_figures/geom_tet.png        (MBE at the tetrahedral optimum)
    notes/mbe_figures/geom_2c2f.png       (MBE for the 2-close-2-far geometry)

Run:
    /Users/dprelogo/Docs/code/polychem/PolyChem-sisnf/.venv/bin/python \\
        /Users/dprelogo/Docs/code/polychem/mace-jax/notes/mbe_tetrahedral.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NOTES = Path(__file__).parent
sys.path.insert(0, str(NOTES))
from mbe_test import (  # noqa: E402
    MODEL_PATH,
    MaceEnergy,
    all_subsets,
    mbe_terms,
    plot_geometry_and_decomposition,
)

OUT_DIR = NOTES / 'mbe_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- NH3 internal geometry (matches PolyChem-sisnf zmat/ammonia.zmat) ----
NH_BOND = 1.0127         # Angstrom
HNH_ANGLE_DEG = 107.6432

# Half H-N-H angle implies the angle between N-H and the C3v axis (β):
#   cos(H-N-H) = cos^2 β + sin^2 β · cos(120°) = 1.5 cos^2 β - 0.5
hnh = np.deg2rad(HNH_ANGLE_DEG)
cos2_beta = (np.cos(hnh) + 0.5) / 1.5
COS_BETA = float(np.sqrt(cos2_beta))
SIN_BETA = float(np.sqrt(1.0 - cos2_beta))

# Tetrahedral unit vectors (mutual angle 109.47°). With C3v lone-pair pointing
# at Cu, each NH3's C3 axis is oriented OUTWARD along these directions.
TETRA_DIRS = (1.0 / np.sqrt(3.0)) * np.array([
    [+1.0, +1.0, +1.0],
    [+1.0, -1.0, -1.0],
    [-1.0, +1.0, -1.0],
    [-1.0, -1.0, +1.0],
])


def _rotation_v_to_u(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector v onto unit vector u."""
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)
    cos_t = float(np.dot(v, u))
    if cos_t > 1.0 - 1e-12:
        return np.eye(3)
    if cos_t < -1.0 + 1e-12:
        # 180° rotation about any axis perpendicular to v.
        seed = (
            np.array([1.0, 0.0, 0.0])
            if abs(v[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(v, seed)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(v, u)
    axis /= np.linalg.norm(axis)
    sin_t = float(np.linalg.norm(np.cross(v, u)))
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def _rotation_z_to_u(u: np.ndarray) -> np.ndarray:
    """Rotation matrix taking +z onto the unit vector u (kept for API)."""
    return _rotation_v_to_u(np.array([0.0, 0.0, 1.0]), u)


def _reference_nh3() -> np.ndarray:
    """Standard NH3 with N at origin, C3v axis along +z, H tripod at +z side."""
    h_z = NH_BOND * COS_BETA
    h_xy = NH_BOND * SIN_BETA
    H = np.stack(
        [
            np.array(
                [
                    h_xy * np.cos(2.0 * np.pi * k / 3.0),
                    h_xy * np.sin(2.0 * np.pi * k / 3.0),
                    h_z,
                ]
            )
            for k in range(3)
        ],
        axis=0,
    )
    return np.vstack([np.zeros(3), H])  # N, H, H, H


def make_tetrahedral_complex(d_CuN: float):
    """Return (positions, elements, fragment_index) for tetrahedral Cu(NH3)4+.

    Cu sits at the origin; four NH3 molecules are placed at the vertices
    of a tetrahedron, each with its C3v axis radial outward (lone pair
    pointing at Cu, H tripod on the far side).
    """
    nh3_ref = _reference_nh3()  # (4, 3): N, H, H, H along +z C3v
    positions = []
    elements = []
    fragment_index = []
    for fid, u in enumerate(TETRA_DIRS):
        u = np.asarray(u, dtype=float)
        u /= np.linalg.norm(u)
        R = _rotation_z_to_u(u)
        # The NH3 reference has its C3 axis on +z and the H tripod on +z.
        # We want the C3 axis pointing OUTWARD (along +u) so the lone pair
        # points back toward Cu.
        nh3 = (R @ nh3_ref.T).T
        # Translate so N is at d_CuN * u.
        nh3 = nh3 + d_CuN * u
        positions.append(nh3)
        elements.extend(['N', 'H', 'H', 'H'])
        fragment_index.extend([fid] * 4)
    # Cu at origin.
    positions.append(np.zeros((1, 3)))
    elements.append('Cu')
    fragment_index.append(4)

    positions = np.vstack(positions)
    elements = list(elements)
    fragment_index = np.asarray(fragment_index, dtype=int)
    return positions, elements, fragment_index


def total_energy_for_distance(
    energy_calc: MaceEnergy, d_CuN: float
) -> tuple[float, np.ndarray, list[str], np.ndarray]:
    positions, elements, fragment_index = make_tetrahedral_complex(d_CuN)
    # Total charge of the complex is +1 (only Cu carries it).
    e_full = energy_calc(elements, positions, total_charge=1.0)
    return e_full, positions, elements, fragment_index


def make_two_close_two_far_complex(
    d_CuN: float,
    far_offset: float = 9.0,
    d_NN: float = 3.40,
):
    """Cu + 2 axially coordinated NH3 + 2 H-bonded NH3 placed far from Cu.

    Layout
    ------
    - Cu at origin.
    - NH3#1 at (0, 0, +d_CuN); C3v axis along +z, lone pair toward Cu.
    - NH3#2 at (0, 0, -d_CuN); C3v axis along -z, lone pair toward Cu.
    - NH3#3 (acceptor) at (far_offset, +d_NN/2, 0); C3v axis along +y,
      lone pair toward -y (i.e. toward NH3#4).
    - NH3#4 (donor) at (far_offset, -d_NN/2, 0); C3v axis along +y,
      H tripod on +y side --- one of its H atoms points toward NH3#3.

    The far_offset moves the NH3#3-NH3#4 dimer well outside the
    receptive field around Cu so the only sizeable interactions are
    Cu<->NH3#1, Cu<->NH3#2 (close coordination) and NH3#3<->NH3#4
    (H-bonded dimer).
    """
    nh3_ref = _reference_nh3()  # (4, 3): N at origin, C3 axis along +z

    positions = []
    elements = []
    fragment_index = []

    # NH3#1: above Cu, lone pair at -z toward Cu, C3 axis along +z.
    R1 = _rotation_z_to_u(np.array([0.0, 0.0, +1.0]))
    nh3_1 = (R1 @ nh3_ref.T).T + np.array([0.0, 0.0, +d_CuN])
    # NH3#2: below Cu, lone pair at +z toward Cu, C3 axis along -z.
    R2 = _rotation_z_to_u(np.array([0.0, 0.0, -1.0]))
    nh3_2 = (R2 @ nh3_ref.T).T + np.array([0.0, 0.0, -d_CuN])

    # NH3#3 (acceptor): at +y of the far cluster. C3 axis along +y so its
    # lone pair (-z in ref -> -y in lab) points at NH3#4 (which is at -y).
    R3 = _rotation_v_to_u(
        np.array([0.0, 0.0, 1.0]),  # ref +z
        np.array([0.0, 1.0, 0.0]),  # lab +y (C3 outward, lone pair -y)
    )
    nh3_3 = (R3 @ nh3_ref.T).T + np.array([far_offset, +d_NN / 2.0, 0.0])

    # NH3#4 (donor): at -y of the far cluster. We want one of its H atoms
    # (H_0 in the reference frame) to point along +y, directly at the
    # acceptor N. H_0 in the reference is at (sin β, 0, cos β); rotate so
    # this direction maps to +y. The other two H atoms swing into a proper
    # H-bond geometry as a side effect.
    h0_dir_ref = np.array([SIN_BETA, 0.0, COS_BETA])
    R4 = _rotation_v_to_u(h0_dir_ref, np.array([0.0, 1.0, 0.0]))
    nh3_4 = (R4 @ nh3_ref.T).T + np.array([far_offset, -d_NN / 2.0, 0.0])

    for fid, frag in enumerate([nh3_1, nh3_2, nh3_3, nh3_4]):
        positions.append(frag)
        elements.extend(['N', 'H', 'H', 'H'])
        fragment_index.extend([fid] * 4)

    # Cu at origin.
    positions.append(np.zeros((1, 3)))
    elements.append('Cu')
    fragment_index.append(4)

    return (
        np.vstack(positions),
        list(elements),
        np.asarray(fragment_index, dtype=int),
    )


def main():
    print('Loading MACE-jax model...')
    energy_calc = MaceEnergy(MODEL_PATH, dtype='float32')

    # ---- 1. Coarse + fine 1D scan over Cu-N distance ----
    coarse_grid = np.linspace(1.6, 3.5, 20)
    print(f'Coarse Cu-N scan over {len(coarse_grid)} points...')
    coarse_E = np.array([
        total_energy_for_distance(energy_calc, d)[0] for d in coarse_grid
    ])
    i_min_coarse = int(np.argmin(coarse_E))
    d_coarse = float(coarse_grid[i_min_coarse])
    print(f'  coarse minimum at d_CuN = {d_coarse:.3f} Å, '
          f'E = {coarse_E[i_min_coarse]:.4f} eV')

    fine_grid = np.linspace(
        max(1.5, d_coarse - 0.25), d_coarse + 0.25, 41
    )
    print(f'Fine Cu-N scan over {len(fine_grid)} points...')
    fine_E = np.array([
        total_energy_for_distance(energy_calc, d)[0] for d in fine_grid
    ])
    i_min = int(np.argmin(fine_E))
    d_star = float(fine_grid[i_min])
    e_star = float(fine_E[i_min])
    print(f'  fine minimum at d_CuN* = {d_star:.4f} Å, E* = {e_star:.4f} eV')

    # Concatenate scan data for plot.
    grid = np.concatenate([coarse_grid, fine_grid])
    E_grid = np.concatenate([coarse_E, fine_E])
    order = np.argsort(grid)
    grid = grid[order]
    E_grid = E_grid[order]

    # ---- 2. Plot the 1D scan ----
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(grid, E_grid - e_star, '-', color='C0', alpha=0.4, lw=1.0)
    ax.scatter(coarse_grid, coarse_E - e_star, s=22,
               edgecolors='C0', facecolors='white', zorder=3, label='coarse scan')
    ax.scatter(fine_grid, fine_E - e_star, s=14, color='C0', zorder=3,
               label='fine scan')
    ax.axvline(d_star, ls='--', color='k', lw=0.6)
    ax.scatter([d_star], [0.0], s=80, color='red', marker='*', zorder=5,
               label=fr'minimum $d^*={d_star:.3f}\,$Å')
    ax.set_xlabel(r'Cu--N distance  $d_{\mathrm{Cu-N}}$  [Å]')
    ax.set_ylabel(r'$E(d) - E(d^*)$  [eV]')
    ax.set_title(
        rf'Tetrahedral [Cu(NH$_3$)$_4$]$^+$:  '
        rf'1D scan of Cu--N distance   ($E^* = {e_star:.4f}\,$eV)'
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scan_path = OUT_DIR / 'geom_tet_scan.png'
    fig.savefig(scan_path, dpi=140)
    plt.close(fig)
    print(f'  wrote {scan_path}')

    # ---- 3. Full MBE on the optimum geometry ----
    positions, elements, fragment_index = make_tetrahedral_complex(d_star)
    M = int(fragment_index.max() + 1)
    mol_short = [f'NH3#{i+1}' for i in range(M - 1)] + ['Cu']
    subsets = all_subsets(M)

    # Per-atom charges: only Cu has +1.
    atom_charges = np.zeros(len(elements), dtype=float)
    atom_charges[-1] = 1.0  # Cu is the last atom.

    print('Running MBE on the tetrahedral optimum...')
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
        f'  E_full={energies[full]:.4f} eV, '
        f'sum(terms)={sum_terms:.4f} eV, residual={residual:+.2e} eV'
    )

    fig_path = OUT_DIR / 'geom_tet.png'
    plot_geometry_and_decomposition(
        positions,
        elements,
        fragment_index,
        energies,
        terms,
        mol_short,
        M,
        title=fr'tetrahedral [Cu(NH$_3$)$_4$]$^+$  (d_Cu-N = {d_star:.3f} Å)',
        savepath=fig_path,
    )
    print(f'  wrote {fig_path}')

    # Focus-set version: F = {Cu} (fragment id M-1).
    fig_path_focus = OUT_DIR / 'geom_tet_focus.png'
    plot_geometry_and_decomposition(
        positions,
        elements,
        fragment_index,
        energies,
        terms,
        mol_short,
        M,
        title=(
            r'tetrahedral [Cu(NH$_3$)$_4$]$^+$ — focus $F=\{$Cu$\}$ '
            r'(grey bars cancel in $E(\mathcal{C})-E(\mathcal{C}\setminus F)$)'
        ),
        savepath=fig_path_focus,
        focus_set={M - 1},
    )
    print(f'  wrote {fig_path_focus}')

    # ---- 4. Two-close + two-far geometry, MBE on it ----
    d_NN = 3.40
    far_offset = 9.0
    print('\nBuilding two-close-two-far geometry '
          f'(d_CuN={d_star:.3f} Å, dimer N-N={d_NN:.2f} Å, '
          f'far offset={far_offset:.1f} Å)...')
    pos_2c2f, el_2c2f, frag_2c2f = make_two_close_two_far_complex(
        d_CuN=d_star, far_offset=far_offset, d_NN=d_NN
    )
    M2 = int(frag_2c2f.max() + 1)
    mol_short_2c2f = ['NH3#1c', 'NH3#2c', 'NH3#3f', 'NH3#4f', 'Cu']

    atom_charges_2c2f = np.zeros(len(el_2c2f), dtype=float)
    atom_charges_2c2f[-1] = 1.0
    subsets_2c2f = all_subsets(M2)

    print('Running MBE on two-close-two-far geometry...')
    energies_2 = {}
    for S in subsets_2c2f:
        atom_mask = np.isin(frag_2c2f, list(S))
        sub_elems = [el_2c2f[i] for i in np.where(atom_mask)[0]]
        sub_pos = pos_2c2f[atom_mask]
        sub_charge = float(atom_charges_2c2f[atom_mask].sum())
        energies_2[S] = energy_calc(sub_elems, sub_pos, sub_charge)
    terms_2 = mbe_terms(energies_2)

    full_2 = tuple(range(M2))
    sum_terms_2 = sum(terms_2.values())
    residual_2 = energies_2[full_2] - sum_terms_2
    print(
        f'  E_full={energies_2[full_2]:.4f} eV, '
        f'sum(terms)={sum_terms_2:.4f} eV, residual={residual_2:+.2e} eV'
    )

    fig_path_2 = OUT_DIR / 'geom_2c2f.png'
    plot_geometry_and_decomposition(
        pos_2c2f,
        el_2c2f,
        frag_2c2f,
        energies_2,
        terms_2,
        mol_short_2c2f,
        M2,
        title=(
            r'two close + two far:  NH$_3$#1c, NH$_3$#2c axial on Cu;  '
            r'NH$_3$#3f$\cdots$NH$_3$#4f H-bond dimer @ '
            f'{far_offset:.0f} Å'
        ),
        savepath=fig_path_2,
    )
    print(f'  wrote {fig_path_2}')

    # Focus-set version: F = {Cu}.
    fig_path_2_focus = OUT_DIR / 'geom_2c2f_focus.png'
    plot_geometry_and_decomposition(
        pos_2c2f,
        el_2c2f,
        frag_2c2f,
        energies_2,
        terms_2,
        mol_short_2c2f,
        M2,
        title=(
            r'two close + two far — focus $F=\{$Cu$\}$ '
            r'(grey bars are dropped from $E_\mathrm{eff}^F$;'
            r' the NH$_3$#3f$\cdots$#4f H-bond is in there)'
        ),
        savepath=fig_path_2_focus,
        focus_set={M2 - 1},
    )
    print(f'  wrote {fig_path_2_focus}')


if __name__ == '__main__':
    main()
