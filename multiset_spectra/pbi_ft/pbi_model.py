from __future__ import annotations

import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.utils import Quantity, constant


PBI_MODEL_SIZES = {
    "monomer": 1,
    "dimer": 2,
    "trimer": 3,
    "hexamer": 6,
    "hexmer": 6,
}


def normalize_model_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in PBI_MODEL_SIZES:
        choices = ", ".join(sorted(PBI_MODEL_SIZES))
        raise ValueError(f"Unknown PBI aggregate {name!r}. Choose one of: {choices}")
    return "hexamer" if normalized == "hexmer" else normalized


def display_model_name(name: str) -> str:
    return normalize_model_name(name).capitalize()


def construct_pbi_model(name: str) -> HolsteinModel:
    nmols = PBI_MODEL_SIZES[normalize_model_name(name)]

    elocalex = Quantity(2.13 / constant.au2ev)
    dipole_abs = 1.0

    omega_value = (
        np.array(
            [206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0]
        )
        * constant.cm2au
    )
    s_value = np.array(
        [0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039]
    )

    # Keep the largest vibronic couplings first, matching the older PBI scripts.
    vibronic_coupling = np.sqrt(s_value) * omega_value
    sort_idx = np.argsort(vibronic_coupling)[::-1]
    omega_value = omega_value[sort_idx]
    s_value = s_value[sort_idx]

    omega = [[Quantity(x), Quantity(x)] for x in omega_value]
    displacement_value = np.sqrt(s_value) / np.sqrt(omega_value / 2.0)
    displacement = [[Quantity(0), Quantity(x)] for x in displacement_value]
    ph_phys_dim = [5] * len(omega_value)
    ph_list = [
        Phonon(omega_i, displacement_i, phys_dim)
        for omega_i, displacement_i, phys_dim in zip(omega, displacement, ph_phys_dim)
    ]

    mols = [Mol(elocalex, ph_list, dipole_abs) for _ in range(nmols)]
    return HolsteinModel(mols, Quantity(-500, "cm-1"))
