from pathlib import Path

import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset import MsEvolveMethod, MultisetSpectraZeroT
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, Quantity, constant


def construct_model(nmols) -> HolsteinModel:
    if isinstance(nmols, str):
        nmols_map = {"monomer": 1, "dimer": 2, "hexmer": 6}
        try:
            nmols = nmols_map[nmols.strip().lower()]
        except KeyError as exc:
            raise ValueError("nmols should be 1, 2, 6, or 'monomer', 'dimer', 'hexmer'") from exc

    elocalex = Quantity(2.13 / constant.au2ev)
    dipole_abs = 1.0

    omega_value = (
        np.array(
            [206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0]
        )
        * constant.cm2au
    )
    S_value = np.array(
        [0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039]
    )

    gw = np.sqrt(S_value) * omega_value
    idx = np.argsort(gw)[::-1]
    omega_value = omega_value[idx]
    S_value = S_value[idx]

    omega = [[Quantity(x), Quantity(x)] for x in omega_value]
    D_value = np.sqrt(S_value) / np.sqrt(omega_value / 2.0)
    displacement = [[Quantity(0), Quantity(x)] for x in D_value]
    ph_phys_dim = [5] * 10

    ph_list = [
        Phonon(*args[:10])
        for args in zip(omega, displacement, ph_phys_dim)
    ]

    return HolsteinModel([Mol(elocalex, ph_list, dipole_abs)] * nmols, Quantity(-500, "cm-1"))


def main():
    type_ = "dimer"
    spectratype = "emi"
    spectra_tag = "zt"
    max_bonddim = 4
    dump_dir = Path(__file__).resolve().parent
    job_name = f"pbi_{type_}_{spectra_tag}_{spectratype}_multiset"

    model = construct_model(type_)

    evolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps, adaptive=False)
    compress_config = CompressConfig(
        CompressCriteria.both, threshold=1e-8, max_bonddim=max_bonddim
    )

    spectra = MultisetSpectraZeroT(
        model=model,
        spectratype=spectratype,
        max_bonddim=max_bonddim,
        offset=Quantity(0),
        evolve_config=evolve_config,
        compress_config=compress_config,
        expand=True,
        dump_dir=dump_dir,
        job_name=job_name,
    )

    spectra.evolve(evolve_dt=20, nsteps=5000)


if __name__ == "__main__":
    main()
