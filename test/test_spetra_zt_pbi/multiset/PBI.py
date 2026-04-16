import numpy as np

from renormalizer.utils import EvolveConfig, Quantity, constant, CompressConfig, CompressCriteria
from renormalizer.model import HolsteinModel, Mol, Phonon

from renormalizer.multiset import MsEvolveMethod, MultisetSpectraZeroT

def construct_model(nmols) -> HolsteinModel:
    if isinstance(nmols, str):
        nmols_map = {"monomer": 1, "dimer": 2, "hexmer": 6}
        try:
            nmols = nmols_map[nmols.strip().lower()]
        except KeyError as exc:
            raise ValueError("nmols should be 1, 2, 6, or 'monomer', 'dimer', 'hexmer'") from exc

    elocalex = Quantity(2.13 / constant.au2ev)
    dipole_abs = 1.0

    # cm^-1
    omega_value = (
        np.array(
            [206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0]
        )
        * constant.cm2au
    )
    S_value = np.array(
        [0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039]
    )

    # sort from large to small
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

    model = HolsteinModel([Mol(elocalex, ph_list, dipole_abs)] * nmols, Quantity(-500, "cm-1"), )

    return model

def save_zerot_data(filename, time_series, autocorr):
    np.savez(
        filename,
        time_series=time_series,
        autocorr=autocorr,
    )


def main():
    # "monomer": 1, "dimer": 2, "hexmer": 6
    type_ = "hexmer"

    model = construct_model(type_)

    max_bonddim = 42
    evolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps, adaptive=False)
    compress_config = CompressConfig(CompressCriteria.both, threshold=1e-8, max_bonddim=max_bonddim)

    spectra = MultisetSpectraZeroT(
        model=model,
        max_bonddim=max_bonddim,
        evolve_config=evolve_config,
        compress_config=compress_config,
        expand=True
    )

    spectra.evolve(evolve_dt=100, nsteps=4000)

    autocorr = spectra.autocorr
    time_points = spectra.evolve_times_array

    save_zerot_data(
        filename="pbi_{}_zt_expand.npz".format(type_),
        time_series=time_points,
        autocorr=autocorr,
    )


if __name__ == "__main__":
    main()
