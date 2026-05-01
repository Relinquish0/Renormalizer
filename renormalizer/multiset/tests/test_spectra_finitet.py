import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset import MultisetMps
from renormalizer.multiset.multisetspectra import MultisetSpectraFiniteT
from renormalizer.utils import Quantity


def _build_small_model():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph], dipole=1.0)
    return HolsteinModel([mol] * 2, Quantity(0.1), 2)


def test_multiset_spectra_finitet_emi_uses_multiset_transition_operators():
    job = MultisetSpectraFiniteT(
        model=_build_small_model(),
        spectratype="emi",
        temperature=Quantity(298, "K"),
        insteps=4,
        thermal_init_method="imaginary_time_propagate",
        max_bonddim=4,
        expand=False,
    )
    bra, ket = job.latest_mps

    assert isinstance(bra, MultisetMps)
    assert isinstance(ket, MultisetMps)
    assert np.isfinite(job.autocorr[0].real)
    assert job.autocorr[0].real < 4.0


def test_multiset_spectra_finitet_emi_one_step_keeps_multiset_branch_types():
    job = MultisetSpectraFiniteT(
        model=_build_small_model(),
        spectratype="emi",
        temperature=Quantity(298, "K"),
        insteps=4,
        thermal_init_method="imaginary_time_propagate",
        max_bonddim=4,
        expand=False,
    )

    job.evolve(evolve_dt=0.05, nsteps=1)
    bra, ket = job.latest_mps

    assert isinstance(bra, MultisetMps)
    assert isinstance(ket, MultisetMps)
    assert len(job.autocorr) == 2
    assert np.isfinite(job.autocorr[-1].real)
    assert not np.allclose(job.autocorr[-1], job.autocorr[0])
