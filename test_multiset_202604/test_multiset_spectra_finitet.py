import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.mps import MpDm
from renormalizer.multiset import MsEvolveMethod, MultisetMps, MultisetSpectraFiniteT
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
    constant,
)


def _small_model():
    omega = [[Quantity(500 * constant.cm2au), Quantity(500 * constant.cm2au)]]
    displacement = [[Quantity(0), Quantity(0.2)]]
    ph_list = [Phonon(omega[0], displacement[0], 3)]
    return HolsteinModel([Mol(Quantity(1.0, "eV"), ph_list, 1.0)] * 2, Quantity(0))


def _configs(max_bonddim=2):
    evolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps, adaptive=False)
    compress_config = CompressConfig(
        CompressCriteria.fixed, threshold=1e-10, max_bonddim=max_bonddim
    )
    return evolve_config, compress_config


def test_multiset_spectra_finitet_abs_exact_initializes_mpdm_components():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_exact",
        evolve_config=evolve_config,
        compress_config=compress_config,
    )

    bra, ket = spectra.latest_mps

    assert isinstance(bra, MultisetMps)
    assert isinstance(ket, MultisetMps)
    assert all(isinstance(mp, MpDm) for mp in ket.msmps)
    assert max(max(mp.bond_dims) for mp in ket.msmps) > 1
    assert np.asarray(spectra.autocorr).shape == (1,)


def test_multiset_spectra_finitet_emi_exact_initializes_ground_mpdm_pair():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="emi",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_exact",
        evolve_config=evolve_config,
        compress_config=compress_config,
    )

    bra, ket = spectra.latest_mps

    assert isinstance(bra, MpDm)
    assert isinstance(ket, MpDm)
    assert np.asarray(spectra.autocorr).shape == (1,)
    assert spectra.evolve_config.method is EvolveMethod.tdvp_ps
    assert ket.evolve_config.method is EvolveMethod.tdvp_ps
    assert max(ket.bond_dims) > 1


def test_multiset_spectra_finitet_abs_propagate_initializes_mpdm_components():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_propagate",
        insteps=1,
        evolve_config=evolve_config,
        compress_config=compress_config,
    )

    _, ket = spectra.latest_mps

    assert isinstance(ket, MultisetMps)
    assert all(isinstance(mp, MpDm) for mp in ket.msmps)
    assert spectra.local_evolve_config.method is EvolveMethod.tdvp_ps


def test_multiset_spectra_finitet_emi_propagate_initializes_ground_mpdm_pair():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="emi",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_propagate",
        insteps=1,
        evolve_config=evolve_config,
        compress_config=compress_config,
        ievolve_config=evolve_config,
    )

    bra, ket = spectra.latest_mps

    assert isinstance(bra, MpDm)
    assert isinstance(ket, MpDm)


def test_multiset_spectra_finitet_abs_evolves_one_step():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_exact",
        evolve_config=evolve_config,
        compress_config=compress_config,
    )

    spectra.evolve(evolve_dt=1, nsteps=1)

    assert np.asarray(spectra.autocorr).shape == (2,)


def test_multiset_spectra_finitet_does_not_auto_stop_on_small_correlation():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_exact",
        evolve_config=evolve_config,
        compress_config=compress_config,
    )
    spectra._autocorr = [1.0 + 0j] + [0.0 + 0j] * 10

    assert spectra.stop_evolve_criteria() is False


def test_multiset_spectra_finitet_emi_evolves_one_step():
    evolve_config, compress_config = _configs()
    spectra = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="emi",
        temperature=Quantity(298, "K"),
        max_bonddim=2,
        thermal_init_method="imaginary_time_exact",
        evolve_config=evolve_config,
        compress_config=compress_config,
    )

    spectra.evolve(evolve_dt=1, nsteps=1)

    assert np.asarray(spectra.autocorr).shape == (2,)
