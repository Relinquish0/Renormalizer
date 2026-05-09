import numpy as np
import pytest

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset import MsEvolveMethod, MultisetMps, MultisetSpectraZeroT as ExportedMultisetSpectraZeroT
from renormalizer.multiset.multisetspectra import MultisetSpectraZeroT
from renormalizer.utils import Quantity


def _build_small_model():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph], dipole=1.0)
    return HolsteinModel([mol] * 2, Quantity(0.1), 2)


def test_multiset_spectra_zerot_exported_from_package():
    assert ExportedMultisetSpectraZeroT is MultisetSpectraZeroT


def test_multiset_spectra_zerot_bootstrap_autocorr():
    job = MultisetSpectraZeroT(
        model=_build_small_model(),
        max_bonddim=4,
    )
    bra, ket = job.latest_mps

    assert len(job.autocorr) == 1
    assert job.autocorr_components.shape == (1, bra.N_electron, ket.N_electron)
    assert bra.N_electron == ket.N_electron == job.model.n_edofs
    assert all(
        [site.shape for site in bra.msmps[a]] == [site.shape for site in ket.msmps[a]]
        for a in range(ket.N_electron)
    )
    assert np.isfinite(job.autocorr[0].real)
    assert np.allclose(job.autocorr[0], np.trace(job.autocorr_components[0]))


def test_multiset_spectra_zerot_emi_stays_in_multiset_container():
    job = MultisetSpectraZeroT(
        model=_build_small_model(),
        spectratype="emi",
        max_bonddim=4,
    )
    bra, ket = job.latest_mps

    assert isinstance(bra, MultisetMps)
    assert isinstance(ket, MultisetMps)
    assert job.evolve_config.method is MsEvolveMethod.ms_evolve_tdvp_ps
    assert np.isclose(job.autocorr[0].real, 4.0)
    assert np.allclose(job.autocorr[0], job.autocorr_components[0].sum())


@pytest.mark.parametrize("spectratype", ["abs", "emi"])
def test_multiset_spectra_zerot_channel_norms_follow_dipoles_before_normalization(spectratype):
    model = _build_small_model()
    keys = list(model.dipole)
    model.dipole = {keys[0]: 1.0, keys[1]: 2.0}
    job = MultisetSpectraZeroT(model=model, spectratype=spectratype, max_bonddim=4)

    ket = job._init_dipole()
    channel_norms = [
        ket.msmps[a].conj().dot(ket.msmps[a]).real
        for a in range(ket.N_electron)
    ]

    assert np.isclose(channel_norms[1], 4.0 * channel_norms[0])


def test_multiset_spectra_zerot_one_step_updates_autocorr():
    job = MultisetSpectraZeroT(
        model=_build_small_model(),
        max_bonddim=4,
    )

    bra0, ket0 = job.latest_mps
    job.evolve(evolve_dt=0.05, nsteps=1)
    bra1, ket1 = job.latest_mps

    assert len(job.autocorr) == 2
    assert job.autocorr_components.shape == (2, bra0.N_electron, ket0.N_electron)
    assert np.isfinite(job.autocorr[-1].real)
    assert np.allclose(
        [bra0.msmps[a].conj().dot(bra1.msmps[a]) for a in range(bra0.N_electron)],
        [bra0.msmps[a].conj().dot(bra0.msmps[a]) for a in range(bra1.N_electron)],
    )
    assert not np.allclose(job.autocorr[-1], job.autocorr[0])
    assert any(
        not np.allclose(
            ket0.msmps[a].conj().dot(ket1.msmps[a]),
            ket0.msmps[a].conj().dot(ket0.msmps[a]),
        )
        for a in range(ket0.N_electron)
    )


def test_multiset_spectra_zerot_dipole_changes_t0_autocorr():
    model1 = _build_small_model()
    model2 = _build_small_model()
    model2.dipole = {key: value * 2.0 for key, value in model2.dipole.items()}

    job1 = MultisetSpectraZeroT(model=model1, max_bonddim=4)
    job2 = MultisetSpectraZeroT(model=model2, max_bonddim=4)

    assert not np.allclose(job1.autocorr[0], job2.autocorr[0])
    assert np.isclose(job2.autocorr[0].real, 4.0 * job1.autocorr[0].real)


def test_multiset_spectra_zerot_requires_model_and_max_bonddim():
    with pytest.raises(ValueError, match="model.*max_bonddim"):
        MultisetSpectraZeroT(model=_build_small_model())


def test_multiset_spectra_zerot_dump_dict_contains_component_autocorr():
    job = MultisetSpectraZeroT(
        model=_build_small_model(),
        max_bonddim=4,
    )

    dump_dict = job.get_dump_dict()

    assert "autocorr_components" in dump_dict
    assert dump_dict["autocorr_components"].shape == (
        len(job.autocorr),
        job.model.n_edofs,
        job.model.n_edofs,
    )
