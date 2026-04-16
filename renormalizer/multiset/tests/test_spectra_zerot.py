import numpy as np
import pytest

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset import MultisetSpectraZeroT as ExportedMultisetSpectraZeroT
from renormalizer.multiset.multiset_model import MultisetModel
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
    assert bra.N_electron == ket.N_electron == job.model.n_edofs
    assert all(
        [site.shape for site in bra.msmps[a]] == [site.shape for site in ket.msmps[a]]
        for a in range(ket.N_electron)
    )
    assert np.isfinite(job.autocorr[0].real)


def test_multiset_spectra_zerot_channel_norms_follow_dipoles_before_normalization():
    model = _build_small_model()
    keys = list(model.dipole)
    model.dipole = {keys[0]: 1.0, keys[1]: 2.0}
    job = MultisetSpectraZeroT(model=model, max_bonddim=4)

    ket = job._build_absorption_ket()
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


def test_multiset_spectra_zerot_rejects_finite_temperature_ms_model():
    ms_model = MultisetModel(
        _build_small_model(),
        max_bonddim=4,
        temperature=Quantity(300, "K"),
        auto_init=False,
    )

    with pytest.raises(ValueError, match="zero-temperature"):
        MultisetSpectraZeroT(ms_model=ms_model)


def test_multiset_spectra_zerot_ms_model_path_matches_model_path():
    model = _build_small_model()
    job_from_model = MultisetSpectraZeroT(model=model, max_bonddim=4)
    ms_model = MultisetModel(model, max_bonddim=4)
    job_from_ms_model = MultisetSpectraZeroT(ms_model=ms_model)

    assert np.allclose(job_from_model.autocorr[0], job_from_ms_model.autocorr[0])

    job_from_model.evolve(evolve_dt=0.05, nsteps=1)
    job_from_ms_model.evolve(evolve_dt=0.05, nsteps=1)

    assert np.allclose(job_from_model.autocorr[-1], job_from_ms_model.autocorr[-1])
