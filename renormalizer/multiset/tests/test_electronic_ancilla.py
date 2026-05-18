import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.mps import MpDm, Mps
from renormalizer.multiset import ElectronicAncillaMultisetMps, MultisetMps
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multisetspectra import MultisetSpectraFiniteT
from renormalizer.utils import Quantity


def _small_model():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(0.2), 2)
    mol0 = Mol(Quantity(0.0), [ph], dipole=1.0)
    mol1 = Mol(Quantity(0.1), [ph], dipole=1.0)
    return HolsteinModel([mol0, mol1], np.array([[0.0, 0.05], [0.05, 0.0]]), 2)


def _electronic_hamiltonian(model):
    h_e = np.asarray(model.j_matrix, dtype=np.complex128).copy()
    for alpha, mol in enumerate(model.mol_list):
        h_e[alpha, alpha] = mol.elocalex + mol.e0
    return h_e


def test_diagonal_electronic_ancilla_initialization_recovers_diagonal_rho():
    probs = np.array([0.25, 0.75])
    job = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        thermal_init_method="imaginary_time_exact",
        max_bonddim=4,
        expand=False,
        electronic_ancilla=True,
        electronic_purification="diagonal",
        electronic_initial_distribution=probs,
    )

    _, ket = job.latest_mps
    assert isinstance(ket, ElectronicAncillaMultisetMps)
    assert np.allclose(ket.rho_el(), np.diag(probs))
    assert np.isclose(ket.total_norm(), 1.0)


def test_thermal_electronic_ancilla_initialization_recovers_thermal_rho():
    model = _small_model()
    beta = 1.7
    job = MultisetSpectraFiniteT(
        model=model,
        spectratype="abs",
        temperature=Quantity(298, "K"),
        thermal_init_method="imaginary_time_exact",
        max_bonddim=4,
        expand=False,
        electronic_ancilla=True,
        electronic_purification="thermal",
        electronic_beta=beta,
    )

    _, ket = job.latest_mps
    h_e = _electronic_hamiltonian(model)
    evals, evecs = np.linalg.eigh(h_e)
    weights = np.exp(-beta * evals)
    weights /= weights.sum()
    rho_expected = evecs @ np.diag(weights) @ evecs.conj().T

    assert np.allclose(ket.rho_el(), rho_expected)
    assert np.isclose(ket.total_norm(), 1.0)


def test_electronic_ancilla_trace_uses_only_matching_ancilla_index():
    ms_model = MultisetModel(_small_model(), max_bonddim=4, auto_init=False)
    dofs = ms_model.init_model.v_dofs
    phi0 = MpDm.from_mps(Mps.hartree_product_state(ms_model.init_model, condition={dofs[0]: 0}))
    phi1 = MpDm.from_mps(Mps.hartree_product_state(ms_model.init_model, condition={dofs[0]: 1}))

    state = ElectronicAncillaMultisetMps(
        ms_model.MsModel,
        n_phys=2,
        n_anc=2,
        init_model=ms_model.init_model,
        msmps=[phi0, phi1, phi1, phi0],
    )

    rho = state.rho_el()
    assert np.allclose(rho[0, 1], 0.0)
    assert np.allclose(state.e_occupations_multiset, [2.0, 2.0])


def test_multiset_spectra_finitet_without_electronic_ancilla_keeps_original_state_type():
    job = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        thermal_init_method="imaginary_time_exact",
        max_bonddim=4,
        expand=False,
        electronic_ancilla=False,
    )

    bra, ket = job.latest_mps
    assert isinstance(bra, MultisetMps)
    assert isinstance(ket, MultisetMps)
    assert not isinstance(ket, ElectronicAncillaMultisetMps)
    assert job.autocorr_components.shape == (1, ket.N_electron, ket.N_electron)


def test_electronic_ancilla_abs_evolves_one_step_and_preserves_trace():
    job = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        thermal_init_method="imaginary_time_exact",
        max_bonddim=4,
        expand=False,
        electronic_ancilla=True,
        electronic_purification="diagonal",
        electronic_initial_distribution=np.array([0.4, 0.6]),
    )

    job.evolve(evolve_dt=0.05, nsteps=1)
    _, ket = job.latest_mps

    assert isinstance(ket, ElectronicAncillaMultisetMps)
    assert np.isclose(np.trace(ket.rho_el()).real, 1.0)


def test_electronic_ancilla_abs_expand_initial_state():
    job = MultisetSpectraFiniteT(
        model=_small_model(),
        spectratype="abs",
        temperature=Quantity(298, "K"),
        thermal_init_method="imaginary_time_exact",
        max_bonddim=4,
        expand=True,
        electronic_ancilla=True,
        electronic_purification="thermal",
    )

    _, ket = job.latest_mps

    assert isinstance(ket, ElectronicAncillaMultisetMps)
    assert np.isclose(np.trace(ket.rho_el()).real, 1.0)
