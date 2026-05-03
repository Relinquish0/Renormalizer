import logging

import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.mps import MpDm, Mps
from renormalizer.mps.backend import xp
import renormalizer.multiset.multiset_model as multiset_model
import renormalizer.multiset.multiset_tdjob as multiset_tdjob
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MultisetMps
from renormalizer.multiset.multiset_tdjob import MultisetChargeDiffusionDynamics
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveMethod, Quantity


def _build_small_model(displacements=(0.0, 0.0), electronic_coupling=0.0):
    ph0 = Phonon.simple_phonon(Quantity(0.5), Quantity(displacements[0]), 6)
    ph1 = Phonon.simple_phonon(Quantity(0.8), Quantity(displacements[1]), 6)
    return HolsteinModel(
        [Mol(Quantity(0.0), [ph0]), Mol(Quantity(0.1), [ph1])],
        np.array([[0.0, electronic_coupling], [electronic_coupling, 0.0]]),
        4,
    )


def _build_uninitialized_job(method, model=None, temperature=Quantity(300, "K"), max_bonddim=8):
    ms_model = MultisetModel(
        _build_small_model() if model is None else model,
        max_bonddim=max_bonddim,
        temperature=temperature,
        method=method,
        auto_init=False,
    )
    job = MultisetChargeDiffusionDynamics.__new__(MultisetChargeDiffusionDynamics)
    job.ms_model = ms_model
    job.temperature = ms_model.temperature
    job.initial_site = ms_model.N_electron // 2
    job.use_init_hint = True
    return job


def _build_charge_diffusion_job(model=None, max_bonddim=8):
    return MultisetChargeDiffusionDynamics(
        model=_build_small_model(electronic_coupling=0.02) if model is None else model,
        max_bonddim=max_bonddim,
        temperature=Quantity(0, "K"),
        stop_at_edge=False,
    )


def test_imaginary_time_propagate_returns_ground_state_thermal_mpdm():
    model = _build_small_model(electronic_coupling=0.01)
    job = _build_uninitialized_job("imaginary_time_propagate", model=model, max_bonddim=16)
    max_thermal_bonddim = max(MpDm.max_entangled_gs(job.ms_model.init_model).bond_dims)

    propagated = job.init_mp("imaginary_time_propagate")

    assert isinstance(propagated, MpDm)
    assert propagated.model.nsite == job.ms_model.init_model.nsite
    assert propagated.model.v_dofs == job.ms_model.init_model.v_dofs
    assert max(propagated.bond_dims) <= max_thermal_bonddim


def test_imaginary_time_propagate_uses_local_tdvp_not_multiset_evolution(monkeypatch):
    job = _build_uninitialized_job("imaginary_time_propagate", max_bonddim=64)
    nsteps = max(20, len(MpDm.max_entangled_gs(job.ms_model.init_model)))
    calls = {}

    class FakeThermalProp:
        def __init__(self, init_mpdm, h_mpo_model=None, evolve_config=None, auto_expand=True, **kwargs):
            calls["init_mpdm"] = init_mpdm
            calls["h_mpo_model"] = h_mpo_model
            calls["evolve_config"] = evolve_config
            calls["auto_expand"] = auto_expand
            self.latest_mps = init_mpdm.copy()

        def evolve(self, evolve_dt, nsteps_arg, total_time):
            calls["evolve"] = (evolve_dt, nsteps_arg, total_time)

    def fail_multiset_evolve(*args, **kwargs):
        raise AssertionError("imaginary-time thermalization should not use multiset evolve_state")

    monkeypatch.setattr(multiset_tdjob, "ThermalProp", FakeThermalProp, raising=False)
    monkeypatch.setattr(job.ms_model, "evolve_state", fail_multiset_evolve)

    propagated = job.init_mp("imaginary_time_propagate")

    assert isinstance(propagated, MpDm)
    assert calls["h_mpo_model"] is job.ms_model.init_model
    assert calls["evolve_config"].method is EvolveMethod.tdvp_ps
    assert calls["auto_expand"] is False
    assert calls["init_mpdm"].compress_config.criteria is CompressCriteria.fixed
    assert calls["init_mpdm"].compress_config.bond_dim_max_value == 1
    assert calls["evolve"] == (None, nsteps, job.temperature.to_beta() / 2j)


def test_imaginary_time_propagate_uses_bonddim_one_icompress_config(monkeypatch):
    job = _build_uninitialized_job("imaginary_time_propagate", max_bonddim=16)
    seen_config = {}

    class FakeThermalProp:
        def __init__(self, init_mpdm, **kwargs):
            seen_config["config"] = init_mpdm.compress_config
            self.latest_mps = init_mpdm.copy()

        def evolve(self, evolve_dt, nsteps, total_time):
            pass

    monkeypatch.setattr(multiset_tdjob, "ThermalProp", FakeThermalProp, raising=False)

    propagated = job.init_mp("imaginary_time_propagate")

    assert isinstance(propagated, MpDm)
    assert seen_config["config"].criteria is CompressCriteria.fixed
    assert seen_config["config"].bond_dim_max_value == 1


def test_imaginary_time_propagate_logs_phonon_occupations(monkeypatch, caplog):
    job = _build_uninitialized_job("imaginary_time_propagate", max_bonddim=16)

    class FakeThermalProp:
        def __init__(self, init_mpdm, **kwargs):
            self.latest_mps = init_mpdm.copy()

        def evolve(self, evolve_dt, nsteps, total_time):
            pass

    monkeypatch.setattr(multiset_tdjob, "ThermalProp", FakeThermalProp, raising=False)
    caplog.set_level(logging.INFO, logger="renormalizer.multiset.multiset_tdjob")

    job.init_mp("imaginary_time_propagate")

    assert any("[thermal init] imaginary-time" in message for message in caplog.messages)
    assert any("[thermal init] ph occupations:" in message for message in caplog.messages)


def test_batched_qr_qn_respects_max_rank():
    job = _build_uninitialized_job("imaginary_time_propagate", max_bonddim=16)
    job.ms_model.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=1)
    coef_batch = xp.arange(18, dtype=float).reshape(2, 3, 3)
    qnbigl = np.zeros((3, 1), dtype=int)
    qnbigr = np.zeros((3, 1), dtype=int)
    qntot = np.zeros(1, dtype=int)

    u_batch, qnlset, vt_batch, qnrset = job.ms_model._batched_qr_qn(
        coef_batch,
        qnbigl,
        qnbigr,
        qntot,
        "L",
        max_rank=1,
    )

    assert u_batch.shape[-1] == 1
    assert vt_batch.shape[1] == 1
    assert len(qnlset) == 1
    assert len(qnrset) == 1


def test_multiset_mps_calculates_electron_and_phonon_occupations():
    job = _build_uninitialized_job("imaginary_time_propagate")
    dofs = job.ms_model.init_model.v_dofs

    mps0 = MpDm.from_mps(
        Mps.hartree_product_state(
            job.ms_model.init_model,
            condition={dofs[0]: 1, dofs[1]: 0},
        )
    )
    mps1 = MpDm.from_mps(
        Mps.hartree_product_state(
            job.ms_model.init_model,
            condition={dofs[0]: 0, dofs[1]: 2},
        )
    )
    mps0.scale(np.sqrt(0.25), inplace=True)
    mps1.scale(np.sqrt(0.75), inplace=True)

    state = MultisetMps(
        job.ms_model.MsModel,
        job.ms_model.N_electron,
        init_model=job.ms_model.init_model,
        msmps=[mps0, mps1],
    )

    assert np.allclose(state.e_occupations_multiset, [0.25, 0.75])
    assert np.allclose(state.ph_occupations_multiset, [0.25, 1.5])


def test_multiset_model_reuses_environ_cache_on_second_timestep(monkeypatch):
    job = _build_charge_diffusion_job(max_bonddim=4)
    environ_init_count = 0
    original_init = multiset_model.Environ.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal environ_init_count
        environ_init_count += 1
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(multiset_model.Environ, "__init__", counting_init)

    job.evolve(evolve_dt=0.05, nsteps=1)
    first_step_inits = environ_init_count
    environ_init_count = 0

    job.evolve(evolve_dt=0.05, nsteps=1)

    assert first_step_inits > 0
    assert environ_init_count == 0


def test_multiset_model_environ_cache_matches_rebuild_path():
    model = _build_small_model(displacements=(0.1, 0.2), electronic_coupling=0.02)
    cached_job = _build_charge_diffusion_job(model=model, max_bonddim=4)
    rebuilt_job = _build_charge_diffusion_job(model=model, max_bonddim=4)
    rebuilt_job.ms_model._reuse_environ_cache = False

    for _ in range(2):
        cached_job.evolve(evolve_dt=0.05, nsteps=1)
        rebuilt_job.evolve(evolve_dt=0.05, nsteps=1)

    assert np.allclose(cached_job.latest_mps.rho_el(), rebuilt_job.latest_mps.rho_el())
    assert np.allclose(cached_job.e_occupations_array[-1], rebuilt_job.e_occupations_array[-1])
    assert [mps.bond_dims for mps in cached_job.latest_mps.msmps] == [
        mps.bond_dims for mps in rebuilt_job.latest_mps.msmps
    ]
