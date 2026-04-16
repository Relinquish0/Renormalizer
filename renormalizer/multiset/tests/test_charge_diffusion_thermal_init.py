import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_tdjob import MultisetChargeDiffusionDynamics
from renormalizer.utils import Quantity


def _build_small_model():
    ph0 = Phonon.simple_phonon(Quantity(0.5), Quantity(0.1), 4)
    ph1 = Phonon.simple_phonon(Quantity(0.8), Quantity(0.2), 4)
    return HolsteinModel(
        [Mol(Quantity(0.0), [ph0]), Mol(Quantity(0.1), [ph1])],
        Quantity(0.05),
        4,
    )


def _build_uninitialized_job(method):
    ms_model = MultisetModel(
        _build_small_model(),
        max_bonddim=8,
        temperature=Quantity(300, "K"),
        method=method,
        auto_init=False,
    )
    job = MultisetChargeDiffusionDynamics.__new__(MultisetChargeDiffusionDynamics)
    job.ms_model = ms_model
    job.temperature = ms_model.temperature
    return job


def test_imaginary_time_propagate_matches_exact_product_thermal_state():
    job = _build_uninitialized_job("imaginary_time_propagate")

    exact = job.init_mp("imaginary_time_exact")
    propagated = job.init_mp("imaginary_time_propagate")

    overlap = exact.conj().dot(propagated)
    assert np.isclose(abs(overlap), 1.0)
