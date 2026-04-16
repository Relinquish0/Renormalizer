# Multiset Zero-Temperature Absorption Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MultisetSpectraZeroT`, a zero-temperature absorption autocorrelation driver for `renormalizer/multiset` that prepares a dipole-weighted multiset initial state and propagates it with the existing multiset TDVP kernel.

**Architecture:** Keep the current multiset ansatz unchanged. Implement a new job class in `renormalizer/multiset/spectra.py` that inherits from `MultisetTdJob`, uses `Model.dipole` to build the absorption initial state in `init_mps()`, and records the multiset overlap as `autocorr`. Do not add a zero-exciton channel and do not define a fake internal multiset `a^\dagger`.

**Tech Stack:** Python, pytest, existing `renormalizer.multiset` classes (`MultisetTdJob`, `MultisetModel`, `MultisetMps`), existing `renormalizer.utils.EvolveConfig`

---

### Task 1: Add the failing multiset spectra tests

**Files:**
- Create: `renormalizer/multiset/tests/test_spectra_zerot.py`
- Test: `renormalizer/multiset/tests/test_spectra_zerot.py`

- [ ] **Step 1: Write the failing test file**

```python
import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.multiset.multisetspectra import MultisetSpectraZeroT
from renormalizer.utils import Quantity


def _build_small_model():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph], dipole_abs=1.0)
    return HolsteinModel([mol] * 2, Quantity(0.1), 2)


def test_multiset_spectra_zerot_bootstrap_autocorr():
    job = MultisetSpectraZeroT(
        model=_build_small_model(),
        max_bonddim=4,
    )

    assert len(job.autocorr) == 1
    assert np.isfinite(job.autocorr[0].real)


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
        [bra0.msmps[a].conj().dot(bra0.msmps[a]) for a in range(bra0.N_electron)],
        [bra1.msmps[a].conj().dot(bra1.msmps[a]) for a in range(bra1.N_electron)],
    )
    assert not np.allclose(job.autocorr[-1], job.autocorr[0])
    assert any(
        not np.allclose(
            ket0.msmps[a].conj().dot(ket0.msmps[a]),
            ket1.msmps[a].conj().dot(ket1.msmps[a]),
        )
        for a in range(ket0.N_electron)
    )


def test_multiset_spectra_zerot_dipole_changes_t0_autocorr():
    model1 = _build_small_model()
    model2 = _build_small_model()
    model2.dipole = np.array(model2.dipole) * 2.0

    job1 = MultisetSpectraZeroT(model=model1, max_bonddim=4)
    job2 = MultisetSpectraZeroT(model=model2, max_bonddim=4)

    assert not np.allclose(job1.autocorr[0], job2.autocorr[0])
```

- [ ] **Step 2: Run the test file to verify it fails because the class does not exist yet**

Run:

```bash
pytest renormalizer/multiset/tests/test_spectra_zerot.py -v
```

Expected:

```text
E   ModuleNotFoundError: No module named 'renormalizer.multiset.multisetspectra'
```

- [ ] **Step 3: Commit the failing tests**

```bash
git add renormalizer/multiset/tests/test_spectra_zerot.py
git commit -m "test: add failing multiset zero-temperature spectra tests"
```

### Task 2: Implement `MultisetSpectraZeroT`

**Files:**
- Create: `renormalizer/multiset/spectra.py`
- Test: `renormalizer/multiset/tests/test_spectra_zerot.py`

- [ ] **Step 1: Create the spectra module with the class skeleton and helpers**

```python
import numpy as np

from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MultisetMps
from renormalizer.multiset.multiset_tdjob import MultisetTdJob
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity


def _multiset_overlap(bra: MultisetMps, ket: MultisetMps) -> complex:
    total = 0j
    for alpha in range(bra.N_electron):
        total += bra.msmps[alpha].conj().dot(ket.msmps[alpha])
    return complex(total)


class MultisetSpectraZeroT(MultisetTdJob):
    def __init__(
        self,
        model=None,
        max_bonddim=None,
        ms_model: MultisetModel = None,
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
    ):
        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            ms_model = MultisetModel(
                model,
                max_bonddim=max_bonddim,
                temperature=Quantity(0, "K"),
                evolve_config=evolve_config,
                compress_config=compress_config,
                auto_init=False,
            )
        else:
            if evolve_config is not None:
                ms_model.evolve_config = evolve_config
            if compress_config is not None:
                ms_model.compress_config = compress_config

        self.ms_model = ms_model
        self.model = self.ms_model.model
        self.temperature = Quantity(0, "K")
        self._autocorr = []

        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )
```

- [ ] **Step 2: Implement dipole extraction and dipole-weighted initial-state construction**

```python
    def _get_abs_dipole_vector(self) -> np.ndarray:
        dipole = getattr(self.model, "dipole", None)
        if dipole is None:
            raise ValueError("`model.dipole` is required for MultisetSpectraZeroT.")

        dipole = np.asarray(dipole, dtype=float)
        if dipole.ndim == 0:
            dipole = np.repeat(dipole, self.model.n_edofs)
        elif dipole.ndim > 1:
            dipole = dipole.reshape(-1)

        if len(dipole) != self.model.n_edofs:
            raise ValueError(
                f"`model.dipole` must resolve to one absorption coefficient per electronic excitation. "
                f"Expected {self.model.n_edofs}, got {len(dipole)}."
            )
        return dipole

    def _build_absorption_ket(self) -> MultisetMps:
        template = MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=Quantity(0, "K"),
            init_model=self.ms_model.init_model,
            method=self.ms_model.method,
        )
        phi_g = template.init_mp(method=template.method)
        dipole = self._get_abs_dipole_vector()

        ket = MultisetMps.__new__(MultisetMps)
        ket.MsModel = self.ms_model.MsModel
        ket.N_electron = self.ms_model.N_electron
        ket.temperature = Quantity(0, "K")
        ket.init_model = self.ms_model.init_model
        ket.method = self.ms_model.method
        ket.msmps = []
        for alpha in range(ket.N_electron):
            state = phi_g.copy()
            state.scale(float(dipole[alpha]), inplace=True)
            ket.msmps.append(state)
        ket.ms_normalize("mps_only")
        return ket

    def init_mps(self):
        ket = self._build_absorption_ket()
        bra = ket.copy()
        return bra, ket
```

- [ ] **Step 3: Implement propagation, autocorrelation recording, and dump payload**

```python
    def process_mps(self, mps):
        bra, ket = mps
        self._autocorr.append(_multiset_overlap(bra, ket))

    def evolve_single_step(self, evolve_dt):
        bra, ket = self.latest_mps
        new_ket = self.ms_model.evolve_state(ket, evolve_dt, normalize=False)
        return bra, new_ket

    @property
    def autocorr(self):
        return np.array(self._autocorr)

    def get_dump_dict(self):
        return {
            "temperature": self.temperature.as_au(),
            "time series": self.evolve_times,
            "autocorr": self.autocorr,
        }
```

- [ ] **Step 4: Run the new test file and fix any shape or dipole-contract issues until it passes**

Run:

```bash
pytest renormalizer/multiset/tests/test_spectra_zerot.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit the spectra implementation**

```bash
git add renormalizer/multiset/spectra.py renormalizer/multiset/tests/test_spectra_zerot.py
git commit -m "feat: add multiset zero-temperature absorption spectra job"
```

### Task 3: Export the class and run targeted regression checks

**Files:**
- Modify: `renormalizer/multiset/__init__.py`
- Test: `renormalizer/multiset/tests/test_spectra_zerot.py`
- Test: `renormalizer/model/tests/test_multiset_tdjob.py`
- Test: `renormalizer/model/tests/test_multiset_kubo.py`

- [ ] **Step 1: Export `MultisetSpectraZeroT` from the multiset package**

```python
# -*- coding: utf-8 -*-

from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.multiset.multiset_mpo import MultisetBlockMpo, MultisetMpo
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_tdjob import (
    MultisetChargeDiffusionDynamics,
    MultisetTdJob,
    MultisetTransportKubo,
)
from renormalizer.multiset.multisetspectra import MultisetSpectraZeroT
```

- [ ] **Step 2: Run the new spectra tests plus the existing multiset regression tests**

Run:

```bash
pytest renormalizer/multiset/tests/test_spectra_zerot.py renormalizer/model/tests/test_multiset_tdjob.py renormalizer/model/tests/test_multiset_kubo.py -v
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Commit the package export and regression-safe final state**

```bash
git add renormalizer/multiset/__init__.py
git commit -m "refactor: export multiset zero-temperature spectra API"
```
