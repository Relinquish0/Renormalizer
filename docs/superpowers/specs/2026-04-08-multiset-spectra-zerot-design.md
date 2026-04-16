# Multiset Zero-Temperature Absorption Spectra Design

## Goal

Add a dedicated zero-temperature multiset absorption-spectra time-evolution driver that works with the current `renormalizer/multiset` ansatz without introducing an explicit zero-exciton channel into `MultisetMps`.

The target object is a new `MultisetSpectraZeroT` class that:

- inherits from `renormalizer/multiset/multiset_tdjob.py`'s `MultisetTdJob`
- prepares a dipole-excited multiset initial state in `init_mps()`
- propagates the excited state with the existing multiset TDVP-PS kernel
- records the time-domain autocorrelation function for zero-temperature absorption

## Context

The current `renormalizer/multiset` implementation makes one structural choice that controls the design:

- `MultisetMps` stores a list of phonon-only MPS objects, one per electronic excitation label
- explicit local electronic basis states are removed from the MPS representation
- `MultisetModel` converts electronic bilinears such as `a^\dagger a` into block Hamiltonian structure `H^{alpha,beta}`
- single-electron creation and annihilation operators are not represented as internal multiset MPO operators

This means the singleset implementation pattern

```python
Mpo.onsite(model, r"a^\dagger", dipole=True).apply(psi_g)
```

cannot be transferred literally to multiset code.

For zero-temperature absorption, the dipole action must instead be represented as an initial-state construction rule:

\[
|\Psi_{\mathrm{abs}}(0)\rangle
=
\sum_\alpha \mu_\alpha |\phi_g\rangle_\alpha
\]

where:

- `mu_alpha` comes from `Model.dipole`
- `phi_g` is the zero-temperature phonon-only reference state
- the result is a `MultisetMps` whose `alpha`-th component is `mu_alpha * phi_g`

## Decision

Use a dedicated multiset absorption initial-state builder inside `MultisetSpectraZeroT.init_mps()`.

Do not:

- add a zero-exciton reference channel to `MultisetMps`
- redefine `a^\dagger` as an internal multiset MPO operator
- extend the multiset Hilbert space just to mimic singleset operator semantics

This preserves the current multiset ansatz and matches the current code structure.

## Scope

This design covers only:

- zero-temperature absorption
- one-way time propagation
- time-domain autocorrelation output

This design explicitly does not cover:

- emission
- finite-temperature spectra
- two-way propagation
- Fourier transform / spectrum post-processing

Those are follow-up tasks and should not be coupled to the first implementation.

## Architecture

### 1. New task-level class

Add a new module:

- `renormalizer/multiset/spectra.py`

Add a new class:

- `MultisetSpectraZeroT`

This class inherits from:

- `renormalizer.multiset.multiset_tdjob.MultisetTdJob`

Its responsibility is orchestration only:

- build the initial bra/ket pair
- evolve the ket state in time
- evaluate and store autocorrelation values

It should not reimplement multiset propagation kernels. Those remain in `MultisetModel`.

### 2. State-preparation strategy

`MultisetSpectraZeroT.init_mps()` should:

1. construct or reuse a `MultisetModel` with `auto_init=False`
2. obtain the zero-temperature phonon-only reference state from the existing `MultisetMps.init_mp()` pathway
3. read absorption dipoles from `self.model.dipole`
4. construct a `MultisetMps` whose `msmps[alpha] = mu_alpha * phi_g`
5. preserve the overall dipole amplitude in the constructed multiset state
6. return `(bra, ket)` where the initial bra and ket are equal copies of the dipole-excited state

The reference state must reuse the current zero-temperature initialization path rather than introducing a second ground-state preparation route.

### 3. Dipole source

The dipole interface remains attached to `Model`.

The class must read dipole data from:

- `model.dipole`

It should not accept a second, parallel dipole input source unless the model data is absent.

For the first implementation, the accepted contract should be explicit:

- the absorption dipole must resolve to one scalar coefficient per electronic excitation label `alpha`

If `model.dipole` is stored in a richer structure, the conversion rule used by `MultisetSpectraZeroT` must be explicit and local to this class.

No hidden assumptions about shape should be left undocumented.

### 4. Time propagation strategy

The first implementation uses one-way propagation only.

`evolve_single_step(evolve_dt)` should:

- keep the bra state fixed
- propagate only the ket state using `self.ms_model.evolve_state(...)`

This matches the simplest working analogue of singleset zero-temperature absorption and minimizes moving parts in the first multiset spectra implementation.

### 5. Autocorrelation evaluation

`process_mps()` should compute the multiset overlap

\[
C(t) = \sum_\alpha \langle \phi_\alpha^{\mathrm{bra}}(t) | \phi_\alpha^{\mathrm{ket}}(t) \rangle
\]

using the current multiset state representation.

This requires a local utility or inline implementation for the multiset inner product of two `MultisetMps` objects.

The value should be appended to an internal autocorrelation array and exposed through a property such as `autocorr`.

### 6. Output contract

`get_dump_dict()` should include:

- `temperature`
- `time series`
- `autocorr`

This should mirror the existing singleset `spectra` output structure closely enough to keep downstream usage consistent.

## File Responsibilities

### New file

`renormalizer/multiset/spectra.py`

Responsibilities:

- define `MultisetSpectraZeroT`
- define any small helper functions that are specific to multiset spectra initialization or overlap evaluation

This file should stay focused on spectra orchestration. It should not absorb generic multiset utilities that are unrelated to spectra.

### Modify

`renormalizer/multiset/__init__.py`

Responsibilities:

- export `MultisetSpectraZeroT`

### New test file

`renormalizer/multiset/tests/test_spectra_zerot.py`

Responsibilities:

- validate initialization
- validate basic propagation
- validate autocorrelation recording
- validate dipole weighting behavior

## Detailed Behavior

### Initial-state construction

The initial-state builder should behave like a dipole-weighted generalization of the current FC excitation logic.

Current FC-style initialization effectively selects one excitation channel and suppresses the others.

The absorption initializer should instead:

- retain all excitation channels
- assign each channel its model dipole weight
- use a shared phonon reference state

Conceptually:

- FC excitation: `msmps[alpha0] <- phi_g`, all others near zero
- absorption dipole excitation: `msmps[alpha] <- mu_alpha * phi_g`

### Error handling

The class should fail early if:

- `model.dipole` is missing
- the dipole data cannot be reduced to one absorption coefficient per excitation channel
- the number of dipole coefficients does not match `model.n_edofs`

The failure message should state the expected contract clearly.

### Dipole amplitude preservation

After building the dipole-weighted multiset initial state, the class must preserve the overall dipole amplitude instead of globally normalizing it away.

This keeps `autocorr[0]` sensitive to the magnitude of `Model.dipole` and matches the physical role of the dipole-prepared state in the singleset spectra path.

### Reuse boundaries

The design intentionally reuses:

- `MultisetModel`
- `MultisetMps`
- `MultisetTdJob`
- `MultisetModel.evolve_state(...)`

It intentionally does not reuse:

- `fc_excitation()` directly

because `fc_excitation()` is a single-channel excitation helper, while the spectra initializer is a weighted all-channel excitation map.

## Testing Strategy

The first implementation only needs small-system tests. The goal is to validate representation and control flow before benchmarking against full production spectra.

### Required tests

1. **Initialization test**

Construct a small model with dipoles and verify:

- `MultisetSpectraZeroT.init_mps()` returns a bra/ket pair
- the returned multiset ket has the same reference MPS shape across channels
- channel norms scale with dipole weights before normalization

2. **Autocorrelation bootstrap test**

Immediately after job construction:

- `autocorr` has length 1
- `autocorr[0]` is finite

3. **One-step evolution test**

After one propagation step:

- `autocorr` length increments
- the new autocorrelation value is finite
- the ket state changes while the bra state remains unchanged

4. **Dipole sensitivity test**

Change the dipole vector in the model and verify that `autocorr[0]` changes consistently.

This test is important because the dipole mapping is the key new abstraction.

### Deferred tests

The following should be deferred until the basic implementation is stable:

- comparison against singleset spectra on a benchmark model
- convergence tests in bond dimension
- time-step convergence tests
- two-way propagation parity checks

## Trade-offs Considered

### Approach A: Add a zero-exciton channel to multiset state space

Pros:

- would make operator semantics closer to singleset

Cons:

- changes the current ansatz
- requires updating multiset state, operator, and propagation logic
- too large a change for the first spectra implementation

Decision: reject.

### Approach B: Define a fake multiset internal `a^\dagger` operator

Pros:

- superficially similar API to singleset

Cons:

- mathematically misleading in the current representation
- hides the fact that the dipole action is an initialization map, not an internal block operator

Decision: reject.

### Approach C: Build a dipole-weighted multiset initial state in `init_mps()`

Pros:

- matches the current ansatz
- minimal code changes
- clear physical meaning
- reuses existing propagation code unchanged

Cons:

- notationally less similar to singleset `Mpo.onsite(...).apply(...)`

Decision: accept.

## Implementation Order

The implementation should proceed in this order:

1. Add `renormalizer/multiset/spectra.py`
2. Implement a multiset overlap helper
3. Implement dipole-weighted absorption-state construction in `init_mps()`
4. Implement one-way propagation
5. Implement autocorrelation recording and dumping
6. Export the class from `renormalizer/multiset/__init__.py`
7. Add small-system tests

Do not start with emission or finite temperature.

## Open Assumptions Made Explicit

These assumptions are fixed for the first implementation:

- the first spectra implementation is zero-temperature absorption only
- dipole data comes from `Model.dipole`
- the reference state is the existing zero-temperature `MultisetMps.init_mp()` result
- the implementation lives in `renormalizer/multiset/spectra.py`
- `MultisetSpectraZeroT` inherits from `MultisetTdJob`
- propagation is one-way only

If any of these assumptions change, the design should be revised before implementation begins.
