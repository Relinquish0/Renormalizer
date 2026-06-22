# Multiset JCTC Manuscript Skeleton Design

## Goal

Create an English JCTC methods-framework manuscript skeleton in `paper/multisetpaper` for the multiset tensor-network work in Renormalizer.

The skeleton should compile as a LaTeX manuscript and make the planned paper structure visible immediately. It should use red text for generic, placeholder, or evidence-dependent language, and use dummy figure boxes where final figures will later be inserted.

## Paper Type and Argument

Paper type: methods framework paper for JCTC.

One-sentence argument:

In vibronic quantum dynamics, this paper introduces a unified multiset tensor-network framework in Renormalizer that supports real-time dynamics, finite-temperature propagation, zero- and finite-temperature spectroscopy, electronic ancilla purification, and multiset TTNS extensions, with Holstein, FMO, PBI, and P3HT:PCBM benchmarks demonstrating the scope and practical utility of the framework.

## Scope Decisions

Use approach A: JCTC methods-framework skeleton.

The manuscript should emphasize:

- A unified multiset MPS and multiset TTNS framework.
- Hamiltonian block decomposition for electronic-state-resolved vibrational tensor networks.
- Projector-splitting TDVP real-time propagation.
- Finite-temperature initialization and propagation.
- Zero- and finite-temperature spectra.
- Electronic ancilla purification for finite-temperature excited-state spectra.
- Multiset TTNS extension and P3HT:PCBM topology benchmarks.
- Holstein, FMO, PBI, and P3HT:PCBM as the main evidence chain.

Do not include as main-text theory or main results:

- Entropy diagnostics as a Theory/Methods module.
- A standalone Results subsection on the entanglement mechanism.
- A main-text result centered on `S_el` versus `S_maxbond`.

If entropy is mentioned at all, it should be a light supporting or future-diagnostic note, preferably in the Supporting Information outline rather than as a central manuscript claim.

## Terminology Ledger

| Canonical term | First-use definition |
|---|---|
| multiset matrix product state | multiset matrix product state (multiset MPS) |
| multiset tree tensor network state | multiset tree tensor network state (multiset TTNS) |
| singleset MPS / singleset TTNS | shared vibrational tensor-network ansatz used as the baseline |
| conditional vibrational wavepacket | electronic-state-conditioned nuclear wavepacket |
| electronic ancilla purification | purification over auxiliary electronic states for finite-temperature spectra |
| projector-splitting TDVP | projector-splitting time-dependent variational principle propagation |

## `main.tex` Design

Keep the existing `achemso` class and bibliography setup unless a compile issue requires a minimal package adjustment.

Replace the template text with:

1. A defensible methods-framework title.
2. An abstract scaffold following the JCTC methods pattern: challenge, gap, framework, benchmark evidence, implication, boundary.
3. Introduction:
   - Vibronic dynamics and spectroscopy require accurate electron-vibrational quantum dynamics.
   - Existing tensor-network approaches include TD-DMRG/MPS, ML-MCTDH/TTNS, and prior multiset MPS.
   - The gap is a unified multiset framework covering real-time dynamics, finite-temperature states, spectra, electronic ancilla purification, and TTNS within Renormalizer.
   - The present work introduces that framework and validates it on Holstein, FMO, PBI, and P3HT:PCBM systems.
4. Theory and Computational Framework:
   - General vibronic Hamiltonian.
   - Multiset ansatz.
   - Hamiltonian block construction.
   - Multiset TDVP propagation.
   - Finite-temperature multiset dynamics.
   - Zero- and finite-temperature spectroscopy.
   - Electronic ancilla purification.
   - Multiset TTNS extension.
   - Computational details and observables.
5. Results and Discussion:
   - Framework overview and implementation validation.
   - Holstein benchmarks for singleset versus multiset behavior across 1D and 2D models.
   - Finite-temperature FMO exciton dynamics.
   - Zero- and finite-temperature PBI spectra.
   - Electronic ancilla finite-temperature spectra.
   - P3HT:PCBM multiset TTNS dynamics and topology comparison.
   - Convergence, cost, and practical boundaries.
6. Conclusions:
   - Restate the framework contribution.
   - Summarize the evidence chain.
   - Bound the claim to vibronic models tested here.
7. Standard JCTC auxiliary sections:
   - Acknowledgement.
   - Data and Code Availability.
   - Competing Interests.
   - Supporting Information.

## Figure Dummy Plan

Use LaTeX dummy boxes that compile without external image files.

Planned main figures:

1. Framework overview: singleset MPS, multiset MPS, finite-temperature and spectra branches, and multiset TTNS.
2. Hamiltonian block decomposition and multiset TDVP workflow.
3. 1D Holstein benchmark.
4. 2D Holstein benchmark.
5. FMO finite-temperature population dynamics.
6. PBI zero- and finite-temperature spectra.
7. Electronic ancilla finite-temperature spectra.
8. P3HT:PCBM multiset TTNS topology and population dynamics.
9. Convergence and computational cost summary.

## `si.tex` Design

Expand `si.tex` into a Supporting Information skeleton with:

1. Supplementary derivations of the multiset ansatz and block Hamiltonian.
2. TDVP propagation details.
3. Finite-temperature initialization details.
4. Spectral correlation-function formulas.
5. Electronic ancilla purification details.
6. Multiset TTNS construction details.
7. Model parameter tables for Holstein, FMO, PBI, and P3HT:PCBM.
8. Full convergence figures and benchmark tables.
9. Optional supporting diagnostics, including entropy-related diagnostics only as noncentral supplementary material.

## Verification

After editing:

- Run `make html` only if this project supports it; otherwise skip.
- Run a LaTeX compile command from `paper/multisetpaper` if available and report whether it succeeds.
- If compilation fails because of missing local LaTeX dependencies, report the exact failure and leave the `.tex` files in a syntactically conservative state.

## Boundaries

Do not generate real scientific plots or fabricate numerical results.

Do not invent references beyond citation placeholders already in the bibliography or marked red placeholders.

Do not turn the manuscript into a software paper; the center remains a JCTC computational-methods manuscript supported by benchmark applications.
