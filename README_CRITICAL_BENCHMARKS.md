# Critical-system benchmarks for `qpe-toolbox`

This directory adds regression tests and research benchmarks for **Quantum Phase
Estimation near quantum criticality**.

The files are designed to complement, rather than reproduce, the public
`qpe-toolbox` tutorials.  The public documentation already contains examples
for:

- textbook QPE with exact time evolution;
- QPE with first- and second-order Trotterization;
- generic Trotter operator-error studies on the Heisenberg chain;
- Robust Phase Estimation on Heisenberg and H2;
- MPS backend performance;
- LCU/block encoding;
- variational state preparation;
- MPO-to-circuit transpilation, including a next-nearest-neighbour Ising
  example.

The tests here target a different gap: **physics-aware validation and resource
benchmarking across quantum-critical regimes**.

## Files

`critical_models.py`  
Small explicit constructors for TFIM, anisotropic XY, XXZ, transverse ANNNI
and long-range TFIM Hamiltonians.  It also contains an independent dense Pauli
builder, exact small-system MPS preparation and a Z2 symmetry-resolved gap
utility.

`test_critical_model_regressions.py`  
Checks qpe-toolbox Hamiltonian matrices against an independent dense-Pauli
oracle. It also verifies that XXZ at Delta=1 matches the built-in Heisenberg
normalization, checks the long-range Ising nearest-neighbour limit, and runs
small end-to-end QPE regressions at g/J = 0.5, 1.0 and 1.5.

`test_symmetry_resolved_gaps.py`  
Adds a criticality-specific physical regression.  It verifies global X parity
for TFIM/ANNNI and checks the ordered TFIM regime where the unrestricted
finite-size gap is much smaller than the first excitation in the ground-state
parity sector.


`benchmark_trotter_criticality.py`  
Compares first- and second-order Trotter error, gate counts and runtime for
ordered, critical and paramagnetic TFIM instances plus a frustrated ANNNI
instance.  This changes the control problem from the Heisenberg model used in
the public Trotter tutorial to critical-regime comparisons.

`benchmark_qpe_resource_scaling.py`  
Builds QPE circuits without full simulation and records one-qubit and
multi-qubit gate counts across system size, field and phase-register size.  The
QPE energy window is tied to the first excitation gap in the ground-state Z2
sector, allowing resource counts to be analysed together with finite-size
critical spectra.

## Installation

Use the same environment as `qpe-toolbox`.

```bash
python -m pip install "qpe-toolbox>=1.1.0" pytest
```

For development against the current `qpe-toolbox` main branch, use the package
installation instructions in the upstream repository.

## Run the regression tests

Place all six files in one directory, then run:

```bash
pytest -q test_critical_model_regressions.py \
          test_symmetry_resolved_gaps.py
```

The end-to-end QPE smoke test is more expensive than the matrix-only tests but
uses only three physical qubits.

## Run the benchmarks

```bash
python benchmark_trotter_criticality.py
python benchmark_qpe_resource_scaling.py
```

Default CSV outputs:

```text
results/trotter_criticality.csv
results/qpe_critical_resources.csv
```

Both benchmark scripts accept an alternate output path with `--output`.

## Interpretation

The regression tests are intended to be stable tests of model construction,
symmetry and basic QPE behaviour.  The benchmark CSV values are **research
outputs**, not hard-coded pass/fail thresholds.  Wall-clock timings should not
be used as CI assertions because they depend strongly on CPU, BLAS and
tensor-network contraction settings.

The resource benchmark deliberately reports the unrestricted gap and the
ground-state symmetry-sector gap separately.  In the finite ordered TFIM, the
lowest two states form a near-degenerate parity doublet, so an unrestricted
gap can imply a much longer evolution-time requirement than the excitation
scale available within the prepared state's symmetry sector.

## Compatibility note

`qpe-toolbox` 1.1.0 and the current development documentation differ in the
return convention of `set_search_window`.  The benchmark contains a small
compatibility adapter accepting either the three-value or four-value return
form.  This is intentional so the research repository can be used against the
public release and current development branch.

## Suggested repository location

A clean layout is:

```text
benchmarks/
    critical_models.py
    test_critical_model_regressions.py
    test_symmetry_resolved_gaps.py
    benchmark_trotter_criticality.py
    benchmark_qpe_resource_scaling.py
    README_CRITICAL_BENCHMARKS.md
```

If these are later proposed upstream to `qpe-toolbox`, the model utilities can
be moved into test fixtures and the regression files adapted to the upstream
test-directory conventions.
