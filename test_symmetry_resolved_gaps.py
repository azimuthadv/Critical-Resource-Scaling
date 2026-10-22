"""Criticality-specific symmetry regression tests.

The public qpe-toolbox tutorials demonstrate QPE, Trotterization, RPE and
tensor-network workflows.  These tests add a physics regression that is
important near the ordered TFIM regime: the smallest gap in the full spectrum
need not be the relevant excitation gap for a state confined to one Z2 sector.
"""

import numpy as np

from critical_models import (
    annni_hamiltonian,
    symmetry_resolved_gap,
    tfim_hamiltonian,
    x_parity_operator,
)


def _commutator_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ b - b @ a, ord="fro"))


def test_global_x_parity_commutes_with_open_tfim():
    n_qubits = 6
    ham = np.asarray(tfim_hamiltonian(n_qubits, field=0.8).to_dense())
    parity = x_parity_operator(n_qubits)

    assert _commutator_norm(ham, parity) < 1e-11


def test_global_x_parity_commutes_with_transverse_annni():
    n_qubits = 6
    ham = np.asarray(
        annni_hamiltonian(
            n_qubits,
            field=0.7,
            kappa=0.35,
        ).to_dense()
    )
    parity = x_parity_operator(n_qubits)

    assert _commutator_norm(ham, parity) < 1e-11


def test_same_sector_gap_is_not_smaller_than_unrestricted_gap():
    n_qubits = 6
    ham = np.asarray(tfim_hamiltonian(n_qubits, field=0.6).to_dense())
    parity = x_parity_operator(n_qubits)

    gaps = symmetry_resolved_gap(ham, parity)

    assert gaps.unrestricted_gap > 0.0
    assert gaps.sector_gap >= gaps.unrestricted_gap - 1e-12


def test_ordered_tfim_exposes_large_symmetry_gap_separation():
    """A regression for the parity-doublet regime used in the grant pilot.

    For a finite open chain deep in the ordered phase, the two lowest states
    form an exponentially close parity doublet.  The first excitation in the
    ground-state parity sector remains much higher.  This test is deliberately
    qualitative: it guards the physical effect without fitting a critical
    exponent.
    """
    n_qubits = 6
    ham = np.asarray(tfim_hamiltonian(n_qubits, field=0.2).to_dense())
    parity = x_parity_operator(n_qubits)

    gaps = symmetry_resolved_gap(ham, parity)
    ratio = gaps.sector_gap / gaps.unrestricted_gap

    assert ratio > 100.0


def test_near_critical_tfim_gap_is_finite_at_small_size():
    """Finite systems retain a nonzero excitation gap at g/J = 1."""
    n_qubits = 6
    ham = np.asarray(tfim_hamiltonian(n_qubits, field=1.0).to_dense())
    evals = np.linalg.eigvalsh(ham)

    assert float(evals[1] - evals[0]) > 0.0
