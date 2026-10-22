"""Regression tests for critical-model Hamiltonians.

These tests target model classes not used as the main regression set in the
public qpe-toolbox tutorials.  They validate qpe-toolbox Hamiltonian
construction against an independent dense-Pauli implementation.
"""

import numpy as np
import pytest

from qpe_toolbox.hamiltonian import heisenberg_hamiltonian

from critical_models import (
    annni_hamiltonian,
    dense_from_terms,
    long_range_tfim_hamiltonian,
    tfim_hamiltonian,
    xy_hamiltonian,
    xxz_hamiltonian,
)


@pytest.mark.parametrize(
    ("hamiltonian", "n_qubits"),
    [
        (tfim_hamiltonian(4, field=1.0), 4),
        (xy_hamiltonian(4, field=1.0, anisotropy=0.5), 4),
        (xxz_hamiltonian(4, delta=1.0), 4),
        (annni_hamiltonian(5, field=0.8, kappa=0.35), 5),
        (long_range_tfim_hamiltonian(4, field=1.0, alpha=2.0), 4),
    ],
)
def test_toolbox_dense_matrix_matches_independent_pauli_builder(
    hamiltonian,
    n_qubits,
):
    """qpe-toolbox dense conversion must reproduce an independent oracle."""
    toolbox_dense = np.asarray(hamiltonian.to_dense())
    reference = dense_from_terms(hamiltonian.terms, n_qubits)

    assert toolbox_dense.shape == reference.shape
    assert np.allclose(toolbox_dense, reference, atol=1e-12, rtol=1e-12)
    assert np.allclose(toolbox_dense, toolbox_dense.conj().T, atol=1e-12)


def test_xxz_delta_one_matches_builtin_heisenberg_normalization():
    """The local XXZ constructor should reduce to qpe-toolbox Heisenberg at Δ=1."""
    n_qubits = 5
    coupling = 0.73

    xxz = np.asarray(
        xxz_hamiltonian(
            n_qubits,
            delta=1.0,
            coupling=coupling,
        ).to_dense()
    )
    heisenberg = np.asarray(
        heisenberg_hamiltonian(
            n_qubits,
            coupling_strength=coupling,
        ).to_dense()
    )

    assert np.allclose(xxz, heisenberg, atol=1e-12, rtol=1e-12)


def test_long_range_tfim_converges_to_nearest_neighbour_limit():
    """Large alpha should suppress all interactions beyond nearest neighbours."""
    n_qubits = 4
    field = 0.91

    short_range = np.asarray(tfim_hamiltonian(n_qubits, field=field).to_dense())
    very_short_range = np.asarray(
        long_range_tfim_hamiltonian(
            n_qubits,
            field=field,
            alpha=50.0,
        ).to_dense()
    )

    assert np.allclose(short_range, very_short_range, atol=2e-14, rtol=1e-14)


def test_long_range_model_contains_all_distinct_pairs():
    n_qubits = 6
    ham = long_range_tfim_hamiltonian(n_qubits, field=1.0, alpha=3.0)

    n_zz = n_qubits * (n_qubits - 1) // 2
    n_x = n_qubits
    assert ham.n_terms == n_zz + n_x


@pytest.mark.parametrize("n_qubits", [0, 1, -2])
def test_invalid_chain_length_is_rejected(n_qubits):
    with pytest.raises(ValueError):
        tfim_hamiltonian(n_qubits, field=1.0)

# ---------------------------------------------------------------------------
# Small end-to-end QPE regression at and away from the TFIM critical point.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [0.5, 1.0, 1.5])
def test_exact_time_evolution_qpe_obeys_phase_resolution_bound(field):
    """Small QPE regression using an exact ground state converted to MPS."""
    import qpe_toolbox.estimation as qpe
    from qpe_toolbox import EXACT
    from qpe_toolbox.circuit import make_circ

    from critical_models import exact_ground_state_mps

    n_qubits = 3
    n_phase_bits = 4
    size_interval = 4.0

    ham = tfim_hamiltonian(n_qubits, field=field)
    exact_energy, psi0_mps = exact_ground_state_mps(ham)

    circ = make_circ(n_phase_bits, psi0_mps)
    traces, estimated_energy = qpe.qpe_energy(
        ham,
        circ,
        EXACT,
        exact_energy,
        size_interval,
        verbosity=0,
    )

    resolution = size_interval / 2**n_phase_bits
    error = abs(float(estimated_energy) - exact_energy)

    assert np.isfinite(estimated_energy)
    assert error <= resolution + 1e-8
    assert 0.0 <= float(traces["prob"]) <= 1.0
