"""Critical-model constructors and independent reference utilities.

These helpers are intentionally small and explicit.  They are designed for
benchmarking qpe-toolbox on quantum-critical spin chains, not as a replacement
for qpe-toolbox model utilities.

Expected environment
--------------------
Python >= 3.12
qpe-toolbox >= 1.1.0
numpy
scipy
quimb
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from qpe_toolbox.hamiltonian import Hamiltonian


def tfim_hamiltonian(
    n_qubits: int,
    field: float,
    *,
    coupling: float = 1.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Open/periodic transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - g sum_i X_i.
    The conventional open-chain critical point is g/J = 1 in the
    thermodynamic limit.
    """
    _validate_n(n_qubits)
    terms: list[tuple[float, str, list[int]]] = []

    last = n_qubits if periodic else n_qubits - 1
    for i in range(last):
        j = (i + 1) % n_qubits
        terms.append((-float(coupling), "zz", [i, j]))

    for i in range(n_qubits):
        terms.append((-float(field), "x", [i]))

    return Hamiltonian(terms, n_qubits)


def xy_hamiltonian(
    n_qubits: int,
    field: float,
    anisotropy: float,
    *,
    coupling: float = 1.0,
) -> Hamiltonian:
    """Anisotropic XY chain with a transverse Z field.

    H = -(J/2) sum_i [(1+gamma) X_i X_{i+1}
                      + (1-gamma) Y_i Y_{i+1}]
        - h sum_i Z_i.
    """
    _validate_n(n_qubits)
    gamma = float(anisotropy)
    J = float(coupling)
    h = float(field)

    terms: list[tuple[float, str, list[int]]] = []
    for i in range(n_qubits - 1):
        terms.append((-0.5 * J * (1.0 + gamma), "xx", [i, i + 1]))
        terms.append((-0.5 * J * (1.0 - gamma), "yy", [i, i + 1]))
    for i in range(n_qubits):
        terms.append((-h, "z", [i]))

    return Hamiltonian(terms, n_qubits)


def xxz_hamiltonian(
    n_qubits: int,
    delta: float,
    *,
    coupling: float = 1.0,
) -> Hamiltonian:
    """Spin-1/2 XXZ chain using S = sigma/2 normalization.

    H = (J/4) sum_i [X_i X_{i+1} + Y_i Y_{i+1}
                     + Delta Z_i Z_{i+1}].

    At Delta=1 this matches qpe-toolbox's nearest-neighbour Heisenberg
    normalization.
    """
    _validate_n(n_qubits)
    J = float(coupling)
    D = float(delta)

    terms: list[tuple[float, str, list[int]]] = []
    for i in range(n_qubits - 1):
        terms.extend(
            [
                (0.25 * J, "xx", [i, i + 1]),
                (0.25 * J, "yy", [i, i + 1]),
                (0.25 * J * D, "zz", [i, i + 1]),
            ]
        )
    return Hamiltonian(terms, n_qubits)


def annni_hamiltonian(
    n_qubits: int,
    field: float,
    kappa: float,
    *,
    coupling: float = 1.0,
) -> Hamiltonian:
    """Transverse ANNNI chain.

    H = -J sum_i Z_i Z_{i+1}
        + kappa J sum_i Z_i Z_{i+2}
        - g sum_i X_i.

    Positive kappa introduces frustration through the next-nearest-neighbour
    Ising interaction.
    """
    _validate_n(n_qubits)
    J = float(coupling)
    g = float(field)
    k = float(kappa)

    terms: list[tuple[float, str, list[int]]] = []
    for i in range(n_qubits - 1):
        terms.append((-J, "zz", [i, i + 1]))
    for i in range(n_qubits - 2):
        terms.append((k * J, "zz", [i, i + 2]))
    for i in range(n_qubits):
        terms.append((-g, "x", [i]))
    return Hamiltonian(terms, n_qubits)


def long_range_tfim_hamiltonian(
    n_qubits: int,
    field: float,
    alpha: float,
    *,
    coupling: float = 1.0,
) -> Hamiltonian:
    """Power-law transverse-field Ising chain.

    H = -J sum_{i<j} Z_i Z_j / |i-j|^alpha - g sum_i X_i.
    """
    _validate_n(n_qubits)
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    J = float(coupling)
    g = float(field)
    a = float(alpha)

    terms: list[tuple[float, str, list[int]]] = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coeff = -J / float(j - i) ** a
            terms.append((coeff, "zz", [i, j]))
    for i in range(n_qubits):
        terms.append((-g, "x", [i]))
    return Hamiltonian(terms, n_qubits)


def dense_from_terms(
    terms: Iterable[tuple[float, str, list[int]]],
    n_qubits: int,
) -> np.ndarray:
    """Independent dense Pauli builder used as a regression oracle.

    This does not call Hamiltonian.to_dense(), so it can detect regressions in
    Pauli-string placement, coefficients, or qubit ordering.
    """
    _validate_n(n_qubits)
    pauli = {
        "i": np.eye(2, dtype=complex),
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    out = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for coeff, pstring, qubits in terms:
        pstring = pstring.lower()
        if len(pstring) != len(qubits):
            raise ValueError("Pauli string and active-qubit list differ in length")

        local = [pauli["i"].copy() for _ in range(n_qubits)]
        for p, q in zip(pstring, qubits, strict=True):
            if p not in pauli or p == "i":
                if p != "i":
                    raise ValueError(f"unsupported Pauli operator {p!r}")
            if q < 0 or q >= n_qubits:
                raise ValueError(f"qubit index {q} outside [0, {n_qubits})")
            local[q] = pauli[p]

        term = local[0]
        for op in local[1:]:
            term = np.kron(term, op)
        out += float(coeff) * term

    return out


def x_parity_operator(n_qubits: int) -> np.ndarray:
    """Global X-parity operator P = product_i X_i."""
    _validate_n(n_qubits)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    out = x
    for _ in range(n_qubits - 1):
        out = np.kron(out, x)
    return out


@dataclass(frozen=True)
class SymmetryGap:
    ground_energy: float
    unrestricted_gap: float
    sector_gap: float
    ground_sector: int


def symmetry_resolved_gap(
    dense_hamiltonian: np.ndarray,
    symmetry_operator: np.ndarray,
    *,
    sector_tol: float = 0.9,
) -> SymmetryGap:
    """Return the first gap in the ground-state symmetry sector.

    The routine classifies a non-degenerate numerical eigenstate by the sign of
    <psi|P|psi>.  It is intended for small exact-diagonalization regression
    tests rather than degenerate-subspace symmetry resolution.
    """
    evals, evecs = np.linalg.eigh(np.asarray(dense_hamiltonian))
    pvals = np.real(np.einsum("ij,ij->j", evecs.conj(), symmetry_operator @ evecs))

    if abs(pvals[0]) < sector_tol:
        raise RuntimeError("ground state is not numerically symmetry resolved")

    sector = 1 if pvals[0] > 0 else -1
    unrestricted = float(evals[1] - evals[0])

    same_sector = [
        k
        for k in range(1, len(evals))
        if abs(pvals[k]) >= sector_tol and (1 if pvals[k] > 0 else -1) == sector
    ]
    if not same_sector:
        raise RuntimeError("no excited state found in the ground-state symmetry sector")

    sector_gap = float(evals[same_sector[0]] - evals[0])
    return SymmetryGap(
        ground_energy=float(evals[0]),
        unrestricted_gap=unrestricted,
        sector_gap=sector_gap,
        ground_sector=sector,
    )


def exact_ground_state_mps(hamiltonian: Hamiltonian):
    """Convert the exact ground-state eigenvector to a quimb MPS.

    Intended only for small benchmark systems where exact diagonalization is
    deliberately used as the reference.
    """
    import quimb.tensor as qtn

    dense = np.asarray(hamiltonian.to_dense())
    evals, evecs = np.linalg.eigh(dense)
    psi0 = evecs[:, 0]
    mps = qtn.MatrixProductState.from_dense(
        psi0,
        dims=[2] * hamiltonian.n_qubits,
    )
    return float(evals[0]), mps


def _validate_n(n_qubits: int) -> None:
    if int(n_qubits) != n_qubits or n_qubits < 2:
        raise ValueError("n_qubits must be an integer >= 2")
