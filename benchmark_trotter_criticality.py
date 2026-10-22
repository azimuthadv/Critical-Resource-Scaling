"""Benchmark Trotter error at and away from quantum criticality.

This extends the public qpe-toolbox Trotter tutorial, which uses a nearest-
neighbour Heisenberg model, to critical-system comparisons.  It reports
operator error, gate counts and construction time for TFIM and transverse
ANNNI instances.

Run
---
python benchmark_trotter_criticality.py
python benchmark_trotter_criticality.py --output results/trotter_criticality.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from qpe_toolbox.circuit import count_gates, count_gates_by_qb

from critical_models import annni_hamiltonian, tfim_hamiltonian


def benchmark_case(
    *,
    model_name: str,
    ham,
    evolution_time: float,
    order: int,
    n_steps: int,
) -> dict[str, float | int | str]:
    n_qubits = ham.n_qubits
    data_reg = list(range(n_qubits))

    dense_h = np.asarray(ham.to_dense())
    u_exact = qu.expm(-1j * evolution_time * dense_h)

    dt = evolution_time / n_steps
    one_step = ham.get_trotter_step(dt, data_reg, order)

    t0 = time.perf_counter()
    circ = qtn.Circuit(n_qubits)
    for _ in range(n_steps):
        circ.apply_gates(one_step)
    u_trotter = circ.get_uni().to_dense()
    elapsed = time.perf_counter() - t0

    eye = np.eye(2**n_qubits, dtype=complex)
    error = qu.norm(u_exact.conj().T @ u_trotter - eye, ntype="fro")

    gates = count_gates(circ)
    by_qb = count_gates_by_qb(gates)

    return {
        "model": model_name,
        "n_qubits": n_qubits,
        "evolution_time": evolution_time,
        "trotter_order": order,
        "n_steps": n_steps,
        "dt": dt,
        "frobenius_unitary_error": float(error),
        "one_qubit_gates": int(by_qb["1qb"]),
        "two_qubit_gates": int(by_qb["2qb"]),
        "three_plus_qubit_gates": int(by_qb["3+qb"]),
        "estimated_swaps_1d": int(gates.get("SWAP", 0)),
        "build_and_contract_seconds": elapsed,
    }


def run(output: Path, n_qubits: int = 6) -> None:
    cases = [
        ("tfim_ordered_g0.5", tfim_hamiltonian(n_qubits, field=0.5)),
        ("tfim_critical_g1.0", tfim_hamiltonian(n_qubits, field=1.0)),
        ("tfim_paramagnetic_g1.5", tfim_hamiltonian(n_qubits, field=1.5)),
        (
            "annni_frustrated",
            annni_hamiltonian(
                n_qubits,
                field=0.8,
                kappa=0.35,
            ),
        ),
    ]

    rows: list[dict[str, float | int | str]] = []
    for model_name, ham in cases:
        for order in (1, 2):
            for n_steps in (1, 2, 4, 8, 16):
                rows.append(
                    benchmark_case(
                        model_name=model_name,
                        ham=ham,
                        evolution_time=1.0,
                        order=order,
                        n_steps=n_steps,
                    )
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trotter_criticality.csv"),
    )
    parser.add_argument("--n-qubits", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output, n_qubits=args.n_qubits)
