"""Resource-only QPE benchmark tied to finite-size critical spectra.

The public qpe-toolbox QPE/Trotter tutorial already demonstrates generic
resource counting.  This benchmark changes the scientific control variable:
the QPE search interval is tied to a symmetry-resolved TFIM excitation scale,
then the resource-only circuit construction is repeated across system size and
field.

Run
---
python benchmark_qpe_resource_scaling.py
python benchmark_qpe_resource_scaling.py --output results/qpe_critical_resources.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import qpe_toolbox.estimation as qpe
from qpe_toolbox.circuit import count_gates_by_qb, make_circ

from critical_models import (
    exact_ground_state_mps,
    symmetry_resolved_gap,
    tfim_hamiltonian,
    x_parity_operator,
)


def _search_window_compat(ham, e_target: float, size_interval: float):
    """Normalize qpe-toolbox 1.1.0 and current-main return conventions."""
    result = qpe.set_search_window(ham, e_target, size_interval)
    if len(result) == 3:
        _energy_shift, evolution_time, global_phase = result
        return float(evolution_time), float(global_phase)
    if len(result) == 4:
        _e_const, _e_max, evolution_time, global_phase = result
        return float(evolution_time), float(global_phase)
    raise RuntimeError(
        f"Unexpected set_search_window return length: {len(result)}"
    )


def resource_row(
    *,
    n_qubits: int,
    field: float,
    n_phase_bits: int,
    trotter_steps: int,
    trotter_order: int,
) -> dict[str, float | int]:
    ham = tfim_hamiltonian(n_qubits, field=field)

    dense = np.asarray(ham.to_dense())
    gaps = symmetry_resolved_gap(dense, x_parity_operator(n_qubits))
    exact_energy, psi0_mps = exact_ground_state_mps(ham)

    # Use the first excitation in the prepared state's symmetry sector as the
    # physical scale.  Four gap widths gives room around E0 while retaining the
    # critical finite-size dependence.
    size_interval = max(4.0 * gaps.sector_gap, 1e-3)
    evolution_time, global_phase = _search_window_compat(
        ham,
        exact_energy,
        size_interval,
    )

    circ = make_circ(n_phase_bits, psi0_mps)
    dt = evolution_time / trotter_steps

    traces, _ = qpe.qpe_sample(
        ham,
        circ,
        evolution_time,
        dt,
        global_phase,
        trotter_order=trotter_order,
        run_simulation=False,
        verbosity=0,
    )

    by_qb = count_gates_by_qb(traces["gates_count"])

    return {
        "n_qubits": n_qubits,
        "field": field,
        "unrestricted_gap": gaps.unrestricted_gap,
        "symmetry_sector_gap": gaps.sector_gap,
        "gap_ratio": gaps.sector_gap / gaps.unrestricted_gap,
        "size_interval": size_interval,
        "n_phase_bits": n_phase_bits,
        "trotter_steps_base": trotter_steps,
        "trotter_order": trotter_order,
        "base_evolution_time": evolution_time,
        "one_qubit_gates": int(by_qb["1qb"]),
        "two_qubit_gates": int(by_qb["2qb"]),
        "three_plus_qubit_gates": int(by_qb["3+qb"]),
    }


def run(output: Path) -> None:
    rows: list[dict[str, float | int]] = []

    for n_qubits in (4, 6, 8):
        for field in (0.5, 1.0, 1.5):
            for n_phase_bits in (3, 5, 7):
                rows.append(
                    resource_row(
                        n_qubits=n_qubits,
                        field=field,
                        n_phase_bits=n_phase_bits,
                        trotter_steps=4,
                        trotter_order=2,
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
        default=Path("results/qpe_critical_resources.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
