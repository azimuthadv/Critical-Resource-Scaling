# Symmetry-Resolved QPE Resource Scaling Near Quantum Criticality

**A many-body quantum phase estimation study using the HHRI–Quobly [`qpe-toolbox`](https://github.com/quobly-sw/qpe-toolbox).**

This project investigates how spectral gaps, symmetry constraints and finite-size critical behaviour determine the quantum resources needed for energy estimation. It uses the `qpe-toolbox` Hamiltonian, circuit and estimation interfaces to connect low-energy many-body spectra with textbook Quantum Phase Estimation (QPE), Trotterised QPE and Robust Phase Estimation (RPE).

The central observation is that the smallest gap in the complete spectrum can differ substantially from the energy separation relevant to a state prepared in a particular symmetry sector. Estimating QPE precision from the global gap alone can therefore overstate the resources required to resolve energies accessible from that state.

## Scientific objective

For the open-boundary transverse-field Ising model,

$$
H=-J\sum_{i=0}^{L-2}Z_iZ_{i+1}-g\sum_{i=0}^{L-1}X_i,
$$

the analysis compares the global low-energy gap,

$$
\Delta_{\mathrm{global}}=E_1-E_0,
$$

with the gap to the first excited state in the ground state's global $X$-parity sector,

$$
\Delta_{\mathrm{sym}}=
\min_{n>0,\;p_n=p_0}(E_n-E_0).
$$

Here $p_n$ is the eigenvalue of the conserved parity operator $P=\prod_i X_i$. For parity-preserving state preparation and dynamics, $\Delta_{\mathrm{sym}}$ is the relevant spectral-resolution scale within that sector. A finite input-state overlap with individual eigenstates imposes a further operational constraint.

The project relates this accessible gap to QPE phase-register precision, coherent evolution time, constructed gate counts and finite-size scaling near $g/J=1$.

## Implemented analyses

| Component | Implementation |
| --- | --- |
| Many-body Hamiltonians | Open-boundary transverse-field Ising chain and spin-$1/2$ XXZ chain constructed using `qpe_toolbox.hamiltonian.Hamiltonian` |
| Spectral diagnostics | Sparse exact diagonalisation, global and parity-resolved gaps, and finite-size critical exponents |
| State preparation | DMRG ground-state MPS using `Hamiltonian.to_mpo()` and `quimb.tensor.DMRG2` |
| Textbook QPE | Exact time evolution and second-order Trotterised evolution through the toolbox estimation interface |
| Resource estimation | Circuit construction with `qpe_sample(..., run_simulation=False)` and package-native gate classification |
| Critical scaling | Entangling-gate scaling versus system size and inverse symmetry-resolved gap; local finite-size exponents |
| Register effects | Phase-qubit staircase and a register-quantisation residual separating integer precision thresholds from smooth inverse-gap scaling |
| Robust Phase Estimation | RPE energy-error analysis and comparison of ancilla and evolution-time requirements with textbook QPE |
| Additional diagnostics | Symmetry-induced resource inflation, resource susceptibility and curvature, XXZ finite-size-gap comparison |

The notebook includes API compatibility wrappers for differing `set_search_window()` and `robust_phase_estimation()` signatures across toolbox revisions. Sparse Hamiltonian conversion is implemented directly from the toolbox's documented Pauli-term representation.

## Resource-scaling interpretation

With target energy accuracy proportional to $\eta\Delta_{\mathrm{sym}}$, the phase-register size is chosen as

$$
m=\left\lceil\log_2\!\left(\frac{2W}{\eta\Delta_{\mathrm{sym}}}\right)\right\rceil,
$$

where $W$ is the selected energy-search window. Controlled evolution powers then yield a coherent-time proxy proportional to $2^m-1$.

For a one-dimensional local Hamiltonian with $O(L)$ terms and a gap closing as $\Delta\sim L^{-z}$, a leading fixed-Trotter-step resource estimate is

$$
N_{\mathrm{ent}}\sim\frac{L}{\Delta}\sim L^{1+z}.
$$

The generated circuits allow the measured finite-size gate exponent to be compared with this expectation. Discrete jumps in $m$ and the details of Hamiltonian-simulation compilation can modify finite-size fits. The resource analysis distinguishes these effects rather than assigning every increase in circuit cost to many-body criticality.

## Reference numerical run

The supplied, executed notebook records the following critical-point resource-only results at $g/J=1$, with $\eta=0.05$, a search-window width of $2$, and second-order Trotterisation using two steps:

| Chain length $L$ | Symmetry-resolved gap | Phase qubits | Entangling gates |
| ---: | ---: | ---: | ---: |
| 4 | 2.694593 | 5 | 2,614 |
| 6 | 1.900566 | 6 | 8,331 |
| 8 | 1.463725 | 6 | 11,355 |
| 10 | 1.189004 | 7 | 28,977 |

For these four resource points, the notebook reports an effective log–log entangling-gate exponent of approximately **2.464**. A separate $L=4,6,8,10,12$ critical spectral sweep reports effective gap exponents of approximately **0.926** (global) and **0.902** (same parity). These are finite-size numerical estimates from the included calculation, rather than thermodynamic-limit exponent determinations.

The reference notebook output reports `qpe-toolbox 1.1.0` and `quimb 1.15.0`. Gate counts and runtime may change with the toolbox version, numerical environment and circuit-construction settings.

## Files

- [`HHRI_QPE_Toolbox_Critical_Resource_Scaling_v2.ipynb`](HHRI_QPE_Toolbox_Critical_Resource_Scaling_v2.ipynb): primary executable notebook, including derivations, numerical sweeps, plots and saved reference outputs.
- [`Critical_resource_scaling.py`](Critical_resource_scaling.py): Google Colab-generated Python export of the analysis.

Both relative links assume the files are placed in the same GitHub directory as this README.

## Installation and execution

The primary supported workflow is **Google Colab or Jupyter**. For local use, the upstream package currently requires **Python 3.12 or later** and supports Linux, macOS and Windows through WSL.

Install the dependencies in a suitable Python environment:

```bash
python -m pip install -U "qpe-toolbox[recommended]" quimb scipy pandas matplotlib jupyterlab
```

Launch the notebook locally:

```bash
jupyter lab HHRI_QPE_Toolbox_Critical_Resource_Scaling_v2.ipynb
```

Run the cells in order to reproduce the spectral analysis, DMRG preparation, QPE simulations, resource-only circuit construction and diagnostic plots. In Google Colab, open the `.ipynb` file and use **Runtime → Run all**; its first code cell installs the relevant packages.

**Python export note:** `Critical_resource_scaling.py` includes the Colab-specific `!pip` command near the beginning. To execute it with a standard Python interpreter, install the dependencies using the terminal command above, remove or comment out that `!pip` line, and then run:

```bash
python Critical_resource_scaling.py
```

The full notebook is the reference entry point for this study.

## Configuration

The main parameters are defined near the start of their respective notebook sections:

| Parameter | Reference value | Purpose |
| --- | --- | --- |
| `L_VALUES` | `[4, 6, 8, 10]` | TFIM field-dependent finite-size sweep |
| `G_VALUES` | `np.linspace(0.60, 1.40, 17)` | Transverse-field grid |
| `L_DMRG` | `12` | DMRG system size |
| `ETA` | `0.05` | Relative gap-resolution tolerance |
| `SEARCH_WINDOW` | `2.0` | QPE energy-window width |
| `RPE_EPSILON` | `0.02` | RPE example precision parameter |
| `RPE_SHOTS` | `3` | Shots per RPE iteration in the example |

The script and notebook also expose Trotter order and step count, phase-register truncation for the small exact-QPE demonstration, and XXZ chain lengths and anisotropy grid.

## Scope and interpretation

This is a **research analysis and numerical benchmark**, rather than a hardware resource estimator or an automated regression test. Sparse exact diagonalisation limits the accessible chain sizes, and the QPE runs use idealised state preparation and simulator-level gates. Resource-only mode counts gates in constructed toolbox circuits without contracting the complete quantum state; it does not include physical error correction, routing or device-dependent compilation costs.

The symmetry-resolved gap is appropriate when preparation and evolution preserve the specified parity sector. For approximate states with finite overlap outside that sector, an overlap-conditioned accessible gap is a more general operational metric. The notebook identifies overlap-based branch tracking, joint error-budget optimisation, Trotter-order optimisation and block-encoding comparisons as possible extensions; these are distinct from its implemented numerical benchmarks.

## Attribution and references

**Research implementation:** Eric Howard.

This work builds on the [`qpe-toolbox`](https://github.com/quobly-sw/qpe-toolbox) developed by the Hon Hai Research Institute Quantum Computing Research Center and Quobly. The parent toolbox is distributed under the Apache-2.0 licence. For package APIs, installation and further QPE examples, consult the [official documentation](https://quobly-sw.github.io/qpe-toolbox/) and [package distribution](https://pypi.org/project/qpe-toolbox/).
