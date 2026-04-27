# Experiments

This directory contains two self-contained experiments that reproduce all
numerical results from the paper.  Each experiment is split into two notebooks:
**`simulate.ipynb`** runs the computations and saves results to disk, and
**`plot.ipynb`** loads those results and generates the figures.

---

## Setup

Install the package with the experiment dependencies from the repository root:

```bash
pip install -e ".[experiments]"
```

This installs `pike` along with `torch`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `tqdm`, and `jupyter`.

A **CUDA-capable GPU is strongly recommended** for the simulation notebooks.
The plotting notebooks load pre-computed `.npz` files and run on CPU in
seconds.

---

## Structure

```
experiments/
├── results/                  # Auto-created; stores .npz and .pdf outputs
├── poly_system/
│   ├── simulate.ipynb        # Run first
│   └── plot.ipynb            # Run after simulate
└── van_der_pol/
    ├── simulate.ipynb        # Run first
    └── plot.ipynb            # Run after simulate
```

---

## Experiment 1 — Closed polynomial system (`poly_system/`)

Validates PIKE on a 3-dimensional polynomial system that admits an **exact
finite-dimensional closure** (dictionary of size M = 8, compared to the
M = 34 monomials up to degree 4 used by gEDMD).

### Running order

1. **`simulate.ipynb`** — generates all numerical results.

   - Builds the PIKE dictionary and runs the pre-training phase for
     empirical-pEDMD and sparse-iEDMD (requires labeled data from Q = 9
     auxiliary systems).
   - Sweeps over L = 1 to 500 training points, averaging prediction error
     over 1 000 random systems (different µ values).
   - Saves results to `results/poly_errors_n3_sys1000.npz`.
   - Also saves the sparse matrix structure to `results/poly_sparse_matrices.npz`
     and Koopman matrices / trajectories at L = 200.
   - **Expected runtime:** approximately 10–20 minutes on a modern GPU
     (RTX 3000 series or equivalent).

2. **`plot.ipynb`** — reproduces Figures 2 and 3 from the paper.

   - Reads `results/poly_errors_n3_sys1000.npz` and
     `results/poly_sparse_matrices.npz`.
   - Exports `results/poly_error_vs_L.pdf` and
     `results/poly_sparse_matrices.pdf`.
   - Also produces eigenvalue and trajectory plots (not included in the
     paper) for diagnostic purposes.
   - **Runtime:** a few seconds on CPU.

---

## Experiment 2 — Van der Pol oscillator (`van_der_pol/`)

Validates PIKE on the Van der Pol oscillator, a system for which **exact
closure is unattainable**. The dictionary is truncated at degree 12
(M = 42 observables, compared to M = 90 monomials for gEDMD).

### Running order

1. **`simulate.ipynb`** — generates all numerical results.

   - Truncates the PIKE algorithm at degree 12 and warns that closure was
     not reached.
   - Runs pre-training for empirical-pEDMD and sparse-iEDMD (Q = 6
     auxiliary systems, 10 000 points each).
   - Sweeps over L = 1 to 1 000 training points, averaging prediction error
     over 10 random systems.
   - Saves results to `results/vdp_errors_sys10.npz`,
     `results/vdp_sparse_matrices.npz`, and Koopman matrices / trajectories
     at L = 1 000.
   - **Expected runtime:** approximately 20–40 minutes on a modern GPU,
     dominated by the per-system sparse-iEDMD loop.

2. **`plot.ipynb`** — reproduces Figures 4 and 5 from the paper.

   - Reads `results/vdp_errors_sys10.npz` and
     `results/vdp_sparse_matrices.npz`.
   - Exports `results/vdp_error_vs_L.pdf` and
     `results/vdp_sparse_matrices.pdf`.
   - **Runtime:** a few seconds on CPU.

---

## Notes

- Both simulation notebooks check whether a results file already exists
  before running; re-running them is safe and will not overwrite existing
  results.  Delete the relevant `.npz` file to force a fresh run.
- The `results/` directory is listed in `.gitignore`. Pre-computed results
  are not tracked by Git; each user must run the simulation notebooks
  locally to generate them.
- The error floor visible at approximately 10⁻¹² in all figures corresponds
  to the numerical precision of double-precision floating-point arithmetic
  (`torch.float64`), as noted in Section V of the paper.