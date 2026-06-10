# optimal-control-lander

Optimal Control of a Variable-Mass Thrust-Vectoring Lander (ECE 270C Final Project)

## Presentation (Parts I & II)

Full slide-by-slide draft with derivations, numerical results, and graph interpretation:

- **[PRESENTATION_PARTS_I_II.md](PRESENTATION_PARTS_I_II.md)**

Regenerate all figures:

```bash
source .venv/bin/activate
python export_presentation_figures.py
```

Figures are written to `figures/presentation/`.

## Notebooks

| Notebook | Content |
|----------|---------|
| `p1_linearization.ipynb` | Part I: controllability, Gramian, LQR regulation, cost sweep |
| `p2_tracking.ipynb` | Part II: reference trajectory, tracking LQR, robustness |

## Modules

| File | Role |
|------|------|
| `param.py` | Physical parameters and control/state bounds |
| `dynamics.py` | Trim, linearization, Gramian |
| `constraints.py` | PMP control saturation and numerical state enforcement |
| `lqr.py` | Riccati solver, tracking feedforward, constrained simulation |
| `analysis.py` | Cost presets, regulation experiments |
| `experiments.py` | Shared ICs, time grids, Part II tracking pipeline |
| `export_presentation_figures.py` | Batch figure export for presentation |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy control matplotlib jupyter
```
