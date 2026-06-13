# optimal-control-lander

Optimal Control of a Variable-Mass Thrust-Vectoring Lander (MAE 270C Final Project)

To run the full simulation AND generate figures for parts I-IV:

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
| `p3_mission.ipynb` | Part III: min-time ascent + landing shooting (M1/M2) |
| `p4_nonlinear.ipynb` | Part IV: LQR on nonlinear plant + nonlinear TPBVP |

## Modules

| File | Role |
|------|------|
| `param.py` | Physical parameters and control/state bounds |
| `dynamics.py` | Trim, linearization, Gramian |
| `constraints.py` | PMP control saturation and numerical state enforcement |
| `lqr.py` | Riccati solver, tracking feedforward, constrained simulation |
| `mission.py` | Part III indirect shooting (Phase A/B), Hamiltonians, residuals |
| `nonlinear.py` | Part IV: three-scenario comparison, nonlinear EOM, TPBVP, figure export |
| `validation.py` | Section 9 optimality checks (Parts I–IV, `validate_part4`) |
| `analysis.py` | Cost presets, regulation experiments |
| `experiments.py` | Shared ICs, time grids, Part II/III pipelines |
| `export_presentation_figures.py` | Batch figure export for presentation |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy control matplotlib jupyter
```
