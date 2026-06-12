"""
Part I regulation experiments: cost-matrix sweeps and linearization comparisons.

State order: [p_x, p_z, v_x, v_z, theta, omega, m]  (deviations from trim)
Control:     [delta_T, tau]
"""

import numpy as np

import constraints as cst
import lqr

STATE_LABELS = ["p_x", "p_z", "v_x", "v_z", "θ", "ω", "m"]
CTRL_LABELS = ["δT", "τ"]

# Reference scales for nondimensional states (typical "good" perturbation sizes).
DEFAULT_REFERENCE_SCALES = np.array([10.0, 10.0, 5.0, 5.0, 0.2, 0.5, 1.0])

# Diagonal cost presets for Part I.5.5 sweeps (physical units).
COST_PRESETS = {
    "balanced": {
        "Q": [1.0, 1.0, 0.5, 0.5, 10.0, 5.0, 0.01],
        "R": [0.1, 1.0],
        "Qf": [10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1],
    },
    "position_heavy": {
        "Q": [20.0, 20.0, 2.0, 2.0, 5.0, 2.0, 0.01],
        "R": [0.1, 1.0],
        "Qf": [50.0, 50.0, 5.0, 5.0, 20.0, 10.0, 0.1],
    },
    "attitude_heavy": {
        "Q": [1.0, 1.0, 0.5, 0.5, 50.0, 25.0, 0.01],
        "R": [0.1, 2.0],
        "Qf": [10.0, 10.0, 1.0, 1.0, 100.0, 50.0, 0.1],
    },
    "cheap_thrust": {
        "Q": [1.0, 1.0, 0.5, 0.5, 10.0, 5.0, 0.01],
        "R": [0.01, 1.0],
        "Qf": [10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1],
    },
    "expensive_torque": {
        "Q": [1.0, 1.0, 0.5, 0.5, 10.0, 5.0, 0.01],
        "R": [0.1, 10.0],
        "Qf": [10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1],
    },
}


def diagonal_cost(Q_diag, R_diag, Qf_diag=None):
    """Build diagonal Q, R, Qf from vector diagonals."""
    Q = np.diag(np.asarray(Q_diag, dtype=float))
    R = np.diag(np.asarray(R_diag, dtype=float))
    Qf = np.diag(np.asarray(Qf_diag, dtype=float)) if Qf_diag is not None else np.zeros_like(Q)
    return Q, R, Qf


def cost_from_preset(name, scales=None):
    """Return (Q, R, Qf) for a named preset or scaled-identity variant."""
    if name == "scaled_identity":
        return identity_cost_in_scaled_coords(scales or DEFAULT_REFERENCE_SCALES)
    if name not in COST_PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Options: {list(COST_PRESETS)} + scaled_identity")
    p = COST_PRESETS[name]
    return diagonal_cost(p["Q"], p["R"], p["Qf"])


def scale_linear_system(A, B, scales):
    """
    Nondimensionalize x_tilde = D^{-1} x with D = diag(scales).
    x_dot = A x + B u  =>  x_tilde_dot = A_s x_tilde + B_s u
    """
    scales = np.asarray(scales, dtype=float)
    D = np.diag(scales)
    Dinv = np.diag(1.0 / scales)
    return Dinv @ A @ D, Dinv @ B


def identity_cost_in_scaled_coords(scales, R_diag=None):
    """
    Q = I, Qf = I in scaled coordinates map to physical Q = D^{-2} (diagonal).
    """
    scales = np.asarray(scales, dtype=float)
    Q = np.diag(1.0 / scales**2)
    Qf = Q.copy()
    if R_diag is None:
        R = np.eye(2)
    else:
        R = np.diag(np.asarray(R_diag, dtype=float))
    return Q, R, Qf


def control_energy(u_hist, t_grid):
    """Integral of ||u||^2 over the horizon."""
    dt = np.diff(t_grid)
    norms = np.linalg.norm(u_hist, axis=1) ** 2
    return np.sum(0.5 * (norms[:-1] + norms[1:]) * dt)


def weighted_state_norm(x, Q):
    return float(np.sqrt(x @ Q @ x))


def regulation_metrics(x_hist, u_hist, t_grid, Q, R, Qf, *, constraint_info=None):
    """Summary metrics for a regulation rollout."""
    t_eff = t_grid[: len(x_hist)]
    dt = np.diff(t_eff)
    state_norm = np.linalg.norm(x_hist, axis=1)
    weighted = np.array([weighted_state_norm(x_hist[k], Q) for k in range(len(x_hist))])

    u_norm = np.linalg.norm(u_hist, axis=1)
    settle_idx = np.where(state_norm < 0.05 * state_norm[0] + 1e-12)[0]
    settle_time = t_eff[settle_idx[0]] if settle_idx.size else np.nan

    metrics = {
        "J": lqr.regulation_cost(x_hist, u_hist, Q, R, Qf, t_eff),
        "control_energy": control_energy(u_hist, t_eff),
        "peak_state_norm": float(np.max(state_norm)),
        "peak_weighted_state": float(np.max(weighted)),
        "peak_control_norm": float(np.max(u_norm)),
        "terminal_state_norm": float(np.linalg.norm(x_hist[-1])),
        "settle_time": settle_time,
        "weighted_state_trace": weighted,
        "state_norm_trace": state_norm,
        "control_norm_trace": u_norm,
    }
    if constraint_info is not None:
        metrics["saturation_fraction"] = constraint_info.saturation_fraction
        metrics["state_violated"] = constraint_info.state_violated
        metrics["violation_time"] = constraint_info.violation_time
        metrics["violation_name"] = constraint_info.violation_name
        metrics["terminated_early"] = constraint_info.terminated_early
    return metrics


def run_regulation(
    dynamics_func,
    t_grid,
    Q,
    R,
    Qf,
    x0,
    params,
    *,
    lti=None,
    enforce_control=True,
    enforce_state=True,
):
    """
    Solve backward Riccati and simulate closed-loop regulation.

    Control saturation (PMP) and state truncation (numerical) are applied
    during the forward pass when the corresponding flags are True.
    """
    A0, B0 = dynamics_func(t_grid[0], params)
    if lti is None:
        Af, Bf = dynamics_func(t_grid[-1], params)
        lti = np.allclose(A0, Af) and np.allclose(B0, Bf)

    constraints = cst.RolloutConstraints(
        params=params,
        trim_func=cst.trim_for_dynamics(dynamics_func),
        enforce_control=enforce_control,
        enforce_state=enforce_state,
    )

    P_interp, _, _ = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf, params)
    ctrl = lambda tt, xx: lqr.regulation_control(tt, xx, dynamics_func, P_interp, Q, R, params)

    if lti:
        x_hist, u_hist, info = lqr.simulate_lti_closed_loop(
            A0, B0, t_grid, ctrl, x0, constraints=constraints
        )
    else:
        x_hist, u_hist, info = lqr.simulate_ltv_closed_loop(
            dynamics_func, t_grid, ctrl, x0, params, constraints=constraints
        )

    metrics = regulation_metrics(x_hist, u_hist, t_grid, Q, R, Qf, constraint_info=info)
    return {
        "x_hist": x_hist,
        "u_hist": u_hist,
        "P_interp": P_interp,
        "lti": lti,
        "metrics": metrics,
        "constraint_info": info,
        "t_eff": t_grid[: len(x_hist)],
    }


def sweep_cost_presets(
    dynamics_func,
    t_grid,
    x0,
    params,
    preset_names=None,
    *,
    include_scaled_identity=True,
    scales=None,
):
    """Run regulation for each cost preset; return list of result records."""
    if preset_names is None:
        preset_names = list(COST_PRESETS.keys())
    if include_scaled_identity:
        preset_names = list(preset_names) + ["scaled_identity"]

    rows = []
    for name in preset_names:
        Q, R, Qf = cost_from_preset(name, scales=scales)
        result = run_regulation(dynamics_func, t_grid, Q, R, Qf, x0, params)
        m = result["metrics"]
        rows.append(
            {
                "preset": name,
                "Q_diag": np.diag(Q),
                "R_diag": np.diag(R),
                "Qf_diag": np.diag(Qf),
                **m,
                "x_hist": result["x_hist"],
                "u_hist": result["u_hist"],
                "t_eff": result["t_eff"],
            }
        )
    return rows


def ic_sensitivity_sweep(
    dynamics_func,
    t_grid,
    Q,
    R,
    Qf,
    x0_base,
    delta_x0,
    alphas,
    params,
):
    """Scale initial perturbation x0 = x0_base + alpha * delta_x0."""
    rows = []
    for alpha in alphas:
        x0 = x0_base + alpha * delta_x0
        result = run_regulation(dynamics_func, t_grid, Q, R, Qf, x0, params)
        m = result["metrics"]
        rows.append(
            {
                "alpha": alpha,
                "x0_norm": float(np.linalg.norm(x0)),
                "terminal_state_norm": m["terminal_state_norm"],
                "J": m["J"],
                "control_energy": m["control_energy"],
                "peak_state_norm": m["peak_state_norm"],
                "settle_time": m["settle_time"],
            }
        )
    return rows


def compare_linearizations(
    hover_dynamics,
    descent_dynamics,
    t_grid,
    Q,
    R,
    Qf,
    x0,
    params,
    ic_alphas=None,
    delta_x0=None,
):
    """
    Part I.5.5: hover (controllable) vs descent (weakly controllable) under identical costs.
    """
    hover = run_regulation(hover_dynamics, t_grid, Q, R, Qf, x0, params, lti=True)
    descent = run_regulation(descent_dynamics, t_grid, Q, R, Qf, x0, params, lti=False)

    ic_rows = {}
    if ic_alphas is not None and delta_x0 is not None:
        ic_rows["hover"] = ic_sensitivity_sweep(
            hover_dynamics, t_grid, Q, R, Qf, x0, delta_x0, ic_alphas, params
        )
        ic_rows["descent"] = ic_sensitivity_sweep(
            descent_dynamics, t_grid, Q, R, Qf, x0, delta_x0, ic_alphas, params
        )

    return {
        "hover": hover,
        "descent": descent,
        "ic_sensitivity": ic_rows,
    }
