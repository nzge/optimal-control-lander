"""
Part IV — nonlinear optimal control near hover.

Three comparison scenarios (project requirement):

1. ``scenario_linear``   — LQR on the linearized (LTI) plant
2. ``scenario_lqr_nl``    — the same LQR law on the nonlinear plant
3. ``scenario_pmp_nl``    — nonlinear PMP / TPBVP on the nonlinear plant
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

import analysis as ana
import constraints as cst
import dynamics as dyn
import lqr

N_STATE = 7
TRIM_FUNC = dyn.hover_trim

SCENARIO_LABELS = {
    "linear": "LQR / linear plant",
    "lqr_nl": "LQR / nonlinear plant",
    "pmp_nl": "Nonlinear PMP / nonlinear plant",
}
SCENARIO_COLORS = {"linear": "C0", "lqr_nl": "C1", "pmp_nl": "C3"}


def _trim_mass_theta(t, x, params, trim_func):
    trim = trim_func(t, params)
    return trim["m"] + x[6], trim["theta"] + x[4]


def nonlinear_dynamics(t, x, u, params, *, trim_func=TRIM_FUNC):
    """True EOM in hover deviation coordinates (trim stationary)."""
    trim = trim_func(t, params)
    m = trim["m"] + x[6]
    theta = trim["theta"] + x[4]
    T = trim["T"] + u[0]
    tau = u[1]
    g, I, alpha = params["g"], params["I"], params["α"]

    x_dot = np.zeros(N_STATE)
    x_dot[0] = x[2]
    x_dot[1] = x[3]
    x_dot[2] = -(T / m) * np.sin(theta)
    x_dot[3] = (T / m) * np.cos(theta) - g
    x_dot[4] = x[5]
    x_dot[5] = tau / I
    x_dot[6] = -alpha * T
    return x_dot


def nonlinear_costate_dynamics(t, x, p, u, Q, params, *, trim_func=TRIM_FUNC):
    """Adjoint ODE dp/dt = -dH/dx with running cost x'Qx + u'Ru (no 1/2)."""
    m, theta = _trim_mass_theta(t, x, params, trim_func)
    T = trim_func(t, params)["T"] + u[0]

    p_dot = np.zeros(N_STATE)
    p_dot[0] = -2.0 * Q[0, 0] * x[0]
    p_dot[1] = -2.0 * Q[1, 1] * x[1]
    p_dot[2] = -2.0 * Q[2, 2] * x[2] - p[0]
    p_dot[3] = -2.0 * Q[3, 3] * x[3] - p[1]
    p_dot[4] = (
        -2.0 * Q[4, 4] * x[4]
        + (T / m) * (p[2] * np.cos(theta) + p[3] * np.sin(theta))
    )
    p_dot[5] = -2.0 * Q[5, 5] * x[5] - p[4]
    p_dot[6] = (
        -2.0 * Q[6, 6] * x[6]
        + (T / m**2) * (p[3] * np.cos(theta) - p[2] * np.sin(theta))
    )
    return p_dot


def optimal_control(t, x, p, R, params, *, trim_func=TRIM_FUNC):
    """PMP stationarity with actuator saturation (deviation control u=[delta_T, tau])."""
    m, theta = _trim_mass_theta(t, x, params, trim_func)
    trim = trim_func(t, params)
    alpha = params["α"]
    I = params["I"]

    T_req = (1.0 / (2.0 * R[0, 0])) * (
        p[2] * np.sin(theta) / m - p[3] * np.cos(theta) / m + p[6] * alpha
    )
    tau_req = -p[5] / (2.0 * I * R[1, 1])
    u_req = np.array([T_req - trim["T"], tau_req])
    lo, hi = cst.control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def hamiltonian(t, x, p, u, Q, R, params, *, trim_func=TRIM_FUNC):
    """Hamiltonian with running cost x'Qx + u'Ru (no 1/2)."""
    f = nonlinear_dynamics(t, x, u, params, trim_func=trim_func)
    return float(x @ Q @ x + u @ R @ u + p @ f)


def bvp_boundaries(ya, yb, x0, Qf):
    """Initial state match + free-terminal transversality p(tf) = 2 Qf x(tf)."""
    res_initial = ya[:N_STATE] - x0
    p_tf_expected = 2.0 * Qf @ yb[:N_STATE]
    res_terminal = yb[N_STATE:] - p_tf_expected
    return np.concatenate((res_initial, res_terminal))


def _bvp_rhs(t, y, Q, R, params, trim_func):
    if np.ndim(t) == 0:
        x, p = y[:N_STATE], y[N_STATE:]
        u = optimal_control(t, x, p, R, params, trim_func=trim_func)
        dx = nonlinear_dynamics(t, x, u, params, trim_func=trim_func)
        dp = nonlinear_costate_dynamics(t, x, p, u, Q, params, trim_func=trim_func)
        return np.concatenate([dx, dp])

    n = y.shape[1]
    out = np.zeros_like(y)
    for k in range(n):
        tk = float(t[k])
        x, p = y[:N_STATE, k], y[N_STATE:, k]
        u = optimal_control(tk, x, p, R, params, trim_func=trim_func)
        out[:N_STATE, k] = nonlinear_dynamics(tk, x, u, params, trim_func=trim_func)
        out[N_STATE:, k] = nonlinear_costate_dynamics(tk, x, p, u, Q, params, trim_func=trim_func)
    return out


def control_effort(u_hist, R, t):
    """J_u = integral u' R u dt."""
    stage = np.einsum("ti,tij,tj->t", u_hist, np.broadcast_to(R, (len(t), *R.shape)), u_hist)
    dt = np.diff(t)
    return float(np.sum((stage[:-1] + stage[1:]) * 0.5 * dt))


def total_running_cost(x_hist, u_hist, Q, R, Qf, t):
    """Integral x'Qx + u'Ru dt + x(tf)' Qf x(tf) (Part IV, no 1/2 factor)."""
    stage = np.einsum("ti,tij,tj->t", x_hist, np.broadcast_to(Q, (len(t), *Q.shape)), x_hist)
    stage += np.einsum("ti,tij,tj->t", u_hist, np.broadcast_to(R, (len(t), *R.shape)), u_hist)
    dt = np.diff(t)
    J = float(np.sum((stage[:-1] + stage[1:]) * 0.5 * dt))
    J += float(x_hist[-1] @ Qf @ x_hist[-1])
    return J


def hamiltonian_along_lqr(dynamics_func, t, x, u, P_interp, Q, R, params, *, nonlinear_plant=False):
    """Hamiltonian trace for LQR rollouts (linear or nonlinear plant)."""
    H = np.zeros(len(t))
    for k, tk in enumerate(t):
        p = 2.0 * P_interp(tk) @ x[k]
        if nonlinear_plant:
            H[k] = hamiltonian(tk, x[k], p, u[k], Q, R, params)
        else:
            A, B = dynamics_func(tk, params)
            H[k] = float(x[k] @ Q @ x[k] + u[k] @ R @ u[k] + p @ (A @ x[k] + B @ u[k]))
    return H


@dataclass
class NonlinearSolution:
    success: bool
    message: str
    t: np.ndarray
    x: np.ndarray
    p: np.ndarray
    u: np.ndarray
    H: np.ndarray
    bc_residual: float
    J_u: float
    n_nodes: int


def simulate_lqr_on_nonlinear(
    dynamics_func,
    t_grid,
    Q,
    R,
    Qf,
    x0,
    params,
    *,
    trim_func=TRIM_FUNC,
    enforce_control=True,
    enforce_state=True,
):
    """Scenario A: time-varying LQR on the nonlinear plant (RK4 + constraints)."""
    constraints = cst.RolloutConstraints(
        params=params,
        trim_func=trim_func,
        enforce_control=enforce_control,
        enforce_state=enforce_state,
    )
    P_interp, _, _ = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf, params)

    n = len(t_grid)
    x_hist = np.zeros((n, N_STATE))
    u_hist = np.zeros((n, 2))
    x_hist[0] = x0
    u_req0 = lqr.regulation_control(t_grid[0], x0, dynamics_func, P_interp, Q, R, params)
    u_hist[0], _ = cst.apply_control(u_req0, t_grid[0], constraints)

    if enforce_state and (vname := cst.check_state(x0, t_grid[0], constraints)):
        return _truncate_nonlinear_rollout(t_grid, x_hist, u_hist, 0, vname, P_interp, constraints, R)

    for k in range(n - 1):
        dt = t_grid[k + 1] - t_grid[k]
        t = t_grid[k]
        x = x_hist[k]
        u, _ = cst.apply_control(
            lqr.regulation_control(t, x, dynamics_func, P_interp, Q, R, params), t, constraints
        )
        u_hist[k] = u

        def f(xv):
            return nonlinear_dynamics(t, xv, u, params, trim_func=trim_func)

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_hist[k + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if enforce_state and (vname := cst.check_state(x_hist[k + 1], t_grid[k + 1], constraints)):
            u_hist[k + 1], _ = cst.apply_control(
                lqr.regulation_control(t_grid[k + 1], x_hist[k + 1], dynamics_func, P_interp, Q, R, params),
                t_grid[k + 1],
                constraints,
            )
            return _truncate_nonlinear_rollout(
                t_grid, x_hist, u_hist, k + 1, vname, P_interp, constraints, R,
            )

    u_hist[-1], _ = cst.apply_control(
        lqr.regulation_control(t_grid[-1], x_hist[-1], dynamics_func, P_interp, Q, R, params),
        t_grid[-1],
        constraints,
    )
    t_eff = t_grid
    return {
        "t": t_eff,
        "x": x_hist,
        "u": u_hist,
        "P_interp": P_interp,
        "constraints": constraints,
        "J_u": control_effort(u_hist, R, t_eff),
        "terminated_early": False,
        "violation": None,
    }


def _truncate_nonlinear_rollout(t_grid, x_hist, u_hist, k, vname, P_interp, constraints, R):
    t_eff = t_grid[: k + 1]
    u_trim = u_hist[: k + 1]
    return {
        "t": t_eff,
        "x": x_hist[: k + 1],
        "u": u_trim,
        "P_interp": P_interp,
        "constraints": constraints,
        "J_u": control_effort(u_trim, R, t_eff) if len(t_eff) >= 2 else 0.0,
        "terminated_early": True,
        "violation": vname,
    }


def _costate_guess(t_grid, x_guess, P_interp, *, half_cost_lqr=True):
    """Map Riccati costate (1/2-cost LQR) to Part IV no-1/2 PMP scaling."""
    scale = 2.0 if half_cost_lqr else 1.0
    p = np.zeros((len(t_grid), N_STATE))
    for k, tk in enumerate(t_grid):
        p[k] = scale * P_interp(tk) @ x_guess[k]
    return p


def solve_nonlinear_tpbvp(
    x0,
    t_guess,
    x_guess,
    p_guess,
    Q,
    R,
    Qf,
    params,
    *,
    trim_func=TRIM_FUNC,
    tol=1e-6,
    max_nodes=8000,
    verbose=False,
):
    """Scenario B: collocation BVP with LQR-warm-started guess."""
    y_guess = np.vstack([x_guess.T, p_guess.T])

    def fun(t, y):
        return _bvp_rhs(t, y, Q, R, params, trim_func)

    def bc(ya, yb):
        return bvp_boundaries(ya, yb, x0, Qf)

    sol = solve_bvp(
        fun,
        bc,
        t_guess,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=1 if verbose else 0,
        bc_tol=tol,
    )

    t = sol.x
    x = sol.y[:N_STATE].T
    p = sol.y[N_STATE:].T
    u = np.array([optimal_control(t[k], x[k], p[k], R, params, trim_func=trim_func) for k in range(len(t))])
    H = np.array([hamiltonian(t[k], x[k], p[k], u[k], Q, R, params, trim_func=trim_func) for k in range(len(t))])
    bc_res = float(np.max(np.abs(bc(sol.y[:, 0], sol.y[:, -1]))))

    return NonlinearSolution(
        success=bool(sol.success),
        message=sol.message,
        t=t,
        x=x,
        p=p,
        u=u,
        H=H,
        bc_residual=bc_res,
        J_u=control_effort(u, R, t),
        n_nodes=int(len(sol.x)),
    )


def run_part4(
    params,
    t_grid,
    Q,
    R,
    Qf,
    x0,
    *,
    dynamics_func=None,
    trim_func=TRIM_FUNC,
    verbose=False,
    enforce_state_nl=False,
):
    """Run all three Part IV scenarios on a common initial condition."""
    dynamics_func = dynamics_func or dyn.get_hover_dynamics

    scenario_linear = ana.run_regulation(
        dynamics_func, t_grid, Q, R, Qf, x0, params, lti=True, enforce_state=False,
    )
    t_lin = scenario_linear["t_eff"]
    P_interp = scenario_linear["P_interp"]
    scenario_linear["J_u"] = control_effort(scenario_linear["u_hist"], R, t_lin)
    scenario_linear["J_total"] = total_running_cost(
        scenario_linear["x_hist"], scenario_linear["u_hist"], Q, R, Qf, t_lin,
    )
    scenario_linear["H"] = hamiltonian_along_lqr(
        dynamics_func, t_lin, scenario_linear["x_hist"], scenario_linear["u_hist"],
        P_interp, Q, R, params, nonlinear_plant=False,
    )

    scenario_lqr_nl = simulate_lqr_on_nonlinear(
        dynamics_func,
        t_grid,
        Q,
        R,
        Qf,
        x0,
        params,
        trim_func=trim_func,
        enforce_state=enforce_state_nl,
    )
    t_nl = scenario_lqr_nl["t"]
    scenario_lqr_nl["J_total"] = total_running_cost(
        scenario_lqr_nl["x"], scenario_lqr_nl["u"], Q, R, Qf, t_nl,
    )
    scenario_lqr_nl["H"] = hamiltonian_along_lqr(
        dynamics_func, t_nl, scenario_lqr_nl["x"], scenario_lqr_nl["u"],
        P_interp, Q, R, params, nonlinear_plant=True,
    )

    p_guess = _costate_guess(t_lin, scenario_linear["x_hist"], P_interp)
    scenario_pmp_nl = solve_nonlinear_tpbvp(
        x0,
        t_lin,
        scenario_linear["x_hist"],
        p_guess,
        Q,
        R,
        Qf,
        params,
        trim_func=trim_func,
        verbose=verbose,
    )
    scenario_pmp_nl.J_total = total_running_cost(
        scenario_pmp_nl.x, scenario_pmp_nl.u, Q, R, Qf, scenario_pmp_nl.t,
    )

    return {
        "scenario_linear": scenario_linear,
        "scenario_lqr_nl": scenario_lqr_nl,
        "scenario_pmp_nl": scenario_pmp_nl,
        "x0": x0,
        "Q": Q,
        "R": R,
        "Qf": Qf,
        "params": params,
        "dynamics_func": dynamics_func,
        "trim_func": trim_func,
    }


def angle_sensitivity(
    params,
    t_grid,
    Q,
    R,
    Qf,
    x0_base,
    theta_scales,
    *,
    dynamics_func=None,
    verbose=False,
):
    """Sweep initial pitch deviation to compare linear vs nonlinear optimality."""
    dynamics_func = dynamics_func or dyn.get_hover_dynamics
    rows = []
    for scale in theta_scales:
        x0 = x0_base.copy()
        x0[4] = x0_base[4] * scale
        try:
            result = run_part4(params, t_grid, Q, R, Qf, x0, dynamics_func=dynamics_func, verbose=verbose)
            sl, sn, sp = result["scenario_linear"], result["scenario_lqr_nl"], result["scenario_pmp_nl"]
            rows.append(
                {
                    "theta0": scale,
                    "J_u_linear": sl["J_u"],
                    "J_u_lqr_nl": sn["J_u"],
                    "J_u_pmp_nl": sp.J_u,
                    "J_total_linear": sl["J_total"],
                    "J_total_lqr_nl": sn["J_total"],
                    "J_total_pmp_nl": sp.J_total,
                    "bc_residual": sp.bc_residual,
                    "bvp_success": sp.success,
                    "terminal_norm_linear": float(np.linalg.norm(sl["x_hist"][-1])),
                    "terminal_norm_lqr_nl": float(np.linalg.norm(sn["x"][-1])),
                    "terminal_norm_pmp_nl": float(np.linalg.norm(sp.x[-1])),
                }
            )
        except Exception as exc:
            rows.append({"theta0": scale, "error": str(exc), "bvp_success": False})
    return rows


def print_part4_report(result):
    """Rubric outputs: BC residual, terminal norms, and cost comparisons."""
    sl = result["scenario_linear"]
    sn = result["scenario_lqr_nl"]
    sp = result["scenario_pmp_nl"]
    print("\n=== Part IV: three-scenario comparison ===")
    print(f"  BVP success         : {sp.success} ({sp.message})")
    print(f"  BC |res|_inf         : {sp.bc_residual:.4e}  (target < 1e-6)")
    print(f"  ||x(tf)|| linear     : {np.linalg.norm(sl['x_hist'][-1]):.4e}")
    print(f"  ||x(tf)|| LQR / nl   : {np.linalg.norm(sn['x'][-1]):.4e}")
    print(f"  ||x(tf)|| PMP / nl   : {np.linalg.norm(sp.x[-1]):.4e}")
    print(f"  J_u  linear / LTI    : {sl['J_u']:.4f}")
    print(f"  J_u  LQR / nonlinear : {sn['J_u']:.4f}")
    print(f"  J_u  PMP / nonlinear : {sp.J_u:.4f}")
    print(f"  J    linear / LTI    : {sl['J_total']:.4f}")
    print(f"  J    LQR / nonlinear : {sn['J_total']:.4f}")
    print(f"  J    PMP / nonlinear : {sp.J_total:.4f}")
    if sn["J_u"] > 0:
        print(f"  PMP control savings  : {100.0 * (1.0 - sp.J_u / sn['J_u']):+.2f}% vs LQR/nonlinear")


def export_part4_figures(result, out_dir, save_figure):
    """Export Part IV three-way comparison figures."""
    import matplotlib.pyplot as plt

    sl = result["scenario_linear"]
    sn = result["scenario_lqr_nl"]
    sp = result["scenario_pmp_nl"]
    x0 = result["x0"]

    xl, tl, ul = sl["x_hist"], sl["t_eff"], sl["u_hist"]
    xn, tn, un = sn["x"], sn["t"], sn["u"]
    xp, tp, up = sp.x, sp.t, sp.u

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xl[:, 0], xl[:, 1], color=SCENARIO_COLORS["linear"], lw=2, label=SCENARIO_LABELS["linear"])
    ax.plot(xn[:, 0], xn[:, 1], color=SCENARIO_COLORS["lqr_nl"], lw=2, label=SCENARIO_LABELS["lqr_nl"])
    ax.plot(xp[:, 0], xp[:, 1], color=SCENARIO_COLORS["pmp_nl"], ls="--", lw=2, label=SCENARIO_LABELS["pmp_nl"])
    ax.scatter([x0[0]], [x0[1]], c="k", s=50, zorder=5, label=r"$x_0$")
    step = max(1, len(tp) // 20)
    ax.quiver(
        xp[::step, 0], xp[::step, 1],
        0.35 * np.cos(xp[::step, 4]), 0.35 * np.sin(xp[::step, 4]),
        angles="xy", scale_units="xy", scale=1, color=SCENARIO_COLORS["pmp_nl"],
        alpha=0.65, width=0.004, label=r"$\theta$ (PMP)",
    )
    ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]", title="Part IV: trajectory comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, f"{out_dir}/p4_mission_plane_quiver.png")

    state_labels = ["p_x", "p_z", "v_x", "v_z", r"$\theta$", r"$\omega$", "m"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for i in range(7):
        ax = axes.flat[i]
        ax.plot(tl, xl[:, i], color=SCENARIO_COLORS["linear"], label=SCENARIO_LABELS["linear"])
        ax.plot(tn, xn[:, i], color=SCENARIO_COLORS["lqr_nl"], label=SCENARIO_LABELS["lqr_nl"])
        ax.plot(tp, xp[:, i], color=SCENARIO_COLORS["pmp_nl"], ls="--", label=SCENARIO_LABELS["pmp_nl"])
        ax.set(title=state_labels[i], xlabel="t [s]")
    axes.flat[7].axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("State trajectories: linear vs LQR/nonlinear vs PMP/nonlinear")
    save_figure(fig, f"{out_dir}/p4_state_trajectories.png", has_suptitle=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, lab in enumerate([r"$\delta T$", r"$\tau$"]):
        axes[j].plot(tl, ul[:, j], color=SCENARIO_COLORS["linear"], label=SCENARIO_LABELS["linear"])
        axes[j].plot(tn, un[:, j], color=SCENARIO_COLORS["lqr_nl"], label=SCENARIO_LABELS["lqr_nl"])
        axes[j].plot(tp, up[:, j], color=SCENARIO_COLORS["pmp_nl"], ls="--", label=SCENARIO_LABELS["pmp_nl"])
        axes[j].set(ylabel=lab)
    axes[1].set(xlabel="t [s]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Control histories")
    save_figure(fig, f"{out_dir}/p4_control_histories.png", has_suptitle=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    tl_h, H_lin = tl, sl["H"]
    axes[0].plot(tl_h, H_lin, color=SCENARIO_COLORS["linear"], lw=1.8, label=SCENARIO_LABELS["linear"])
    if H_lin.size:
        axes[0].axhline(float(np.mean(H_lin)), color="gray", ls=":", lw=0.8, alpha=0.7)
        drift = float(np.max(H_lin) - np.min(H_lin))
        axes[0].text(
            0.02,
            0.02,
            rf"drift={drift:.2e}",
            transform=axes[0].transAxes,
            fontsize=7,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    axes[0].set(
        ylabel=r"$H(t)$",
        title=r"(1) LQR / linear plant — $H = x^\top Q x + u^\top R u + p^\top(Ax+Bu)$, $p=2P(t)x$",
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for key, t_h, H_h, ls in [
        ("lqr_nl", tn, sn["H"], "-"),
        ("pmp_nl", tp, sp.H, "--"),
    ]:
        axes[1].plot(
            t_h,
            H_h,
            color=SCENARIO_COLORS[key],
            lw=1.8,
            ls=ls,
            label=SCENARIO_LABELS[key],
        )
        if H_h.size:
            axes[1].axhline(float(np.mean(H_h)), color=SCENARIO_COLORS[key], ls=":", lw=0.6, alpha=0.45)
            drift = float(np.max(H_h) - np.min(H_h))
            note = rf"{SCENARIO_LABELS[key]}: drift={drift:.2e}"
            if key == "pmp_nl":
                note += rf", $H(t_f)={H_h[-1]:.2e}$"
            axes[1].text(
                0.02 if key == "lqr_nl" else 0.02,
                0.14 if key == "pmp_nl" else 0.02,
                note,
                transform=axes[1].transAxes,
                fontsize=7,
                va="bottom",
                color=SCENARIO_COLORS[key],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )
    axes[1].set(
        xlabel="t [s]",
        ylabel=r"$H(t)$",
        title=r"(2–3) Nonlinear plant — $H = x^\top Q x + u^\top R u + p^\top f_{\mathrm{nl}}(x,u)$",
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Part IV — Hamiltonian evolution by comparison scenario", y=1.01, fontsize=11)
    save_figure(fig, f"{out_dir}/p4_hamiltonian.png", has_suptitle=True)

    term_labels = SCENARIO_LABELS["linear"], SCENARIO_LABELS["lqr_nl"], SCENARIO_LABELS["pmp_nl"]
    xf = [xl[-1], xn[-1], xp[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(term_labels, [np.linalg.norm(x) for x in xf], color=list(SCENARIO_COLORS.values()))
    axes[0].set(ylabel=r"$\|x(t_f)\|$", title="Terminal state norm")
    idx = [0, 1, 4]
    width = 0.25
    xpos = np.arange(len(idx))
    for i, (xf_i, lab, c) in enumerate(zip(xf, term_labels, SCENARIO_COLORS.values())):
        axes[1].bar(xpos + i * width, [xf_i[j] for j in idx], width, color=c, label=lab)
    axes[1].set_xticks(xpos + width)
    axes[1].set_xticklabels(["p_x", "p_z", r"$\theta$"])
    axes[1].set(title="Terminal state components", ylabel="value")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", alpha=0.3)
    save_figure(fig, f"{out_dir}/p4_terminal_behavior.png")

    labels = list(SCENARIO_LABELS.values())
    colors = list(SCENARIO_COLORS.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, [sl["J_u"], sn["J_u"], sp.J_u], color=colors)
    axes[0].set(ylabel=r"$J_u = \int u^\top R u\, dt$", title="Control effort")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, [sl["J_total"], sn["J_total"], sp.J_total], color=colors)
    axes[1].set(ylabel=r"$J = \int (x^\top Q x + u^\top R u)\, dt + x_f^\top Q_f x_f$", title="Total cost")
    axes[1].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, f"{out_dir}/p4_energy_comparison.png")

    return {
        "J_u_linear": sl["J_u"],
        "J_u_lqr_nl": sn["J_u"],
        "J_u_pmp_nl": sp.J_u,
        "J_total_linear": sl["J_total"],
        "J_total_lqr_nl": sn["J_total"],
        "J_total_pmp_nl": sp.J_total,
        "bc_residual": sp.bc_residual,
        "terminal_norm_linear": float(np.linalg.norm(xl[-1])),
        "terminal_norm_lqr_nl": float(np.linalg.norm(xn[-1])),
        "terminal_norm_pmp_nl": float(np.linalg.norm(xp[-1])),
    }
