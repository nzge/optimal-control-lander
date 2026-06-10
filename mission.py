"""
Part III — two-phase mission: minimum-time ascent (Phase A) + free-final-time landing (Phase B).

Indirect single shooting on Z = [p_A(0), p_B(t1), t1, t_f].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import constraints as cst
import dynamics as dyn

Manifold = Literal["M1", "M2"]
N_STATE = 7
TARGET_TOL = 1e-5


@dataclass
class MissionSolution:
    success: bool
    message: str
    manifold: Manifold
    Z: np.ndarray
    shoot_residual: np.ndarray
    shoot_norm: float
    t1: float
    tf: float
    t_a: np.ndarray
    t_b: np.ndarray
    x_a: np.ndarray
    u_a: np.ndarray
    p_a: np.ndarray
    H_a: np.ndarray
    x_b: np.ndarray
    u_b: np.ndarray
    p_b: np.ndarray
    H_b: np.ndarray
    x1: np.ndarray
    H_tf: float


def mission_cost_matrices():
    """Phase B running cost (no 1/2 in integral → factor 2 in adjoint)."""
    Q = np.diag([1.0, 1.0, 0.5, 0.5, 10.0, 5.0, 0.01])
    R = np.diag([0.1, 1.0])
    return Q, R


def _split_xp(z: np.ndarray):
    return z[:N_STATE], z[N_STATE:]


def phase_a_control(t, p, dynamics_func, params, trim_func):
    """Bang-bang thrust; tau held at trim."""
    _, B = dynamics_func(t, params)
    S = B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    u = np.zeros(2)
    u[0] = lo[0] if S[0] > 0.0 else hi[0]
    u[1] = 0.0
    return u


def phase_b_control(t, x, p, dynamics_func, Q, R, params, trim_func):
    _, B = dynamics_func(t, params)
    u_req = -0.5 * np.linalg.inv(R) @ B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def hamiltonian_a(t, x, p, u, dynamics_func, params):
    A, B = dynamics_func(t, params)
    return 1.0 + p @ (A @ x + B @ u)


def hamiltonian_b(t, x, p, u, dynamics_func, Q, R, params):
    A, B = dynamics_func(t, params)
    return x @ Q @ x + u @ R @ u + p @ (A @ x + B @ u)


def _phase_a_rhs(t, z, dynamics_func, params, trim_func):
    x, p = _split_xp(z)
    A, B = dynamics_func(t, params)
    u = phase_a_control(t, p, dynamics_func, params, trim_func)
    return np.concatenate([A @ x + B @ u, -A.T @ p])


def _phase_b_rhs(t, z, dynamics_func, Q, R, params, trim_func):
    x, p = _split_xp(z)
    A, B = dynamics_func(t, params)
    u = phase_b_control(t, x, p, dynamics_func, Q, R, params, trim_func)
    return np.concatenate([A @ x + B @ u, -2.0 * Q @ x - A.T @ p])


def _propagate(rhs, t0, tf, x0, p0, *, args=(), n_eval=200):
    z0 = np.concatenate([x0, p0])
    t_eval = np.linspace(t0, tf, max(n_eval, 2))
    sol = solve_ivp(
        rhs, [t0, tf], z0, t_eval=t_eval, args=args,
        method="RK45", rtol=1e-9, atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[:N_STATE].T, sol.y[N_STATE:].T


def propagate_phase_a(x0, p0, t1, dynamics_func, params, trim_func, *, n_eval=200):
    return _propagate(_phase_a_rhs, 0.0, t1, x0, p0, args=(dynamics_func, params, trim_func), n_eval=n_eval)


def propagate_phase_b(x1, p1, t1, tf, dynamics_func, Q, R, params, trim_func, *, n_eval=200):
    return _propagate(
        _phase_b_rhs, t1, tf, x1, p1,
        args=(dynamics_func, Q, R, params, trim_func), n_eval=n_eval,
    )


def _control_hist_phase_a(t, p_hist, dynamics_func, params, trim_func):
    return np.array([phase_a_control(t[k], p_hist[k], dynamics_func, params, trim_func) for k in range(len(t))])


def _control_hist_phase_b(t, x_hist, p_hist, dynamics_func, Q, R, params, trim_func):
    return np.array([
        phase_b_control(t[k], x_hist[k], p_hist[k], dynamics_func, Q, R, params, trim_func)
        for k in range(len(t))
    ])


def _manifold_eval(x_f, p_f, tf, params, manifold, trim_func, *, use_absolute=False):
    x_eval = cst.absolute_state(x_f, tf, params, trim_func) if use_absolute else x_f
    if manifold == "M1":
        return np.array([x_eval[1], x_eval[3], x_eval[4], p_f[0], p_f[2], p_f[5], p_f[6]])
    h1 = (x_eval[0] - params["p_c"]) ** 2 + x_eval[1] ** 2 - params["r_platform"] ** 2
    collin = p_f[0] * x_eval[1] - p_f[1] * (x_eval[0] - params["p_c"])
    return np.array([h1, x_eval[3], x_eval[4], p_f[2], p_f[5], p_f[6], collin])


def shooting_residuals(
    Z, x0, params, *, manifold: Manifold = "M1", Q=None, R=None,
    dynamics_a=None, dynamics_b=None, trim_a=None, trim_b=None,
    use_absolute_manifold: bool = False, n_eval: int = 120,
):
    if Q is None or R is None:
        Q, R = mission_cost_matrices()
    dynamics_a = dynamics_a or dyn.get_hover_dynamics
    dynamics_b = dynamics_b or dyn.get_hover_dynamics
    trim_a = trim_a or dyn.hover_trim
    trim_b = trim_b or dyn.hover_trim

    p_a0 = Z[:N_STATE]
    p_b1 = Z[N_STATE : 2 * N_STATE]
    t1 = float(Z[2 * N_STATE])
    tf = float(Z[2 * N_STATE + 1])

    if t1 <= 0.05 or tf <= t1 + 0.05:
        return np.full(2 * N_STATE, 1e2)

    try:
        _, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dynamics_a, params, trim_a, n_eval=n_eval)
        x1 = x_a[-1]
        p_a1 = p_a[-1]
        _, x_b, p_b = propagate_phase_b(x1, p_b1, t1, tf, dynamics_b, Q, R, params, trim_b, n_eval=n_eval)
    except RuntimeError:
        return np.full(2 * N_STATE, 1e2)

    x_f, p_f = x_b[-1], p_b[-1]
    u_a1 = phase_a_control(t1, p_a1, dynamics_a, params, trim_a)
    A1, B1 = dynamics_a(t1, params)
    H_a1 = 1.0 + p_a1 @ (A1 @ x1 + B1 @ u_a1)
    u_f = phase_b_control(tf, x_f, p_f, dynamics_b, Q, R, params, trim_b)
    H_f = hamiltonian_b(tf, x_f, p_f, u_f, dynamics_b, Q, R, params)

    residuals = list(p_a1) + [H_a1]
    residuals.extend(_manifold_eval(x_f, p_f, tf, params, manifold, trim_b, use_absolute=use_absolute_manifold))
    residuals.append(H_f)
    return np.asarray(residuals, dtype=float)


def _solve_phase_a(x0, params, *, t1_bounds=(1.0, 2.0)):
    def fun(z):
        p0, t1 = z[:7], float(z[7])
        if t1 < t1_bounds[0]:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_a(x0, p0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=100)
        except RuntimeError:
            return np.ones(8) * 1e2
        p1 = p[-1]
        u1 = phase_a_control(t1, p1, dyn.get_hover_dynamics, params, dyn.hover_trim)
        A, B = dyn.get_hover_dynamics(t1, params)
        return np.concatenate([p1, [1.0 + p1 @ (A @ x[-1] + B @ u1)]])

    z0 = np.zeros(8)
    z0[3] = -0.05
    z0[7] = 1.4
    lb = np.r_[np.full(7, -15), t1_bounds[0]]
    ub = np.r_[np.full(7, 15), t1_bounds[1]]
    res = least_squares(fun, np.clip(z0, lb, ub), bounds=(lb, ub), max_nfev=200, ftol=1e-12, xtol=1e-12)
    return res.x[:7], float(res.x[7]), res


def _solve_phase_b(x1, t1, params, Q, R, *, manifold: Manifold = "M1", tf_bounds=(10.0, 40.0)):
    import lqr

    def fun(z):
        p1, tf = z[:7], float(z[7])
        if tf <= t1 + 0.5:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_b(
                x1, p1, t1, tf, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim, n_eval=100
            )
        except RuntimeError:
            return np.ones(8) * 1e2
        xf, pf = x[-1], p[-1]
        u = phase_b_control(tf, xf, pf, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim)
        H = hamiltonian_b(tf, xf, pf, u, dyn.get_hover_dynamics, Q, R, params)
        term = _manifold_eval(xf, pf, tf, params, manifold, dyn.hover_trim)
        return np.concatenate([term, [H]])

    P, _, _ = lqr.solve_riccati_backward(
        dyn.get_hover_dynamics, np.linspace(0.0, tf_bounds[1], 150), Q, R,
        np.diag([10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1]), params,
    )
    z0 = np.zeros(8)
    z0[:7] = np.clip(P(t1) @ x1, -15, 15)
    z0[7] = max(t1 + 12.0, tf_bounds[0])
    lb = np.r_[np.full(7, -25), t1 + 6.0]
    ub = np.r_[np.full(7, 25), tf_bounds[1]]
    res = least_squares(fun, np.clip(z0, lb, ub), bounds=(lb, ub), max_nfev=300, ftol=1e-12, xtol=1e-12)
    return res.x[:7], float(res.x[7]), res


def _sequential_seed(x0, params, manifold, Q, R, *, t1_seed=1.3):
    """Sequential Phase A → homotopy Phase B → returns Z."""
    p_a0, t1, _ = _solve_phase_a(x0, params, t1_bounds=(max(1.0, t1_seed - 0.25), t1_seed + 0.25))
    p_b1, tf = np.zeros(7), t1 + 12.0
    for q_scale in (0.05, 0.2, 1.0):
        Qs = q_scale * Q
        for _ in range(3):
            _, x_a, _ = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=100)
            p_b1, tf, _ = _solve_phase_b(x_a[-1], t1, params, Qs, R, manifold=manifold)
    Z = np.zeros(2 * N_STATE + 2)
    Z[:N_STATE] = p_a0
    Z[N_STATE : 2 * N_STATE] = p_b1
    Z[2 * N_STATE] = t1
    Z[2 * N_STATE + 1] = tf
    return Z


def _polish(Z, x0, params, manifold, Q, R, *, n_eval=180, max_nfev=600):
    """Full 16-D polish — key step for free-final-time H(tf)=0 accuracy."""

    def fun(z):
        return shooting_residuals(z, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval)

    lb = np.r_[np.full(14, -25), [1.0, 10.0]]
    ub = np.r_[np.full(14, 25), [2.5, params.get("tf_mission_max", 60.0)]]
    z0 = np.clip(Z, lb, ub)
    return least_squares(fun, z0, bounds=(lb, ub), max_nfev=max_nfev, ftol=1e-14, xtol=1e-14)


def _build_solution(Z, x0, params, manifold, Q, R, message="", *, target_tol=TARGET_TOL, n_eval=200):
    shoot_res = shooting_residuals(Z, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval)
    shoot_norm = float(np.linalg.norm(shoot_res, ord=np.inf))
    t1, tf = float(Z[2 * N_STATE]), float(Z[2 * N_STATE + 1])
    p_a0, p_b1 = Z[:N_STATE], Z[N_STATE : 2 * N_STATE]
    t_a, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=n_eval)
    x1 = x_a[-1]
    t_b, x_b, p_b = propagate_phase_b(x1, p_b1, t1, tf, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim, n_eval=n_eval)
    u_a = _control_hist_phase_a(t_a, p_a, dyn.get_hover_dynamics, params, dyn.hover_trim)
    u_b = _control_hist_phase_b(t_b, x_b, p_b, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim)
    H_a = np.array([hamiltonian_a(t_a[k], x_a[k], p_a[k], u_a[k], dyn.get_hover_dynamics, params) for k in range(len(t_a))])
    H_b = np.array([hamiltonian_b(t_b[k], x_b[k], p_b[k], u_b[k], dyn.get_hover_dynamics, Q, R, params) for k in range(len(t_b))])
    return MissionSolution(
        success=shoot_norm <= target_tol,
        message=message,
        manifold=manifold,
        Z=Z,
        shoot_residual=shoot_res,
        shoot_norm=shoot_norm,
        t1=t1, tf=tf,
        t_a=t_a, t_b=t_b, x_a=x_a, u_a=u_a, p_a=p_a, H_a=H_a,
        x_b=x_b, u_b=u_b, p_b=p_b, H_b=H_b, x1=x1,
        H_tf=float(H_b[-1]),
    )


def solve_mission(
    x0, params, *, manifold: Manifold = "M1", Z0=None, verbose=False,
    multi_start=True, target_tol=TARGET_TOL,
):
    Q, R = mission_cost_matrices()
    if Z0 is not None:
        sol = _build_solution(Z0, x0, params, manifold, Q, R, target_tol=target_tol)
        if verbose:
            print(f"Mission ({manifold}): |res|_inf={sol.shoot_norm:.3e}")
        return sol

    seeds = [1.25, 1.4, 1.55] if multi_start else [1.3]
    best_Z, best_norm, best_msg = None, np.inf, ""
    for t1_seed in seeds:
        Z = _sequential_seed(x0, params, manifold, Q, R, t1_seed=t1_seed)
        polish = _polish(Z, x0, params, manifold, Q, R)
        norm = float(np.linalg.norm(polish.fun, ord=np.inf))
        if norm < best_norm:
            best_norm, best_Z, best_msg = norm, polish.x, polish.message

    sol = _build_solution(best_Z, x0, params, manifold, Q, R, best_msg, target_tol=target_tol, n_eval=250)
    if verbose:
        print(
            f"Mission ({manifold}): success={sol.success}, |res|_inf={sol.shoot_norm:.3e}, "
            f"t1={sol.t1:.3f}, tf={sol.tf:.3f}"
        )
    return sol


# --- metrics & figure export ---

def mission_trajectory_plane(sol: MissionSolution):
    """Return phase-A/B px,pz segments for plotting."""
    return (
        sol.x_a[:, 0], sol.x_a[:, 1],
        sol.x_b[:, 0], sol.x_b[:, 1],
        sol.t1,
    )


def phase_a_switching_count(sol: MissionSolution, params: dict):
    """Count thrust bang-bang switches in Phase A."""
    _, B = dyn.get_hover_dynamics(0.0, params)
    S = np.array([B.T @ sol.p_a[k] for k in range(len(sol.t_a))])
    signs = np.sign(S[:, 0])
    return int(np.sum(signs[1:] != signs[:-1]))


def control_energy(sol: MissionSolution, Q, R):
    """Integrated Phase-B running cost approximation."""
    t, u = sol.t_b, sol.u_b
    x = sol.x_b
    stage = np.array([x[k] @ Q @ x[k] + u[k] @ R @ u[k] for k in range(len(t))])
    dt = np.diff(t)
    return float(np.sum(0.5 * (stage[:-1] + stage[1:]) * dt))


def mission_metrics(sol: MissionSolution, params: dict, Q, R):
    term = _manifold_eval(sol.x_b[-1], sol.p_b[-1], sol.tf, params, sol.manifold, dyn.hover_trim)
    return {
        "manifold": sol.manifold,
        "t1": sol.t1,
        "tf": sol.tf,
        "landing_time": sol.tf,
        "phase_b_duration": sol.tf - sol.t1,
        "shoot_norm": sol.shoot_norm,
        "H_tf": sol.H_tf,
        "terminal_manifold_inf": float(np.max(np.abs(term))),
        "control_energy": control_energy(sol, Q, R),
        "phase_a_switches": phase_a_switching_count(sol, params),
        "min_altitude": float(np.min(mission_altitude_history(sol, params))),
    }


def terminal_sensitivity(x0, params, manifold, Q, R, sol: MissionSolution, *, delta=0.05):
    """Max |terminal residual| under small IC perturbations."""
    base = sol.shoot_norm
    worst = base
    rng = np.eye(7)
    for i in range(7):
        for sign in (-1, 1):
            xpert = x0 + sign * delta * rng[i]
            try:
                s2 = solve_mission(xpert, params, manifold=manifold, Z0=sol.Z, target_tol=1e-3, verbose=False)
                worst = max(worst, s2.shoot_norm)
            except Exception:
                worst = max(worst, 1.0)
    return {"base_residual": base, "worst_perturbed": worst, "delta": delta}


def mission_altitude_history(sol: MissionSolution, params: dict, trim_func=None):
    alt_a = np.array([sol.x_a[k, 1] for k in range(len(sol.x_a))])
    alt_b = np.array([sol.x_b[k, 1] for k in range(len(sol.x_b))])
    return np.concatenate([alt_a, alt_b])


def _plot_mission_phases(ax, sol, params, *, label_prefix=""):
    import matplotlib.pyplot as plt

    px_a, pz_a, px_b, pz_b, _ = mission_trajectory_plane(sol)
    ax.plot(px_a, pz_a, "C0-", lw=2, label=f"{label_prefix}Phase A")
    ax.plot(px_b, pz_b, "C1-", lw=2, label=f"{label_prefix}Phase B")
    ax.scatter([px_a[0]], [pz_a[0]], c="C2", s=40, zorder=5)
    ax.scatter([px_b[-1]], [pz_b[-1]], c="C3", s=40, zorder=5)
    if sol.manifold == "M2":
        circ = plt.Circle((params["p_c"], 0), params["r_platform"], fill=False, color="C3", ls="--", alpha=0.6)
        ax.add_patch(circ)


def export_part3_figures(p3_m1, p3_m2, params, out_dir, save_figure):
    """Export Part III manifold comparison and trajectory figures."""
    import matplotlib.pyplot as plt

    sol_m1, sol_m2 = p3_m1["solution"], p3_m2["solution"]
    Q, R = p3_m1["Q"], p3_m1["R"]
    m1 = mission_metrics(sol_m1, params, Q, R)
    m2 = mission_metrics(sol_m2, params, Q, R)
    sens_m1 = terminal_sensitivity(p3_m1["x0"], params, "M1", Q, R, sol_m1)
    sens_m2 = terminal_sensitivity(p3_m2["x0"], params, "M2", Q, R, sol_m2)

    # Individual manifold trajectories (Phase A + B)
    for tag, sol in [("M1", sol_m1), ("M2", sol_m2)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_mission_phases(ax, sol, params)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]", title=f"Part III {tag}: ascent + landing")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        save_figure(fig, f"{out_dir}/p3_trajectory_{tag}.png")

    # Combined M1 + M2 overlay
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_mission_phases(ax, sol_m1, params, label_prefix="M1 ")
    _plot_mission_phases(ax, sol_m2, params, label_prefix="M2 ")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]", title="M1 vs M2 mission planes (Phase A + B)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, f"{out_dir}/p3_trajectory_M1_M2_overlay.png")

    # Comparison bars: landing time, energy, switches, feasibility, sensitivity
    labels = ["M1 flat", "M2 platform"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].bar(labels, [m1["landing_time"], m2["landing_time"]], color=["C0", "C1"])
    axes[0, 0].set(title="Total landing time $t_f$ [s]", ylabel="s")
    axes[0, 1].bar(labels, [m1["phase_b_duration"], m2["phase_b_duration"]], color=["C0", "C1"])
    axes[0, 1].set(title="Phase B duration [s]")
    axes[0, 2].bar(labels, [m1["control_energy"], m2["control_energy"]], color=["C0", "C1"])
    axes[0, 2].set(title="Phase B control energy")
    axes[1, 0].bar(labels, [m1["phase_a_switches"], m2["phase_a_switches"]], color=["C0", "C1"])
    axes[1, 0].set(title="Phase A thrust switches")
    axes[1, 1].bar(labels, [m1["shoot_norm"], m2["shoot_norm"]], color=["C0", "C1"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(TARGET_TOL, color="k", ls="--", lw=0.8, label="target $10^{-5}$")
    axes[1, 1].set(title=r"Shooting $\|res\|_\infty$ (feasibility)")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].bar(labels, [sens_m1["worst_perturbed"], sens_m2["worst_perturbed"]], color=["C0", "C1"])
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(title=r"Terminal sensitivity ($\pm\delta x_0$)")
    fig.suptitle("Part III landing manifold comparison")
    save_figure(fig, f"{out_dir}/p3_manifold_comparison.png", has_suptitle=True)

    # Switching structure: Phase A S(t) and controls
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, (sol, name) in enumerate([(sol_m1, "M1"), (sol_m2, "M2")]):
        _, B = dyn.get_hover_dynamics(0.0, params)
        S_a = np.array([B.T @ sol.p_a[k] for k in range(len(sol.t_a))])
        axes[0, col].plot(sol.t_a, S_a[:, 0], "C0", label=r"$S_{\delta T}$")
        axes[0, col].axhline(0, color="k", lw=0.6)
        axes[0, col].set(title=f"{name}: Phase A switching fn", ylabel=r"$B^\top p$")
        t_full = np.concatenate([sol.t_a, sol.t_b])
        u_full = np.vstack([sol.u_a, sol.u_b])
        axes[1, col].plot(t_full, u_full[:, 0], label=r"$\delta T$")
        axes[1, col].plot(t_full, u_full[:, 1], label=r"$\tau$", alpha=0.8)
        axes[1, col].axvline(sol.t1, color="gray", ls=":", label="handoff")
        axes[1, col].set(xlabel="t [s]", title=f"{name}: controls", ylabel="N / N·m")
        axes[1, col].legend(fontsize=7)
    save_figure(fig, f"{out_dir}/p3_switching_controls.png")

    # Hamiltonian traces both manifolds
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for sol, c, name in [(sol_m1, "C0", "M1"), (sol_m2, "C1", "M2")]:
        axes[0].plot(sol.t_a, sol.H_a, color=c, label=f"{name} $H_A$")
        axes[1].plot(sol.t_b, sol.H_b, color=c, label=f"{name} $H_B$")
    axes[0].set(ylabel=r"$H_A$", title="Hamiltonian consistency")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(xlabel="t [s]", ylabel=r"$H_B$")
    axes[0].legend()
    axes[1].legend()
    save_figure(fig, f"{out_dir}/p3_hamiltonian.png")

    return {"M1": m1, "M2": m2, "sens_M1": sens_m1, "sens_M2": sens_m2}


# keep legacy single-solution export for notebook use
def export_mission_figures(sol, params, out_dir, save_figure):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_mission_phases(ax, sol, params)
    ax.set(xlabel=r"$p_x$", ylabel=r"$p_z$", title=f"Mission ({sol.manifold})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, f"{out_dir}/p3_mission_plane.png")
