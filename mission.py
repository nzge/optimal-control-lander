"""
Part III — two-phase mission: minimum-time ascent (Phase A) + free-final-time landing (Phase B).

Indirect single shooting on the 16-parameter vector
  Z = [p_A(0), p_B(t1), t1, t_f].
State / costate order: [p_x, p_z, v_x, v_z, theta, omega, m].
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


@dataclass
class MissionSolution:
    """Solved two-phase mission trajectory."""

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
    """Phase B running cost (no 1/2 factors — adjoint uses factor 2)."""
    Q = np.diag([0.1, 0.1, 0.05, 0.05, 1.0, 0.5, 0.001])
    R = np.diag([0.01, 0.1])
    return Q, R


def _split_xp(z: np.ndarray):
    return z[:N_STATE], z[N_STATE:]


def _dynamics_at(t: float, x: np.ndarray, dynamics_func: Callable, params: dict):
    return dynamics_func(t, params)


def phase_a_control(t: float, p: np.ndarray, dynamics_func: Callable, params: dict, trim_func: Callable):
    """Bang-bang thrust; torque held at trim (tau=0) to keep ascent locally valid."""
    A, B = _dynamics_at(t, np.zeros(N_STATE), dynamics_func, params)
    S = B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    u = np.zeros(2)
    u[0] = lo[0] if S[0] > 0.0 else hi[0]
    u[1] = 0.0
    return u


def phase_b_control(
    t: float,
    x: np.ndarray,
    p: np.ndarray,
    dynamics_func: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    params: dict,
    trim_func: Callable,
):
    """Saturated LQR law: u* = -0.5 R^{-1} B' p."""
    A, B = _dynamics_at(t, x, dynamics_func, params)
    Rinv = np.linalg.inv(R)
    u_req = -0.5 * Rinv @ B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def hamiltonian_a(t: float, x: np.ndarray, p: np.ndarray, u: np.ndarray, dynamics_func: Callable, params: dict):
    A, B = _dynamics_at(t, x, dynamics_func, params)
    return 1.0 + p @ (A @ x + B @ u)


def hamiltonian_b(
    t: float,
    x: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    dynamics_func: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    params: dict,
):
    A, B = _dynamics_at(t, x, dynamics_func, params)
    return x @ Q @ x + u @ R @ u + p @ (A @ x + B @ u)


def _phase_a_rhs(t, z, dynamics_func, params, trim_func):
    x, p = _split_xp(z)
    A, B = _dynamics_at(t, x, dynamics_func, params)
    u = phase_a_control(t, p, dynamics_func, params, trim_func)
    return np.concatenate([A @ x + B @ u, -A.T @ p])


def _phase_b_rhs(t, z, dynamics_func, Q, R, params, trim_func):
    x, p = _split_xp(z)
    A, B = _dynamics_at(t, x, dynamics_func, params)
    u = phase_b_control(t, x, p, dynamics_func, Q, R, params, trim_func)
    return np.concatenate([A @ x + B @ u, -2.0 * Q @ x - A.T @ p])


def _propagate(
    rhs,
    t0: float,
    tf: float,
    x0: np.ndarray,
    p0: np.ndarray,
    *,
    args=(),
    n_eval: int = 200,
):
    z0 = np.concatenate([x0, p0])
    t_eval = np.linspace(t0, tf, max(n_eval, 2))
    sol = solve_ivp(
        rhs,
        [t0, tf],
        z0,
        t_eval=t_eval,
        args=args,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    x_hist = sol.y[:N_STATE].T
    p_hist = sol.y[N_STATE:].T
    return sol.t, x_hist, p_hist


def propagate_phase_a(
    x0: np.ndarray,
    p0: np.ndarray,
    t1: float,
    dynamics_func: Callable,
    params: dict,
    trim_func: Callable,
    *,
    n_eval: int = 200,
):
    return _propagate(
        _phase_a_rhs,
        0.0,
        t1,
        x0,
        p0,
        args=(dynamics_func, params, trim_func),
        n_eval=n_eval,
    )


def propagate_phase_b(
    x1: np.ndarray,
    p1: np.ndarray,
    t1: float,
    tf: float,
    dynamics_func: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    params: dict,
    trim_func: Callable,
    *,
    n_eval: int = 200,
):
    return _propagate(
        _phase_b_rhs,
        t1,
        tf,
        x1,
        p1,
        args=(dynamics_func, Q, R, params, trim_func),
        n_eval=n_eval,
    )


def _control_hist_phase_a(t, p_hist, dynamics_func, params, trim_func):
    return np.array([phase_a_control(t[k], p_hist[k], dynamics_func, params, trim_func) for k in range(len(t))])


def _control_hist_phase_b(t, x_hist, p_hist, dynamics_func, Q, R, params, trim_func):
    return np.array(
        [
            phase_b_control(t[k], x_hist[k], p_hist[k], dynamics_func, Q, R, params, trim_func)
            for k in range(len(t))
        ]
    )


def _absolute_terminal(x_dev: np.ndarray, t: float, trim_func: Callable, params: dict):
    return cst.absolute_state(x_dev, t, params, trim_func)


def shooting_residuals(
    Z: np.ndarray,
    x0: np.ndarray,
    params: dict,
    *,
    manifold: Manifold = "M1",
    Q=None,
    R=None,
    dynamics_a=None,
    dynamics_b=None,
    trim_a=None,
    trim_b=None,
    use_absolute_manifold: bool = False,
):
    """16-element residual vector for scipy.optimize.root / least_squares."""
    if Q is None or R is None:
        Q, R = mission_cost_matrices()
    if dynamics_a is None:
        dynamics_a = dyn.get_hover_dynamics
    if dynamics_b is None:
        dynamics_b = dyn.get_hover_dynamics
    if trim_a is None:
        trim_a = dyn.hover_trim
    if trim_b is None:
        trim_b = dyn.hover_trim

    p_a0 = Z[:N_STATE]
    p_b1 = Z[N_STATE : 2 * N_STATE]
    t1 = float(Z[2 * N_STATE])
    tf = float(Z[2 * N_STATE + 1])

    if t1 <= 0.05 or tf <= t1 + 0.05:
        return np.full(2 * N_STATE, 1e2)

    try:
        _, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dynamics_a, params, trim_a, n_eval=80)
        x1 = x_a[-1]
        p_a1 = p_a[-1]
        _, x_b, p_b = propagate_phase_b(x1, p_b1, t1, tf, dynamics_b, Q, R, params, trim_b, n_eval=80)
    except RuntimeError:
        return np.full(2 * N_STATE, 1e2)

    x_f = x_b[-1]
    p_f = p_b[-1]
    if use_absolute_manifold:
        x_eval = _absolute_terminal(x_f, tf, trim_b, params)
    else:
        x_eval = x_f

    u_a1 = phase_a_control(t1, p_a1, dynamics_a, params, trim_a)
    A1, B1 = _dynamics_at(t1, x1, dynamics_a, params)
    H_a1 = 1.0 + p_a1 @ (A1 @ x1 + B1 @ u_a1)

    u_f = phase_b_control(tf, x_f, p_f, dynamics_b, Q, R, params, trim_b)
    H_f = hamiltonian_b(tf, x_f, p_f, u_f, dynamics_b, Q, R, params)

    residuals = list(p_a1)
    residuals.append(H_a1)

    if manifold == "M1":
        residuals.extend([x_eval[1], x_eval[3], x_eval[4]])
        residuals.extend([p_f[0], p_f[2], p_f[5], p_f[6]])
    else:
        h1 = (x_eval[0] - params["p_c"]) ** 2 + x_eval[1] ** 2 - params["r_platform"] ** 2
        collin = p_f[0] * x_eval[1] - p_f[1] * (x_eval[0] - params["p_c"])
        residuals.extend([h1, x_eval[3], x_eval[4]])
        residuals.extend([p_f[2], p_f[5], p_f[6], collin])

    residuals.append(H_f)
    return np.asarray(residuals, dtype=float)


def _solve_phase_a(x0, params, *, t1_bounds=(1.0, 2.0)):
    """8-D subproblem: p_A(0) and t1 with p(t1)=0, H_A(t1)=0."""

    def fun(z):
        p0 = z[:7]
        t1 = float(z[7])
        if t1 < t1_bounds[0]:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_a(x0, p0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=80)
        except RuntimeError:
            return np.ones(8) * 1e2
        p1 = p[-1]
        u1 = phase_a_control(t1, p1, dyn.get_hover_dynamics, params, dyn.hover_trim)
        A, B = dyn.get_hover_dynamics(t1, params)
        return np.concatenate([p1, [1.0 + p1 @ (A @ x[-1] + B @ u1)]])

    z0 = np.zeros(8)
    z0[3] = -0.05
    z0[7] = 1.5
    lb = np.r_[np.full(7, -15), t1_bounds[0]]
    ub = np.r_[np.full(7, 15), t1_bounds[1]]
    z0 = np.clip(z0, lb, ub)
    res = least_squares(fun, z0, bounds=(lb, ub), max_nfev=150)
    return res.x[:7], float(res.x[7]), res


def _solve_phase_b(x1, t1, params, Q, R, *, tf_bounds=(8.0, 35.0)):
    """8-D subproblem: p_B(t1) and tf with M1 manifold + H_B(tf)=0."""

    def fun(z):
        p1 = z[:7]
        tf = float(z[7])
        if tf <= t1 + 0.2:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_b(
                x1, p1, t1, tf, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim, n_eval=80
            )
        except RuntimeError:
            return np.ones(8) * 1e2
        xf, pf = x[-1], p[-1]
        u = phase_b_control(tf, xf, pf, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim)
        H = hamiltonian_b(tf, xf, pf, u, dyn.get_hover_dynamics, Q, R, params)
        return np.array([xf[1], xf[3], xf[4], pf[0], pf[2], pf[5], pf[6], H])

    import lqr

    P, _, _ = lqr.solve_riccati_backward(
        dyn.get_hover_dynamics,
        np.linspace(0.0, tf_bounds[1], 120),
        Q,
        R,
        np.diag([10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1]),
        params,
    )
    z0 = np.zeros(8)
    z0[:7] = np.clip(P(t1) @ x1, -15, 15)
    z0[7] = max(t1 + 15.0, tf_bounds[0])
    lb = np.r_[np.full(7, -20), t1 + 8.0]
    ub = np.r_[np.full(7, 20), tf_bounds[1]]
    z0 = np.clip(z0, lb, ub)
    res = least_squares(fun, z0, bounds=(lb, ub), max_nfev=200)
    return res.x[:7], float(res.x[7]), res


def _build_solution(Z, x0, params, manifold, Q, R, dynamics_a, dynamics_b, trim_a, trim_b, message=""):
    t1 = float(Z[2 * N_STATE])
    tf = float(Z[2 * N_STATE + 1])
    shoot_res = shooting_residuals(
        Z, x0, params, manifold=manifold, Q=Q, R=R,
        dynamics_a=dynamics_a, dynamics_b=dynamics_b, trim_a=trim_a, trim_b=trim_b,
    )
    shoot_norm = float(np.linalg.norm(shoot_res, ord=np.inf))
    p_a0 = Z[:N_STATE]
    p_b1 = Z[N_STATE : 2 * N_STATE]
    t_a, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dynamics_a, params, trim_a)
    x1 = x_a[-1]
    t_b, x_b, p_b = propagate_phase_b(x1, p_b1, t1, tf, dynamics_b, Q, R, params, trim_b)
    u_a = _control_hist_phase_a(t_a, p_a, dynamics_a, params, trim_a)
    u_b = _control_hist_phase_b(t_b, x_b, p_b, dynamics_b, Q, R, params, trim_b)
    H_a = np.array([hamiltonian_a(t_a[k], x_a[k], p_a[k], u_a[k], dynamics_a, params) for k in range(len(t_a))])
    H_b = np.array(
        [hamiltonian_b(t_b[k], x_b[k], p_b[k], u_b[k], dynamics_b, Q, R, params) for k in range(len(t_b))]
    )
    return MissionSolution(
        success=shoot_norm < 5e-2,
        message=message,
        manifold=manifold,
        Z=Z,
        shoot_residual=shoot_res,
        shoot_norm=shoot_norm,
        t1=t1,
        tf=tf,
        t_a=t_a,
        t_b=t_b,
        x_a=x_a,
        u_a=u_a,
        p_a=p_a,
        H_a=H_a,
        x_b=x_b,
        u_b=u_b,
        p_b=p_b,
        H_b=H_b,
        x1=x1,
        H_tf=float(H_b[-1]),
    )


def solve_mission(
    x0: np.ndarray,
    params: dict,
    *,
    manifold: Manifold = "M1",
    Z0: np.ndarray | None = None,
    verbose: bool = False,
    multi_start: bool = True,
):
    """Sequential Phase-A / Phase-B shooting with optional costate polish."""
    Q, R = mission_cost_matrices()
    dynamics_a = dyn.get_hover_dynamics
    dynamics_b = dyn.get_hover_dynamics
    trim_a = dyn.hover_trim
    trim_b = dyn.hover_trim

    if Z0 is not None:
        sol = _build_solution(Z0, x0, params, manifold, Q, R, dynamics_a, dynamics_b, trim_a, trim_b)
        if verbose:
            print(f"Mission ({manifold}): |res|_inf={sol.shoot_norm:.3e}, t1={sol.t1:.3f}, tf={sol.tf:.3f}")
        return sol

    candidates = []
    t1_opts = [1.2, 1.5] if multi_start else [1.3]
    for t1_seed in t1_opts:
        p_a0, t1, _ = _solve_phase_a(x0, params, t1_bounds=(max(1.0, t1_seed - 0.2), t1_seed + 0.2))
        for _ in range(4):
            _, x_a, _ = propagate_phase_a(x0, p_a0, t1, dynamics_a, params, trim_a, n_eval=80)
            p_b1, tf, _ = _solve_phase_b(x_a[-1], t1, params, Q, R)
        Z = np.zeros(2 * N_STATE + 2)
        Z[:N_STATE] = p_a0
        Z[N_STATE : 2 * N_STATE] = p_b1
        Z[2 * N_STATE] = t1
        Z[2 * N_STATE + 1] = tf

        def fun_costates(pc):
            Zc = Z.copy()
            Zc[: 2 * N_STATE] = pc
            return shooting_residuals(
                Zc, x0, params, manifold=manifold, Q=Q, R=R,
                dynamics_a=dynamics_a, dynamics_b=dynamics_b, trim_a=trim_a, trim_b=trim_b,
            )

        polish = least_squares(fun_costates, Z[: 2 * N_STATE], max_nfev=300)
        Z[: 2 * N_STATE] = polish.x
        candidates.append(
            _build_solution(Z, x0, params, manifold, Q, R, dynamics_a, dynamics_b, trim_a, trim_b, polish.message)
        )

    sol = min(candidates, key=lambda s: s.shoot_norm)
    if verbose:
        print(
            f"Mission ({manifold}): success={sol.success}, |res|_inf={sol.shoot_norm:.3e}, "
            f"t1={sol.t1:.3f}, tf={sol.tf:.3f}"
        )
    return sol


def export_mission_figures(sol: MissionSolution, params: dict, out_dir: str, save_figure):
    """Write Part III trajectory and Hamiltonian diagnostic figures."""
    import matplotlib.pyplot as plt

    t = np.concatenate([sol.t_a, sol.t_b])
    px = np.concatenate([sol.x_a[:, 0], sol.x_b[:, 0]])
    pz = np.concatenate([sol.x_a[:, 1], sol.x_b[:, 1]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(px, pz, "k-", lw=2)
    ax.axvline(0, color="C3", ls="--", alpha=0.5, label="launch/land origin")
    if sol.manifold == "M2":
        circ = plt.Circle((params["p_c"], 0), params["r_platform"], fill=False, color="C1", ls="--")
        ax.add_patch(circ)
    ax.scatter([px[0]], [pz[0]], c="C2", s=50, zorder=5, label="start")
    ax.scatter([px[-1]], [pz[-1]], c="C3", s=50, zorder=5, label="terminal")
    ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]", title=f"Part III mission ({sol.manifold})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, f"{out_dir}/p3_mission_plane.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(sol.t_a, sol.H_a, "C0", label=r"$H_A$")
    axes[0].axvline(sol.t1, color="gray", ls=":", label="handoff")
    axes[0].set(ylabel=r"$H_A$", title="Hamiltonian consistency")
    axes[1].plot(sol.t_b, sol.H_b, "C3", label=r"$H_B$")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="t [s]", ylabel=r"$H_B$")
    axes[0].legend()
    save_figure(fig, f"{out_dir}/p3_hamiltonian.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    u = np.vstack([sol.u_a, sol.u_b])
    axes[0].plot(t, u[:, 0], "C0")
    axes[1].plot(t, u[:, 1], "C1")
    axes[0].axvline(sol.t1, color="gray", ls=":")
    axes[0].set(ylabel=r"$\delta T$ [N]")
    axes[1].set(ylabel=r"$\tau$ [N m]", xlabel="t [s]")
    save_figure(fig, f"{out_dir}/p3_controls.png")


def mission_altitude_history(sol: MissionSolution, params: dict, trim_func=None):
    """Absolute p_z along the full mission (deviation + trim)."""
    trim_func = trim_func or dyn.descent_trim
    alt_a = np.array([sol.x_a[k, 1] for k in range(len(sol.x_a))])
    alt_b = np.array([sol.x_b[k, 1] for k in range(len(sol.x_b))])
    return np.concatenate([alt_a, alt_b])
