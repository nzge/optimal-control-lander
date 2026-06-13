"""
Part III — two-phase mission on the linearized (hover) dynamics.

Phase A: minimum-time ascent to a *fixed* rendezvous point x_sky (bang-bang
         thrust), solved as an indirect single-shooting TPBVP.
Phase B: free-final-time landing onto a terminal manifold, solved as a true
         indirect PMP boundary-value problem via collocation (scipy.solve_bvp)
         with t_f as an unknown parameter and the free-time condition H(t_f)=0
         supplied as the closing boundary condition.

Both phases satisfy the PMP optimality conditions (stationarity / bang-bang,
adjoint equations, manifold endpoint + transversality conditions, and the
Hamiltonian terminal condition) to numerical tolerance.

Design note (terminal geometry): Phase A is a vertical ascent to x_sky above
the origin.  M1 (flat ground) lands at the origin (p_x,p_z)=(0,0); M2 (circular
platform centred at (p_c,0)=(0,0)) lands on the platform apex (0, r).  Both
therefore share the same horizontal footprint, isolating the effect of the
terminal-manifold geometry on the optimal trajectory (the only difference being
the landing height r of the platform).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import least_squares

import constraints as cst
import dynamics as dyn

Manifold = Literal["M1", "M2"]
N_STATE = 7
TARGET_TOL = 1e-5

PZ_SKY = 8.0  # rendezvous altitude (hover-deviation coords) [m]
VZ_SKY_SWEEP = (-2.0, 0.0, 2.0)  # terminal-velocity comparison at x_sky [m/s]


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
    u_a: np.ndarray
    x_a: np.ndarray
    p_a: np.ndarray
    H_a: np.ndarray
    t_b: np.ndarray
    x_b: np.ndarray
    u_b: np.ndarray
    p_b: np.ndarray
    H_b: np.ndarray
    x1: np.ndarray
    H_tf: float
    x_sky: np.ndarray = field(default_factory=lambda: np.zeros(N_STATE))
    vz_sky: float = 0.0
    phase_a_norm: float = 0.0
    phase_b_norm: float = 0.0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def mission_cost_matrices(manifold: Manifold = "M1"):
    """Phase B running cost weights (no 1/2 in integral -> factor 2 in adjoint)."""
    Q = np.diag([0.5, 8.0, 0.5, 3.0, 8.0, 4.0, 0.01])
    R = np.diag([0.05, 1.0])
    if manifold == "M2":
        Q[0, 0] = 4.0
        Q[1, 1] = 15.0
        Q[3, 3] = 5.0
    return Q, R


def mission_x_sky(params, vz_sky: float = 0.0) -> np.ndarray:
    """Fixed Phase-A rendezvous target (px,pz,vx,vz,theta,omega free-mass)."""
    pz = float(params.get("pz_sky", PZ_SKY))
    return np.array([0.0, pz, 0.0, float(vz_sky), 0.0, 0.0, 0.0])


def platform_geometry(params):
    return float(params.get("p_c", 0.0)), float(params.get("r_platform", 5.0))


# --------------------------------------------------------------------------
# Phase A — minimum-time ascent to fixed x_sky (bang-bang thrust)
# --------------------------------------------------------------------------

def phase_a_control(t, p, dynamics_func, params, trim_func):
    """PMP minimiser of H_A=1+p'(Ax+Bu): bang-bang thrust; torque at trim.

    The rotational subsystem is decoupled and starts/ends at theta=omega=0, so
    the torque switching function S_tau = (B'p)_2 = p_omega/I is singular and the
    optimal torque is the trim value tau*=0 (no attitude excitation is needed for
    a vertical ascent)."""
    _, B = dynamics_func(t, params)
    S = B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    u = np.zeros(2)
    u[0] = lo[0] if S[0] > 0.0 else hi[0]
    u[1] = 0.0
    return u


def hamiltonian_a(t, x, p, u, dynamics_func, params):
    A, B = dynamics_func(t, params)
    return 1.0 + p @ (A @ x + B @ u)


def _phase_a_rhs(t, z, dynamics_func, params, trim_func):
    x, p = z[:N_STATE], z[N_STATE:]
    A, B = dynamics_func(t, params)
    u = phase_a_control(t, p, dynamics_func, params, trim_func)
    return np.concatenate([A @ x + B @ u, -A.T @ p])


def propagate_phase_a(x0, p0, t1, dynamics_func, params, trim_func, *, n_eval=200):
    z0 = np.concatenate([x0, p0])
    t_eval = np.linspace(0.0, t1, max(n_eval, 2))
    sol = solve_ivp(
        _phase_a_rhs, [0.0, t1], z0, t_eval=t_eval,
        args=(dynamics_func, params, trim_func),
        method="RK45", rtol=1e-10, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[:N_STATE].T, sol.y[N_STATE:].T


def _phase_a_residual(z, x0, x_sky, params, *, n_eval=120):
    """8 residuals: x(t1)[0:6]=x_sky[0:6], p_m(t1)=0 (free mass), H_A(t1)=0."""
    p0, t1 = z[:N_STATE], float(z[N_STATE])
    if t1 <= 0.05:
        return np.full(8, 1e3)
    try:
        _, x, p = propagate_phase_a(x0, p0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=n_eval)
    except RuntimeError:
        return np.full(8, 1e3)
    xf, pf = x[-1], p[-1]
    u = phase_a_control(t1, pf, dyn.get_hover_dynamics, params, dyn.hover_trim)
    H = hamiltonian_a(t1, xf, pf, u, dyn.get_hover_dynamics, params)
    return np.concatenate([xf[:6] - x_sky[:6], [pf[6], H]])


def solve_phase_a(x0, x_sky, params, *, n_starts=10, t1_bounds=(0.3, 5.0), seed=0):
    """Minimum-time TPBVP to a fixed x_sky via multi-start least squares."""
    rng = np.random.default_rng(seed)
    lb = np.r_[np.full(N_STATE, -30.0), t1_bounds[0]]
    ub = np.r_[np.full(N_STATE, 30.0), t1_bounds[1]]
    seeds = [np.r_[np.zeros(N_STATE), 2.2], np.r_[np.array([0, -0.3, 0, -0.2, 0, 0, 0.0]), 2.0]]
    for _ in range(max(0, n_starts - len(seeds))):
        seeds.append(np.r_[rng.uniform(-3.0, 3.0, N_STATE), rng.uniform(*t1_bounds)])

    best, best_norm = None, np.inf
    for z0 in seeds:
        res = least_squares(
            _phase_a_residual, np.clip(z0, lb, ub), bounds=(lb, ub),
            args=(x0, x_sky, params), max_nfev=400, ftol=1e-14, xtol=1e-14,
        )
        norm = float(np.linalg.norm(res.fun, ord=np.inf))
        if norm < best_norm:
            best, best_norm = res, norm
        if best_norm <= 1e-9:
            break
    return best.x[:N_STATE], float(best.x[N_STATE]), best_norm


# --------------------------------------------------------------------------
# Phase B — free-final-time landing (true PMP collocation)
# --------------------------------------------------------------------------

def phase_b_control(t, x, p, dynamics_func, Q, R, params, trim_func):
    """Stationarity u*=-1/2 R^{-1} B' p, saturated to the admissible control set."""
    _, B = dynamics_func(t, params)
    u_req = -0.5 * np.linalg.inv(R) @ B.T @ p
    lo, hi = cst.control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def hamiltonian_b(t, x, p, u, dynamics_func, Q, R, params, **_kw):
    A, B = dynamics_func(t, params)
    return x @ Q @ x + u @ R @ u + p @ (A @ x + B @ u)


def _phase_b_u_vec(B, R, lo, hi, p_mat):
    """Vectorised saturated control for a (7,N) costate block."""
    u_req = -0.5 * np.linalg.inv(R) @ (B.T @ p_mat)
    return np.clip(u_req, lo[:, None], hi[:, None])


def _manifold_residual(xf, pf, manifold, params):
    """Terminal manifold endpoint + transversality residuals (7 components)."""
    p_c, r = platform_geometry(params)
    if manifold == "M1":
        # M1 = {p_z=0, v_z=0, theta=0}; free p_x,v_x,omega,m -> their costates vanish.
        return np.array([xf[1], xf[3], xf[4], pf[0], pf[2], pf[5], pf[6]])
    # M2 = {h1=0, v_z=0, theta=0}; free v_x,omega,m + costate normal to circle.
    h1 = (xf[0] - p_c) ** 2 + xf[1] ** 2 - r ** 2
    collin = pf[0] * xf[1] - pf[1] * (xf[0] - p_c)
    return np.array([h1, xf[3], xf[4], pf[2], pf[5], pf[6], collin])


def _phase_b_bvp(x_sky, manifold, params, Q, R):
    """Build (fun, bc) for solve_bvp on s in [0,1], t=s*tf, parameter p=[tf]."""
    A, B = dyn.get_hover_dynamics(0.0, params)
    lo, hi = cst.control_bounds(0.0, params, dyn.hover_trim)

    def fun(s, y, p):
        tf = p[0]
        x, pc = y[:N_STATE], y[N_STATE:]
        u = _phase_b_u_vec(B, R, lo, hi, pc)
        dx = A @ x + B @ u
        dp = -2.0 * (Q @ x) - A.T @ pc
        return tf * np.vstack([dx, dp])

    def bc(ya, yb, p):
        xf, pf = yb[:N_STATE], yb[N_STATE:]
        u_f = np.clip(-0.5 * np.linalg.inv(R) @ (B.T @ pf), lo, hi)
        H_f = hamiltonian_b(p[0], xf, pf, u_f, dyn.get_hover_dynamics, Q, R, params)
        return np.concatenate([
            ya[:N_STATE] - x_sky,                       # x(t1) = x_sky (handoff)
            _manifold_residual(xf, pf, manifold, params),  # manifold + transversality
            [H_f],                                      # free-time condition H(tf)=0
        ])

    return fun, bc


def _phase_b_cost(x_b, p_b, t_b, Q, R, params):
    """Phase-B objective J = int (x'Qx + u'Ru) dt along a converged rollout."""
    A, B = dyn.get_hover_dynamics(0.0, params)
    lo, hi = cst.control_bounds(0.0, params, dyn.hover_trim)
    Rinv = np.linalg.inv(R)
    stage = np.empty(len(t_b))
    for k in range(len(t_b)):
        u = np.clip(-0.5 * Rinv @ (B.T @ p_b[k]), lo, hi)
        stage[k] = x_b[k] @ Q @ x_b[k] + u @ R @ u
    return float(np.trapezoid(stage, t_b))


def solve_phase_b(x_sky, manifold, params, Q, R, *, tf_guesses=None, n_nodes=60,
                  ground_tol=1e-3):
    """Free-final-time landing BVP.

    Multiple t_f local optima satisfy the PMP necessary conditions; among the
    converged, ground-respecting candidates we return the global *minimiser* of
    the objective J (the true optimum), falling back to least cost overall.
    Returns (t_b, x_b, p_b, tf, bc_norm)."""
    p_c, r = platform_geometry(params)
    if tf_guesses is None:
        tf_guesses = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
    x_end = (np.array([0.0, 0.0, 0, 0, 0, 0, x_sky[6]]) if manifold == "M1"
             else np.array([p_c, r, 0, 0, 0, 0, x_sky[6]]))
    fun, bc = _phase_b_bvp(x_sky, manifold, params, Q, R)

    candidates = []
    for n_try in (n_nodes, 2 * n_nodes):  # refine node count if the coarse pass fails
        for tf0 in tf_guesses:
            s = np.linspace(0.0, 1.0, n_try)
            y0 = np.vstack([np.outer(x_sky, 1 - s) + np.outer(x_end, s), np.zeros((N_STATE, n_try))])
            try:
                sol = solve_bvp(fun, bc, s, y0, p=[tf0], max_nodes=40000, tol=1e-8)
            except Exception:
                continue
            if not sol.success or sol.p[0] <= 0.2:
                continue
            tf = float(sol.p[0])
            ss = np.linspace(0.0, 1.0, 300)
            Y = sol.sol(ss)
            x_b, p_b, t_b = Y[:N_STATE].T, Y[N_STATE:].T, ss * tf
            bc_norm = float(np.max(np.abs(bc(Y[:, 0], Y[:, -1], sol.p))))
            if bc_norm > 1e-5:
                continue
            J = _phase_b_cost(x_b, p_b, t_b, Q, R, params)
            candidates.append({"bc": bc_norm, "t_b": t_b, "x_b": x_b, "p_b": p_b,
                               "tf": tf, "J": J, "min_alt": float(np.min(x_b[:, 1]))})
        if candidates:
            break
    if not candidates:
        raise RuntimeError(f"Phase B BVP did not converge for {manifold}")

    feasible = [c for c in candidates if c["min_alt"] >= -ground_tol]
    pool = feasible or candidates
    best = min(pool, key=lambda c: c["J"])  # global minimiser of the objective
    return best["t_b"], best["x_b"], best["p_b"], best["tf"], best["bc"]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _control_hist_phase_a(t, p, params):
    return np.array([
        phase_a_control(t[k], p[k], dyn.get_hover_dynamics, params, dyn.hover_trim)
        for k in range(len(t))
    ])


def _control_hist_phase_b(t, x, p, Q, R, params):
    return np.array([
        phase_b_control(t[k], x[k], p[k], dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim)
        for k in range(len(t))
    ])


def solve_mission(
    x0, params, *, manifold: Manifold = "M1", vz_sky: float = 0.0,
    Z0=None, verbose=False, multi_start=True, target_tol=TARGET_TOL,
):
    """Solve the decoupled Phase A (min-time to x_sky) + Phase B (landing) TPBVPs."""
    Q, R = mission_cost_matrices(manifold)
    x_sky = mission_x_sky(params, vz_sky)
    n_starts = 12 if multi_start else 8

    p_a0, t1, phase_a_norm = solve_phase_a(x0, x_sky, params, n_starts=n_starts)
    t_a, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=240)
    x1 = x_a[-1]
    u_a = _control_hist_phase_a(t_a, p_a, params)
    H_a = np.array([hamiltonian_a(t_a[k], x_a[k], p_a[k], u_a[k], dyn.get_hover_dynamics, params)
                    for k in range(len(t_a))])

    # Phase B starts exactly at the achieved handoff state x1 (= x_sky to tol).
    t_b_dur, x_b, p_b, tf_dur, phase_b_norm = solve_phase_b(x1, manifold, params, Q, R)
    t_b = t1 + t_b_dur
    u_b = _control_hist_phase_b(t_b, x_b, p_b, Q, R, params)
    H_b = np.array([hamiltonian_b(t_b[k], x_b[k], p_b[k], u_b[k], dyn.get_hover_dynamics, Q, R, params)
                    for k in range(len(t_b))])
    tf = float(t_b[-1])

    shoot_residual = np.concatenate([
        _phase_a_residual(np.r_[p_a0, t1], x0, x_sky, params),
        _manifold_residual(x_b[-1], p_b[-1], manifold, params),
        [H_b[-1]],
    ])
    shoot_norm = max(phase_a_norm, phase_b_norm)
    Z = np.concatenate([p_a0, p_b[0], [t1, tf]])

    sol = MissionSolution(
        success=shoot_norm <= target_tol,
        message=f"phaseA |res|={phase_a_norm:.2e}, phaseB bc|res|={phase_b_norm:.2e}",
        manifold=manifold, Z=Z, shoot_residual=shoot_residual, shoot_norm=shoot_norm,
        t1=t1, tf=tf, t_a=t_a, u_a=u_a, x_a=x_a, p_a=p_a, H_a=H_a,
        t_b=t_b, x_b=x_b, u_b=u_b, p_b=p_b, H_b=H_b, x1=x1, H_tf=float(H_b[-1]),
        x_sky=x_sky, vz_sky=vz_sky, phase_a_norm=phase_a_norm, phase_b_norm=phase_b_norm,
    )
    if verbose:
        print(f"Mission ({manifold}, vz_sky={vz_sky:+.1f}): success={sol.success}, "
              f"|res|_inf={sol.shoot_norm:.3e}, t1={t1:.3f}, tf={tf:.3f}, H(tf)={sol.H_tf:.2e}")
    return sol


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def mission_trajectory_plane(sol: MissionSolution):
    return sol.x_a[:, 0], sol.x_a[:, 1], sol.x_b[:, 0], sol.x_b[:, 1], sol.t1


def phase_a_switching_count(sol: MissionSolution, params: dict):
    """Count bang-bang thrust switches in Phase A (sign changes of S_dT)."""
    _, B = dyn.get_hover_dynamics(0.0, params)
    S = np.array([(B.T @ sol.p_a[k])[0] for k in range(len(sol.t_a))])
    return int(np.sum(np.sign(S[1:]) != np.sign(S[:-1])))


def phase_a_switch_time(sol: MissionSolution, params: dict):
    """Time of the first thrust switch (np.nan if none)."""
    _, B = dyn.get_hover_dynamics(0.0, params)
    S = np.array([(B.T @ sol.p_a[k])[0] for k in range(len(sol.t_a))])
    idx = np.where(np.sign(S[1:]) != np.sign(S[:-1]))[0]
    return float(sol.t_a[idx[0] + 1]) if idx.size else float("nan")


def control_energy(sol: MissionSolution, Q, R):
    """Integrated Phase-B control energy J_u = int u' R u dt."""
    t, u = sol.t_b, sol.u_b
    stage = np.array([u[k] @ R @ u[k] for k in range(len(t))])
    return float(np.trapezoid(stage, t))


def phase_a_fuel(sol: MissionSolution, params: dict):
    """Phase-A propellant burned (kg), absolute: int alpha*T dt with T=trim+dT."""
    trim_T = params["m0"] * params["g"]
    T_abs = sol.u_a[:, 0] + trim_T
    return float(np.trapezoid(params["α"] * T_abs, sol.t_a))


def mission_altitude_history(sol: MissionSolution, params: dict, trim_func=None):
    return np.concatenate([sol.x_a[:, 1], sol.x_b[:, 1]])


def mission_metrics(sol: MissionSolution, params: dict, Q, R):
    term = _manifold_residual(sol.x_b[-1], sol.p_b[-1], sol.manifold, params)
    return {
        "manifold": sol.manifold,
        "t1": sol.t1,
        "tf": sol.tf,
        "landing_time": sol.tf,
        "phase_b_duration": sol.tf - sol.t1,
        "shoot_norm": sol.shoot_norm,
        "H_tf": sol.H_tf,
        "terminal_manifold_inf": float(np.max(np.abs(np.r_[term, sol.H_tf]))),
        "control_energy": control_energy(sol, Q, R),
        "phase_a_switches": phase_a_switching_count(sol, params),
        "phase_a_switch_time": phase_a_switch_time(sol, params),
        "phase_a_fuel": phase_a_fuel(sol, params),
        "landing_point": (float(sol.x_b[-1, 0]), float(sol.x_b[-1, 1])),
        "min_altitude": float(np.min(mission_altitude_history(sol, params))),
    }


def terminal_sensitivity(x0, params, manifold, Q, R, sol: MissionSolution, *, delta=0.05):
    """Worst terminal-manifold residual growth under +/- delta perturbations of x_sky.

    The optimal costate (p_b0) and t_f are held fixed while the Phase-B initial
    state is perturbed; the resulting drift of the terminal manifold residual
    measures sensitivity of the landing to handoff error."""
    p_b0 = sol.p_b[0]
    tf_dur = sol.tf - sol.t1
    A, B = dyn.get_hover_dynamics(0.0, params)
    lo, hi = cst.control_bounds(0.0, params, dyn.hover_trim)
    Rinv = np.linalg.inv(R)

    def rollout(x_start):
        def rhs(t, z):
            x, p = z[:N_STATE], z[N_STATE:]
            u = np.clip(-0.5 * Rinv @ (B.T @ p), lo, hi)
            return np.concatenate([A @ x + B @ u, -2.0 * (Q @ x) - A.T @ p])
        s = solve_ivp(rhs, [0.0, tf_dur], np.r_[x_start, p_b0], method="RK45", rtol=1e-9, atol=1e-11)
        return s.y[:N_STATE, -1], s.y[N_STATE:, -1]

    base = float(np.max(np.abs(np.r_[_manifold_residual(sol.x_b[-1], sol.p_b[-1], manifold, params), sol.H_tf])))
    worst = base
    for i in range(N_STATE):
        for sign in (-1.0, 1.0):
            xf, pf = rollout(sol.x1 + sign * delta * np.eye(N_STATE)[i])
            r = float(np.max(np.abs(_manifold_residual(xf, pf, manifold, params))))
            worst = max(worst, r)
    return {"base_residual": base, "worst_perturbed": worst, "delta": delta}


def phase_a_terminal_velocity_sweep(x0, params, *, vz_values=VZ_SKY_SWEEP):
    """Solve Phase A for several terminal vertical velocities at x_sky (Sec. 7.1)."""
    rows = []
    for vz in vz_values:
        x_sky = mission_x_sky(params, vz)
        p_a0, t1, norm = solve_phase_a(x0, x_sky, params, n_starts=12)
        t_a, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=240)
        u_a = _control_hist_phase_a(t_a, p_a, params)
        _, B = dyn.get_hover_dynamics(0.0, params)
        S = np.array([(B.T @ p_a[k])[0] for k in range(len(t_a))])
        switches = int(np.sum(np.sign(S[1:]) != np.sign(S[:-1])))
        sw_idx = np.where(np.sign(S[1:]) != np.sign(S[:-1]))[0]
        t_sw = float(t_a[sw_idx[0] + 1]) if sw_idx.size else float("nan")
        trim_T = params["m0"] * params["g"]
        fuel = float(np.trapezoid(params["α"] * (u_a[:, 0] + trim_T), t_a))
        rows.append({
            "vz_sky": vz, "t1": t1, "res_norm": norm, "switches": switches,
            "t_switch": t_sw, "switch_frac": t_sw / t1 if t1 else np.nan,
            "fuel": fuel, "t_a": t_a, "x_a": x_a, "u_a": u_a, "S": S,
        })
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _attitude_quiver(ax, x_hist, color, *, n=14, length=0.45):
    step = max(1, len(x_hist) // n)
    ax.quiver(
        x_hist[::step, 0], x_hist[::step, 1],
        length * np.sin(x_hist[::step, 4]), length * np.cos(x_hist[::step, 4]),
        angles="xy", scale_units="xy", scale=1, color=color, alpha=0.5, width=0.004,
    )


def _plot_mission_phases(ax, sol, params, *, label_prefix="", quiver=True, mark_labels=True):
    import matplotlib.pyplot as plt

    px_a, pz_a, px_b, pz_b, _ = mission_trajectory_plane(sol)
    is_m2 = label_prefix.startswith("M2")
    c_a, c_b = ("C4", "C5") if is_m2 else ("C0", "C1")
    ax.plot(px_a, pz_a, color=c_a, ls="-", lw=2, label=f"{label_prefix}Phase A (ascent)")
    ax.plot(px_b, pz_b, color=c_b, ls="--" if is_m2 else "-", lw=2,
            label=f"{label_prefix}Phase B (landing)")
    if quiver:
        _attitude_quiver(ax, sol.x_a, "C0")
        _attitude_quiver(ax, sol.x_b, "C1")
    lab = (lambda s: s) if mark_labels else (lambda s: None)
    ax.scatter([px_a[0]], [pz_a[0]], c="C2", s=45, zorder=5, label=lab("launch"))
    ax.scatter([sol.x_sky[0]], [sol.x_sky[1]], marker="*", c="k", s=110, zorder=6, label=lab(r"$x_{sky}$"))
    ax.scatter([px_b[-1]], [pz_b[-1]], c="C3", s=55, zorder=6, label=lab("touchdown"))
    if sol.manifold == "M2":
        p_c, r = platform_geometry(params)
        ax.add_patch(plt.Circle((p_c, 0), r, fill=False, color="C3", ls="--", alpha=0.6))


def export_part3_figures(p3_m1, p3_m2, params, out_dir, save_figure):
    """Export all Part III figures (trajectories, terminal-velocity sweep,
    switching, Hamiltonian consistency, manifold comparison)."""
    import matplotlib.pyplot as plt

    sol_m1, sol_m2 = p3_m1["solution"], p3_m2["solution"]
    Q, R = p3_m1["Q"], p3_m1["R"]
    x0 = p3_m1["x0"]
    m1 = mission_metrics(sol_m1, params, Q, R)
    m2 = mission_metrics(sol_m2, params, Q, R)
    sens_m1 = terminal_sensitivity(x0, params, "M1", Q, R, sol_m1)
    sens_m2 = terminal_sensitivity(p3_m2["x0"], params, "M2", Q, R, sol_m2)

    # (1) Mission-plane trajectories with attitude quivers (Sec. 9 requirement).
    for tag, sol in [("M1", sol_m1), ("M2", sol_m2)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_mission_phases(ax, sol, params)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]",
               title=f"Part III {tag}: ascent + landing (attitude shown)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        save_figure(fig, f"{out_dir}/p3_trajectory_{tag}.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_mission_phases(ax, sol_m1, params, label_prefix="M1 ", quiver=False, mark_labels=True)
    _plot_mission_phases(ax, sol_m2, params, label_prefix="M2 ", quiver=False, mark_labels=False)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set(xlabel=r"$p_x$ [m]", ylabel=r"$p_z$ [m]", title="M1 vs M2 mission planes (common footprint)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, f"{out_dir}/p3_trajectory_M1_M2_overlay.png")

    # (2) Phase A terminal-velocity comparison (Sec. 7.1: vz_sky = 0 vs +/-).
    sweep = phase_a_terminal_velocity_sweep(x0, params)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sweep)))
    for row, c in zip(sweep, colors):
        lab = rf"$v_{{z,sky}}={row['vz_sky']:+.0f}$"
        axes[0, 0].plot(row["t_a"], row["x_a"][:, 1], color=c, label=lab)
        axes[0, 1].plot(row["t_a"], row["x_a"][:, 3], color=c)
        axes[1, 0].plot(row["t_a"], row["u_a"][:, 0], color=c)
        axes[1, 1].plot(row["t_a"], row["S"], color=c)
    axes[0, 0].set(title=r"altitude $p_z(t)$", xlabel="t [s]", ylabel="m")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title=r"vertical velocity $v_z(t)$", xlabel="t [s]", ylabel="m/s")
    axes[1, 0].set(title=r"thrust deviation $\delta T(t)$ (bang-bang)", xlabel="t [s]", ylabel="N")
    axes[1, 1].axhline(0, color="k", lw=0.6)
    axes[1, 1].set(title=r"switching function $S_{\delta T}=(B^\top p)_1$", xlabel="t [s]")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    save_figure(fig, f"{out_dir}/p3_phaseA_terminal_velocity.png")

    # (3) Switching structure + full control history.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, (sol, name) in enumerate([(sol_m1, "M1"), (sol_m2, "M2")]):
        _, B = dyn.get_hover_dynamics(0.0, params)
        S_a = np.array([(B.T @ sol.p_a[k])[0] for k in range(len(sol.t_a))])
        axes[0, col].plot(sol.t_a, S_a, "C0")
        axes[0, col].axhline(0, color="k", lw=0.6)
        axes[0, col].set(title=f"{name}: Phase A switching fn $S_{{\\delta T}}$", ylabel=r"$B^\top p$")
        t_full = np.concatenate([sol.t_a, sol.t_b])
        u_full = np.vstack([sol.u_a, sol.u_b])
        axes[1, col].plot(t_full, u_full[:, 0], label=r"$\delta T$ [N]")
        axes[1, col].plot(t_full, u_full[:, 1], label=r"$\tau$ [N$\cdot$m]", alpha=0.8)
        axes[1, col].axvline(sol.t1, color="gray", ls=":", label="handoff $t_1$")
        axes[1, col].set(xlabel="t [s]", title=f"{name}: controls", ylabel="N / N·m")
        axes[1, col].legend(fontsize=7)
    save_figure(fig, f"{out_dir}/p3_switching_controls.png")

    # (4) Hamiltonian consistency (H_A == 0 min-time; H_B == 0 free-final-time).
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for sol, c, name in [(sol_m1, "C0", "M1 flat"), (sol_m2, "C1", "M2 platform")]:
        axes[0].plot(sol.t_a, sol.H_a, color=c, lw=1.8, label=name)
        axes[1].plot(sol.t_b, sol.H_b, color=c, lw=1.8, label=name)
        axes[1].scatter([sol.t_b[-1]], [sol.H_b[-1]], color=c, s=36, zorder=5)
    axes[0].set(title=r"Phase A — minimum time ($H_A\equiv 0$)", ylabel=r"$H_A$")
    axes[1].set(title=r"Phase B — free final time ($H_B\equiv 0$, $H_B(t_f)=0$)",
                ylabel=r"$H_B$", xlabel="t [s]")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    save_figure(fig, f"{out_dir}/p3_hamiltonian.png")

    # (5) Manifold comparison bars.
    labels = ["M1 flat", "M2 platform"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].bar(labels, [m1["landing_time"], m2["landing_time"]], color=["C0", "C1"])
    axes[0, 0].set(title="Total landing time $t_f$ [s]")
    axes[0, 1].bar(labels, [m1["phase_b_duration"], m2["phase_b_duration"]], color=["C0", "C1"])
    axes[0, 1].set(title="Phase B duration [s]")
    axes[0, 2].bar(labels, [m1["control_energy"], m2["control_energy"]], color=["C0", "C1"])
    axes[0, 2].set(title=r"Phase B energy $\int u^\top R u\,dt$")
    axes[1, 0].bar(labels, [m1["phase_a_switches"], m2["phase_a_switches"]], color=["C0", "C1"])
    axes[1, 0].set(title="Phase A thrust switches")
    axes[1, 1].bar(labels, [max(m1["shoot_norm"], 1e-16), max(m2["shoot_norm"], 1e-16)], color=["C0", "C1"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(TARGET_TOL, color="k", ls="--", lw=0.8, label="target $10^{-5}$")
    axes[1, 1].set(title=r"Shooting $\|res\|_\infty$ (feasibility)")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].bar(labels, [sens_m1["worst_perturbed"], sens_m2["worst_perturbed"]], color=["C0", "C1"])
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(title=r"Terminal sensitivity ($\pm\delta x_0$)")
    save_figure(fig, f"{out_dir}/p3_manifold_comparison.png")

    return {"M1": m1, "M2": m2, "sens_M1": sens_m1, "sens_M2": sens_m2,
            "sweep": [{k: v for k, v in r.items() if k not in ("t_a", "x_a", "u_a", "S")} for r in sweep]}
