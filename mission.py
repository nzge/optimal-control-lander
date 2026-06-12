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
import lqr

Manifold = Literal["M1", "M2"]
N_STATE = 7
N_SHOOT = 2 * N_STATE + 2
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


def mission_cost_matrices(manifold: Manifold = "M1"):
    """Phase B running cost (no 1/2 in integral → factor 2 in adjoint)."""
    Q = np.diag([0.5, 8.0, 0.5, 3.0, 8.0, 4.0, 0.01])
    R = np.diag([0.05, 1.0])
    if manifold == "M2":
        Q[0, 0] = 4.0
        Q[1, 1] = 15.0
        Q[3, 3] = 5.0
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


def _phase_b_error(x, manifold, params, *, phi=0.0):
    return x - _phase_b_target(manifold, params, phi=phi)


def hamiltonian_b_error(x, p, u, Q, R, params, manifold, *, phi=0.0):
    """Phase-B Hamiltonian in error coordinates xi = x - x_ref (H(tf)=0 when xi->0)."""
    xi = _phase_b_error(x, manifold, params, phi=phi)
    A, B = dyn.get_hover_dynamics(0.0, params)
    return xi @ Q @ xi + u @ R @ u + p @ (A @ xi + B @ u)


def hamiltonian_b(t, x, p, u, dynamics_func, Q, R, params, *, manifold="M1", phi=0.0):
    return hamiltonian_b_error(x, p, u, Q, R, params, manifold, phi=phi)


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


def _mission_Qf():
    return np.diag([10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1])


_RICCATI_CACHE: dict[float, object] = {}


def _phase_b_riccati(tf, Q, R, params):
    """Riccati gain for Phase B on [0, tf] (cached on tf)."""
    key = round(float(tf), 2)
    if key not in _RICCATI_CACHE:
        _RICCATI_CACHE[key] = lqr.solve_riccati_backward(
            dyn.get_hover_dynamics,
            np.linspace(0.0, max(key, 1.0), max(80, int(4 * key))),
            Q,
            R,
            _mission_Qf(),
            params,
        )[0]
    return _RICCATI_CACHE[key]


def clear_riccati_cache():
    _RICCATI_CACHE.clear()


def _m2_landing_target(phi: float, params: dict) -> np.ndarray:
    """Point on the platform circle in hover-dev / absolute coords."""
    r, pc = params["r_platform"], params["p_c"]
    return np.array([pc + r * np.cos(phi), r * np.sin(phi), 0.0, 0.0, 0.0, 0.0, 0.0])


def _phase_b_target(manifold, params, *, phi: float = 0.0):
    if manifold == "M2":
        return _m2_landing_target(phi, params)
    return np.zeros(N_STATE)


def phase_b_feedback_control(t, x, P_interp, R, params, trim_func, *, x_target=None, gain=0.5):
    """LQR feedback: u = -gain * R^{-1} B' P(t) (x - x_ref); gain=0.5 matches root.tex PMP."""
    x_target = np.zeros(N_STATE) if x_target is None else x_target
    _, B = dyn.get_hover_dynamics(t, params)
    u_req = -gain * np.linalg.inv(R) @ B.T @ P_interp(t) @ (x - x_target)
    lo, hi = cst.control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def propagate_phase_b_feedback(
    x1, t1, tf, Q, R, params, trim_func, *, n_eval=200, manifold: Manifold = "M1", phi: float = 0.0,
):
    """Phase-B LQR feedback rollout with p(t) = P(t)(x - x_ref); RK4 via closed-loop sim."""
    if tf <= t1 + 0.05:
        raise RuntimeError("tf must exceed t1")
    x_target = _phase_b_target(manifold, params, phi=phi)
    gain = 1.0 if manifold == "M2" else 0.5
    P_interp = _phase_b_riccati(tf, Q, R, params)
    t_eval = np.linspace(t1, tf, max(n_eval, 2))
    A, B = dyn.get_hover_dynamics(t1, params)

    def ctrl(t, x):
        return phase_b_feedback_control(
            t, x, P_interp, R, params, trim_func, x_target=x_target, gain=gain,
        )

    x_hist, u_hist, _ = lqr.simulate_lti_closed_loop(A, B, t_eval, ctrl, x1)
    p_hist = np.array([P_interp(t_eval[k]) @ (x_hist[k] - x_target) for k in range(len(t_eval))])
    return t_eval, x_hist, p_hist


def propagate_phase_b(x1, p1, t1, tf, dynamics_func, Q, R, params, trim_func, *, n_eval=200, manifold="M1", phi=0.0):
    return propagate_phase_b_feedback(
        x1, t1, tf, Q, R, params, trim_func, n_eval=n_eval, manifold=manifold, phi=phi,
    )


def _control_hist_phase_b(t, x_hist, p_hist, dynamics_func, Q, R, params, trim_func, *, manifold="M1", phi=0.0):
    P_interp = _phase_b_riccati(float(t[-1]), Q, R, params)
    x_target = _phase_b_target(manifold, params, phi=phi)
    gain = 1.0 if manifold == "M2" else 0.5
    return np.array([
        phase_b_feedback_control(
            t[k], x_hist[k], P_interp, R, params, trim_func, x_target=x_target, gain=gain,
        )
        for k in range(len(t))
    ])


def _split_Z(Z, manifold: Manifold = "M1"):
    Z = np.asarray(Z, dtype=float)
    p_a0 = Z[:N_STATE]
    t1 = float(Z[N_STATE])
    tf = float(Z[N_STATE + 1])
    phi = float(Z[N_STATE + 2]) if manifold == "M2" and Z.size >= N_STATE + 3 else 0.0
    return p_a0, t1, tf, phi


def _split_Z9(Z):
    p_a0, t1, tf, _ = _split_Z(Z, "M1")
    return p_a0, t1, tf


def _pack_Z(p_a0, t1, tf, *, manifold: Manifold = "M1", phi: float = 0.0):
    if manifold == "M2":
        return np.r_[p_a0, t1, tf, phi]
    return np.r_[p_a0, t1, tf]


def _expand_Z(Z, x1, Q, R, params, *, manifold: Manifold = "M1"):
    """Embed shooting vector into legacy 16-D storage."""
    p_a0, t1, tf, phi = _split_Z(Z, manifold)
    P_interp = _phase_b_riccati(tf, Q, R, params)
    x_target = _phase_b_target(manifold, params, phi=phi)
    Z16 = np.zeros(2 * N_STATE + 2)
    Z16[:N_STATE] = p_a0
    Z16[N_STATE : 2 * N_STATE] = P_interp(t1) @ (x1 - x_target)
    Z16[2 * N_STATE] = t1
    Z16[2 * N_STATE + 1] = tf
    return Z16


def _as_Z(Z, manifold: Manifold = "M1"):
    Z = np.asarray(Z, dtype=float)
    if Z.size == 2 * N_STATE + 2:
        return _pack_Z(Z[:N_STATE], Z[2 * N_STATE], Z[2 * N_STATE + 1], manifold=manifold)
    return Z


def _manifold_eval(x_f, p_f, tf, params, manifold, trim_func, *, use_absolute=False):
    x_eval = cst.absolute_state(x_f, tf, params, trim_func) if use_absolute else x_f
    if manifold == "M1":
        return np.array([x_eval[1], x_eval[3], x_eval[4], p_f[0], p_f[2], p_f[5], p_f[6]])
    h1 = (x_eval[0] - params["p_c"]) ** 2 + x_eval[1] ** 2 - params["r_platform"] ** 2
    collin = p_f[0] * x_eval[1] - p_f[1] * (x_eval[0] - params["p_c"])
    return np.array([h1, x_eval[3], x_eval[4], p_f[2], p_f[5], p_f[6], collin])


def _manifold_opts(manifold: Manifold):
    """Shooting options for terminal manifold evaluation."""
    return {"use_absolute_manifold": manifold == "M2"}


def _shooting_weights(manifold: Manifold) -> np.ndarray:
    """
    Weight terminal residuals for Riccati-based Phase B.

    State manifold + H(tf) and Phase A handoff are fully weighted; terminal
    costate components that Riccati feedback cannot independently satisfy are
    down-weighted so the joint solver prioritizes platform geometry.
    """
    w = np.ones(N_SHOOT)
    if manifold == "M1":
        w[11:15] = 0.12
    else:
        w[11:14] = 0.12
    return w


def _primary_shoot_norm(residuals: np.ndarray) -> float:
    """Feasibility metric for Riccati Phase B: Phase A + terminal state + H(tf)."""
    state_terminal = residuals[8:11]
    return float(max(np.linalg.norm(residuals[:8], ord=np.inf), np.linalg.norm(state_terminal, ord=np.inf), abs(residuals[15])))


def _weighted_shooting_residuals(Z, x0, params, *, manifold, Q, R, weights, n_eval, m_opts):
    return shooting_residuals(
        Z, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval, **m_opts,
    ) * weights


def _control_hist_phase_a(t, p_hist, dynamics_func, params, trim_func):
    return np.array([phase_a_control(t[k], p_hist[k], dynamics_func, params, trim_func) for k in range(len(t))])


def shooting_residuals(
    Z, x0, params, *, manifold: Manifold = "M1", Q=None, R=None,
    dynamics_a=None, dynamics_b=None, trim_a=None, trim_b=None,
    use_absolute_manifold: bool = False, n_eval: int = 120,
):
    if Q is None or R is None:
        Q, R = mission_cost_matrices(manifold)
    dynamics_a = dynamics_a or dyn.get_hover_dynamics
    trim_a = trim_a or dyn.hover_trim
    trim_b = trim_b or dyn.hover_trim

    Z = _as_Z(Z, manifold)
    p_a0, t1, tf, phi = _split_Z(Z, manifold)

    if t1 <= 0.05 or tf <= t1 + 0.05:
        return np.full(2 * N_STATE, 1e2)

    try:
        _, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dynamics_a, params, trim_a, n_eval=n_eval)
        x1 = x_a[-1]
        p_a1 = p_a[-1]
        _, x_b, p_b = propagate_phase_b_feedback(
            x1, t1, tf, Q, R, params, trim_b, n_eval=n_eval, manifold=manifold, phi=phi,
        )
    except RuntimeError:
        return np.full(2 * N_STATE, 1e2)

    x_f, p_f = x_b[-1], p_b[-1]
    u_a1 = phase_a_control(t1, p_a1, dynamics_a, params, trim_a)
    A1, B1 = dynamics_a(t1, params)
    H_a1 = 1.0 + p_a1 @ (A1 @ x1 + B1 @ u_a1)
    P_tf = _phase_b_riccati(tf, Q, R, params)
    gain = 1.0 if manifold == "M2" else 0.5
    x_target = _phase_b_target(manifold, params, phi=phi)
    u_f = phase_b_feedback_control(
        tf, x_f, P_tf, R, params, trim_b, x_target=x_target, gain=gain,
    )
    H_f = hamiltonian_b(tf, x_f, p_f, u_f, dyn.get_hover_dynamics, Q, R, params, manifold=manifold, phi=phi)

    residuals = list(p_a1) + [H_a1]
    residuals.extend(_manifold_eval(x_f, p_f, tf, params, manifold, trim_b, use_absolute=use_absolute_manifold))
    residuals.append(H_f)
    return np.asarray(residuals, dtype=float)


def _solve_phase_a(x0, params, *, t1_bounds=(0.8, 3.5), n_starts=6, n_eval=250):
    """Minimum-time Phase A TPBVP with multi-start least squares."""

    def fun(z):
        p0, t1 = z[:7], float(z[7])
        if t1 < t1_bounds[0] or t1 > t1_bounds[1]:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_a(
                x0, p0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=n_eval,
            )
        except RuntimeError:
            return np.ones(8) * 1e2
        p1 = p[-1]
        u1 = phase_a_control(t1, p1, dyn.get_hover_dynamics, params, dyn.hover_trim)
        A, B = dyn.get_hover_dynamics(t1, params)
        return np.concatenate([p1, [1.0 + p1 @ (A @ x[-1] + B @ u1)]])

    lb = np.r_[np.full(7, -20), t1_bounds[0]]
    ub = np.r_[np.full(7, 20), t1_bounds[1]]
    rng = np.random.default_rng(0)
    seed_guesses = [np.r_[np.array([0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0]), 1.4]]
    for _ in range(max(0, n_starts - 1)):
        guess = np.zeros(8)
        guess[:7] = rng.uniform(-8.0, 8.0, 7)
        guess[7] = rng.uniform(t1_bounds[0], t1_bounds[1])
        seed_guesses.append(guess)

    best_res = None
    best_norm = np.inf
    for z0 in seed_guesses:
        res = least_squares(
            fun,
            np.clip(z0, lb, ub),
            bounds=(lb, ub),
            max_nfev=600,
            ftol=1e-14,
            xtol=1e-14,
        )
        norm = float(np.linalg.norm(res.fun, ord=np.inf))
        if norm < best_norm:
            best_norm, best_res = norm, res
    return best_res.x[:7], float(best_res.x[7]), best_res


def _polish_phase_a(Z, x0, params, manifold, Q, R, *, n_eval=150, max_nfev=350):
    """Block polish on (p_A(0), t1) targeting p_A(t1)=0 and H_A(t1)=0."""
    p_a0, t1, tf, phi = _split_Z(_as_Z(Z, manifold), manifold)

    def fun(z):
        p0, t1_trial = z[:7], float(z[7])
        try:
            _, x, p = propagate_phase_a(
                x0, p0, t1_trial, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=n_eval,
            )
        except RuntimeError:
            return np.ones(8) * 1e2
        p1 = p[-1]
        u1 = phase_a_control(t1_trial, p1, dyn.get_hover_dynamics, params, dyn.hover_trim)
        A, B = dyn.get_hover_dynamics(t1_trial, params)
        return np.concatenate([p1, [1.0 + p1 @ (A @ x[-1] + B @ u1)]])

    z0 = np.r_[p_a0, t1]
    lb = np.r_[np.full(N_STATE, -25), 0.9]
    ub = np.r_[np.full(N_STATE, 25), 2.8]
    res = least_squares(
        fun, np.clip(z0, lb, ub), bounds=(lb, ub), max_nfev=max_nfev, ftol=1e-14, xtol=1e-14,
    )
    return _pack_Z(res.x[:7], float(res.x[7]), tf, manifold=manifold, phi=phi), res


def _polish_phase_b_timing(Z, x0, params, manifold, Q, R, *, n_eval=120, max_nfev=200):
    """Block polish on (tf [, phi]) targeting terminal manifold + H(tf)=0."""
    m_opts = _manifold_opts(manifold)
    p_a0, t1, tf, phi = _split_Z(_as_Z(Z, manifold), manifold)

    def fun(z):
        if manifold == "M2":
            Zt = _pack_Z(p_a0, t1, float(z[0]), manifold=manifold, phi=float(z[1]))
        else:
            Zt = _pack_Z(p_a0, t1, float(z[0]), manifold=manifold)
        return shooting_residuals(
            Zt, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval, **m_opts,
        )[8:16]

    tf_lo = max(12.0, t1 + 8.0)
    tf_hi = min(50.0 if manifold == "M2" else 45.0, params.get("tf_mission_max", 60.0))
    if manifold == "M2":
        z0 = np.array([tf, phi])
        lb, ub = np.array([tf_lo, -0.4]), np.array([tf_hi, 0.4])
    else:
        z0 = np.array([tf])
        lb, ub = np.array([tf_lo]), np.array([tf_hi])
    res = least_squares(
        fun, np.clip(z0, lb, ub), bounds=(lb, ub), max_nfev=max_nfev, ftol=1e-14, xtol=1e-14,
    )
    if manifold == "M2":
        return _pack_Z(p_a0, t1, float(res.x[0]), manifold=manifold, phi=float(res.x[1])), res
    return _pack_Z(p_a0, t1, float(res.x[0]), manifold=manifold), res


def _alternating_polish(Z, x0, params, manifold, Q, R, *, rounds=2):
    """Alternate Phase-A / Phase-B / full polish for tighter handoff."""
    Z = _as_Z(Z, manifold)
    best_Z, best_norm = Z, np.inf
    m_opts = _manifold_opts(manifold)
    for _ in range(rounds):
        Z, _ = _polish_phase_a(Z, x0, params, manifold, Q, R, n_eval=200, max_nfev=450)
        Z, _ = _polish_phase_b_timing(Z, x0, params, manifold, Q, R, n_eval=180, max_nfev=300)
        polish = _polish(Z, x0, params, manifold, Q, R, n_eval=200, max_nfev=600, weighted=True)
        Z = polish.x
        norm = _primary_shoot_norm(polish.fun)
        if norm < best_norm:
            best_norm, best_Z = norm, Z.copy()
        if norm <= TARGET_TOL:
            break
    return best_Z, best_norm


def _solve_phase_b_tf(x1, t1, params, Q, R, *, manifold: Manifold = "M1", tf_bounds=(18.0, 45.0), phi=0.0):
    """Search (tf [, phi]) for terminal manifold + H(tf)=0."""
    m_opts = _manifold_opts(manifold)
    tf_lo = max(tf_bounds[0], t1 + 12.0)

    def eval_tf_phi(z):
        tf = float(z[0])
        ph = float(z[1]) if z.size > 1 else phi
        if tf <= t1 + 0.5:
            return np.ones(8) * 1e2
        try:
            _, x, p = propagate_phase_b_feedback(
                x1, t1, tf, Q, R, params, dyn.hover_trim, n_eval=80, manifold=manifold, phi=ph,
            )
        except RuntimeError:
            return np.ones(8) * 1e2
        xf, pf = x[-1], p[-1]
        P_tf = _phase_b_riccati(tf, Q, R, params)
        gain = 1.0 if manifold == "M2" else 0.5
        xt = _phase_b_target(manifold, params, phi=ph)
        u = phase_b_feedback_control(tf, xf, P_tf, R, params, dyn.hover_trim, x_target=xt, gain=gain)
        H = hamiltonian_b(tf, xf, pf, u, dyn.get_hover_dynamics, Q, R, params, manifold=manifold, phi=ph)
        term = _manifold_eval(
            xf, pf, tf, params, manifold, dyn.hover_trim, use_absolute=m_opts["use_absolute_manifold"]
        )
        return np.concatenate([term, [H]])

    if manifold == "M2":
        z0 = np.array([max(t1 + 22.0, tf_lo), 0.0])
        lb = np.array([tf_lo, -0.35])
        ub = np.array([tf_bounds[1], 0.35])
        res = least_squares(eval_tf_phi, z0, bounds=(lb, ub), max_nfev=120, ftol=1e-12, xtol=1e-12)
        return float(res.x[0]), float(res.x[1]), res

    z0 = np.array([max(t1 + 20.0, tf_lo)])
    res = least_squares(
        eval_tf_phi, z0, bounds=([tf_lo], [tf_bounds[1]]), max_nfev=80, ftol=1e-12, xtol=1e-12
    )
    return float(res.x[0]), 0.0, res


def _sequential_seed(x0, params, manifold, Q, R, *, t1_seed=1.3, phi_seed=0.0):
    """Sequential Phase A → Phase B (tf [, phi]) seed."""
    t1_lo = max(0.8, t1_seed - 0.35)
    t1_hi = min(4.0, t1_seed + 0.35)
    p_a0, t1, _ = _solve_phase_a(x0, params, t1_bounds=(t1_lo, t1_hi), n_starts=5)
    _, x_a, _ = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=120)
    tf, phi, _ = _solve_phase_b_tf(
        x_a[-1], t1, params, Q, R, manifold=manifold, phi=phi_seed if manifold == "M2" else 0.0,
    )
    return _pack_Z(p_a0, t1, tf, manifold=manifold, phi=phi)


def _polish(Z, x0, params, manifold, Q, R, *, n_eval=100, max_nfev=350, weighted=False):
    m_opts = _manifold_opts(manifold)
    weights = _shooting_weights(manifold) if weighted else np.ones(N_SHOOT)

    def fun(z):
        return _weighted_shooting_residuals(
            z, x0, params, manifold=manifold, Q=Q, R=R, weights=weights, n_eval=n_eval, m_opts=m_opts,
        )

    Z = _as_Z(Z, manifold)
    _, t1, _, _ = _split_Z(Z, manifold)
    lb = np.r_[np.full(N_STATE, -25), [0.9, max(12.0, t1 + 8.0)]]
    ub = np.r_[np.full(N_STATE, 25), [2.8, min(45.0, params.get("tf_mission_max", 60.0))]]
    if manifold == "M2":
        ub = np.r_[np.full(N_STATE, 25), [2.8, min(50.0, params.get("tf_mission_max", 60.0))]]
    if manifold == "M2":
        lb = np.r_[lb, -0.5]
        ub = np.r_[ub, 0.5]
    z0 = np.clip(Z, lb, ub)
    res = least_squares(
        fun, z0, bounds=(lb, ub), max_nfev=max_nfev, ftol=1e-14, xtol=1e-14, method="trf"
    )
    res.fun = shooting_residuals(
        res.x, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval, **m_opts
    )
    return res


def _build_solution(Z, x0, params, manifold, Q, R, message="", *, target_tol=TARGET_TOL, n_eval=200):
    m_opts = _manifold_opts(manifold)
    Z = _as_Z(Z, manifold)
    shoot_res = shooting_residuals(
        Z, x0, params, manifold=manifold, Q=Q, R=R, n_eval=n_eval, **m_opts
    )
    shoot_norm = float(np.linalg.norm(shoot_res, ord=np.inf))
    p_a0, t1, tf, phi = _split_Z(Z, manifold)
    t_a, x_a, p_a = propagate_phase_a(x0, p_a0, t1, dyn.get_hover_dynamics, params, dyn.hover_trim, n_eval=n_eval)
    x1 = x_a[-1]
    t_b, x_b, p_b = propagate_phase_b_feedback(
        x1, t1, tf, Q, R, params, dyn.hover_trim, n_eval=n_eval, manifold=manifold, phi=phi,
    )
    Z16 = _expand_Z(Z, x1, Q, R, params, manifold=manifold)
    u_a = _control_hist_phase_a(t_a, p_a, dyn.get_hover_dynamics, params, dyn.hover_trim)
    u_b = _control_hist_phase_b(
        t_b, x_b, p_b, dyn.get_hover_dynamics, Q, R, params, dyn.hover_trim, manifold=manifold, phi=phi,
    )
    H_a = np.array([hamiltonian_a(t_a[k], x_a[k], p_a[k], u_a[k], dyn.get_hover_dynamics, params) for k in range(len(t_a))])
    H_b = np.array([
        hamiltonian_b(t_b[k], x_b[k], p_b[k], u_b[k], dyn.get_hover_dynamics, Q, R, params, manifold=manifold, phi=phi)
        for k in range(len(t_b))
    ])
    return MissionSolution(
        success=shoot_norm <= target_tol,
        message=message,
        manifold=manifold,
        Z=Z16,
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
    clear_riccati_cache()
    Q, R = mission_cost_matrices(manifold)
    if Z0 is not None:
        sol = _build_solution(_as_Z(Z0, manifold), x0, params, manifold, Q, R, target_tol=target_tol)
        if verbose:
            print(f"Mission ({manifold}): |res|_inf={sol.shoot_norm:.3e}")
        return sol

    t1_seeds = [1.0, 1.25, 1.4, 1.55, 1.75] if multi_start else [1.3]
    phi_seeds = [-0.2, 0.0, 0.2] if manifold == "M2" and multi_start else [0.0]
    best_Z, best_primary, best_msg = None, np.inf, ""
    m_opts = _manifold_opts(manifold)
    for t1_seed in t1_seeds:
        for phi_seed in phi_seeds:
            Z = _sequential_seed(x0, params, manifold, Q, R, t1_seed=t1_seed, phi_seed=phi_seed)
            Z, primary = _alternating_polish(Z, x0, params, manifold, Q, R, rounds=2)
            if primary < best_primary:
                best_primary, best_Z, best_msg = primary, Z.copy(), "alternating polish"
            if primary <= target_tol:
                break
        if best_primary <= target_tol:
            break

    if best_Z is not None:
        polish = _polish(
            best_Z, x0, params, manifold, Q, R, n_eval=250, max_nfev=700, weighted=True,
        )
        primary = _primary_shoot_norm(polish.fun)
        if primary < best_primary:
            best_primary, best_Z, best_msg = primary, polish.x.copy(), "joint weighted polish"

    sol = _build_solution(best_Z, x0, params, manifold, Q, R, best_msg, target_tol=target_tol, n_eval=280)
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
    m_opts = _manifold_opts(sol.manifold)
    term = _manifold_eval(
        sol.x_b[-1], sol.p_b[-1], sol.tf, params, sol.manifold, dyn.hover_trim,
        use_absolute=m_opts["use_absolute_manifold"],
    )
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
    """Max |shooting residual| under small IC perturbations (fixed Z)."""
    m_opts = _manifold_opts(manifold)
    base = sol.shoot_norm
    worst = base
    for i in range(7):
        for sign in (-1, 1):
            xpert = x0 + sign * delta * np.eye(7)[i]
            res = shooting_residuals(_as_Z(sol.Z, manifold), xpert, params, manifold=manifold, Q=Q, R=R, **m_opts)
            worst = max(worst, float(np.linalg.norm(res, ord=np.inf)))
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
