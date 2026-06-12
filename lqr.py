"""
Finite-horizon LQR for the lander linearization (Part I regulation, Part II tracking).

State order: [p_x, p_z, v_x, v_z, theta, omega, m]  (deviations from trim)
Control:     [delta_T, tau]
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

import constraints as cst


def _unpack_P(P_flat, n):
    return P_flat.reshape((n, n))


def _pack_P(P):
    return P.flatten()


def riccati_rhs(t, P_flat, dynamics_func, Q, R, params):
    """Backward-time Riccati: -dP/dt = A'P + PA - PBR^{-1}B'P + Q."""
    n = Q.shape[0]
    P = _unpack_P(P_flat, n)
    A, B = dynamics_func(t, params)
    Rinv = np.linalg.inv(R)
    dP_dt_backward = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
    return -_pack_P(dP_dt_backward)


def tracking_s_rhs(t, s, P_interp, xref_interp, dynamics_func, Q, R, params):
    """Backward-time feedforward ODE for tracking: -ds/dt = (A - BR^{-1}B'P)'s + Q x_ref."""
    n = Q.shape[0]
    A, B = dynamics_func(t, params)
    P = P_interp(t)
    Rinv = np.linalg.inv(R)
    xref = xref_interp(t)
    Acl = A - B @ Rinv @ B.T @ P
    ds_dt_backward = Acl.T @ s + Q @ xref
    return -ds_dt_backward


def solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf, params):
    """
    Integrate Riccati backward on t_grid (must be increasing).
    Returns P(t) interpolator and gain history K(t) = R^{-1} B' P.
    """
    n = Q.shape[0]
    t0, tf = t_grid[0], t_grid[-1]

    sol = solve_ivp(
        riccati_rhs,
        [tf, t0],
        _pack_P(Qf),
        t_eval=t_grid[::-1],
        args=(dynamics_func, Q, R, params),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati integration failed: {sol.message}")

    t_P = sol.t[::-1]
    P_hist = np.array([_unpack_P(sol.y[:, i], n) for i in range(sol.t.size)])

    def P_interp(t):
        t = np.clip(t, t0, tf)
        idx = np.searchsorted(t_P, t, side="right") - 1
        idx = np.clip(idx, 0, len(t_P) - 2)
        alpha = (t - t_P[idx]) / (t_P[idx + 1] - t_P[idx] + 1e-15)
        return (1 - alpha) * P_hist[idx] + alpha * P_hist[idx + 1]

    return P_interp, t_P, P_hist


def solve_tracking_feedforward(dynamics_func, t_grid, Q, R, xref_grid, P_interp, params):
    """
    Integrate s backward with s(tf)=0. xref_grid aligns with t_grid.
    Returns s(t) interpolator.
    """
    n = Q.shape[0]
    t0, tf = t_grid[0], t_grid[-1]

    def xref_interp(t):
        t = np.clip(t, t0, tf)
        idx = np.searchsorted(t_grid, t, side="right") - 1
        idx = np.clip(idx, 0, len(t_grid) - 2)
        alpha = (t - t_grid[idx]) / (t_grid[idx + 1] - t_grid[idx] + 1e-15)
        return (1 - alpha) * xref_grid[idx] + alpha * xref_grid[idx + 1]

    sol = solve_ivp(
        tracking_s_rhs,
        [tf, t0],
        np.zeros(n),
        t_eval=t_grid[::-1],
        args=(P_interp, xref_interp, dynamics_func, Q, R, params),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Tracking feedforward ODE failed: {sol.message}")

    t_s = sol.t[::-1]
    s_hist = sol.y.T

    def s_interp(t):
        t = np.clip(t, t0, tf)
        idx = np.searchsorted(t_s, t, side="right") - 1
        idx = np.clip(idx, 0, len(t_s) - 2)
        alpha = (t - t_s[idx]) / (t_s[idx + 1] - t_s[idx] + 1e-15)
        return (1 - alpha) * s_hist[idx] + alpha * s_hist[idx + 1]

    return s_interp, t_s, s_hist, xref_interp


def tracking_control(t, x, xref, dynamics_func, P_interp, s_interp, Q, R, params):
    """
    u*(t) = -K(t) x + K(t) x_ref - R^{-1} B' s
          = -R^{-1} B' P (x - x_ref) - R^{-1} B' s
    """
    A, B = dynamics_func(t, params)
    P = P_interp(t)
    s = s_interp(t)
    Rinv = np.linalg.inv(R)
    K = Rinv @ B.T @ P
    u = -K @ (x - xref) - Rinv @ B.T @ s
    return u, K


def regulation_control(t, x, dynamics_func, P_interp, Q, R, params):
    """u = -R^{-1} B' P x  (deviation coordinates about origin)."""
    A, B = dynamics_func(t, params)
    P = P_interp(t)
    Rinv = np.linalg.inv(R)
    return -Rinv @ B.T @ P @ x


def _eval_control(control_func, t, x, constraints=None, info=None, step=0):
    """Evaluate control law with optional saturation and bookkeeping."""
    u_req = np.asarray(control_func(t, x), dtype=float)
    if constraints is None:
        return u_req, u_req
    u_act, saturated = cst.apply_control(u_req, t, constraints)
    if info is not None:
        info.control_saturated[step] = saturated
        if info.u_req_hist is not None:
            info.u_req_hist[step] = u_req
    return u_act, u_req


def _truncate_histories(x_hist, u_hist, info, k, violation_name):
    info.state_violated = True
    info.violation_time = float(info.violation_time) if info.violation_time else None
    info.violation_name = violation_name
    info.terminated_early = True
    info.control_saturated = info.control_saturated[: k + 1]
    if info.u_req_hist is not None:
        info.u_req_hist = info.u_req_hist[: k + 1]
    return x_hist[: k + 1], u_hist[: k + 1]


def simulate_lti_closed_loop(A, B, t_grid, control_func, x0, *, constraints=None):
    """Forward RK4 closed-loop roll-out for constant (A,B).

    When ``constraints`` is set, control is saturated (PMP) and integration
    stops at the first state constraint violation (numerical enforcement).
    Returns (x_hist, u_hist, info).
    """
    n = A.shape[0]
    m = B.shape[1]
    x_hist = np.zeros((len(t_grid), n))
    u_hist = np.zeros((len(t_grid), m))
    info = cst.empty_info(len(t_grid), m) if constraints else cst.RolloutConstraintInfo()

    x_hist[0] = x0
    u_hist[0], _ = _eval_control(control_func, t_grid[0], x0, constraints, info, 0)

    if constraints and (vname := cst.check_state(x0, t_grid[0], constraints)):
        info.violation_time = t_grid[0]
        x_hist, u_hist = _truncate_histories(x_hist, u_hist, info, 0, vname)
        return x_hist, u_hist, cst.finalize_info(info)

    for k in range(len(t_grid) - 1):
        dt = t_grid[k + 1] - t_grid[k]
        t = t_grid[k]
        x = x_hist[k]
        u, _ = _eval_control(control_func, t, x, constraints, info, k)
        u_hist[k] = u
        k1 = A @ x + B @ u
        k2 = A @ (x + 0.5 * dt * k1) + B @ u
        k3 = A @ (x + 0.5 * dt * k2) + B @ u
        k4 = A @ (x + dt * k3) + B @ u
        x_hist[k + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if constraints and (vname := cst.check_state(x_hist[k + 1], t_grid[k + 1], constraints)):
            u_hist[k + 1], _ = _eval_control(
                control_func, t_grid[k + 1], x_hist[k + 1], constraints, info, k + 1
            )
            info.violation_time = t_grid[k + 1]
            x_hist, u_hist = _truncate_histories(x_hist, u_hist, info, k + 1, vname)
            return x_hist, u_hist, cst.finalize_info(info)

    u_hist[-1], _ = _eval_control(control_func, t_grid[-1], x_hist[-1], constraints, info, len(t_grid) - 1)
    return x_hist, u_hist, cst.finalize_info(info)


def simulate_ltv_closed_loop(dynamics_func, t_grid, control_func, x0, params, *, constraints=None):
    """RK4 closed-loop roll-out for LTV linearization.

    See ``simulate_lti_closed_loop`` for constraint handling.
    Returns (x_hist, u_hist, info).
    """
    n = len(x0)
    m = 2
    x_hist = np.zeros((len(t_grid), n))
    u_hist = np.zeros((len(t_grid), m))
    info = cst.empty_info(len(t_grid), m) if constraints else cst.RolloutConstraintInfo()

    x_hist[0] = x0
    u_hist[0], _ = _eval_control(control_func, t_grid[0], x0, constraints, info, 0)

    if constraints and (vname := cst.check_state(x0, t_grid[0], constraints)):
        info.violation_time = t_grid[0]
        x_hist, u_hist = _truncate_histories(x_hist, u_hist, info, 0, vname)
        return x_hist, u_hist, cst.finalize_info(info)

    for k in range(len(t_grid) - 1):
        dt = t_grid[k + 1] - t_grid[k]
        t = t_grid[k]
        x = x_hist[k]
        u, _ = _eval_control(control_func, t, x, constraints, info, k)
        u_hist[k] = u
        A, B = dynamics_func(t, params)

        def f(xv):
            return A @ xv + B @ u

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_hist[k + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if constraints and (vname := cst.check_state(x_hist[k + 1], t_grid[k + 1], constraints)):
            u_hist[k + 1], _ = _eval_control(
                control_func, t_grid[k + 1], x_hist[k + 1], constraints, info, k + 1
            )
            info.violation_time = t_grid[k + 1]
            x_hist, u_hist = _truncate_histories(x_hist, u_hist, info, k + 1, vname)
            return x_hist, u_hist, cst.finalize_info(info)

    u_hist[-1], _ = _eval_control(
        control_func, t_grid[-1], x_hist[-1], constraints, info, len(t_grid) - 1
    )
    return x_hist, u_hist, cst.finalize_info(info)


def tracking_cost(x_hist, u_hist, xref_hist, Q, R, t_grid):
    """Discrete approximation of integral tracking cost."""
    e = x_hist - xref_hist
    stage = np.einsum("ti,tij,tj->t", e, np.broadcast_to(Q, (len(t_grid), *Q.shape)), e)
    stage += np.einsum("ti,tij,tj->t", u_hist, np.broadcast_to(R, (len(t_grid), *R.shape)), u_hist)
    dt = np.diff(t_grid)
    return np.sum(0.5 * (stage[:-1] + stage[1:]) * dt)


def regulation_cost(x_hist, u_hist, Q, R, Qf, t_grid):
    stage = np.einsum("ti,tij,tj->t", x_hist, np.broadcast_to(Q, (len(t_grid), *Q.shape)), x_hist)
    stage += np.einsum("ti,tij,tj->t", u_hist, np.broadcast_to(R, (len(t_grid), *R.shape)), u_hist)
    dt = np.diff(t_grid)
    J = np.sum(0.5 * (stage[:-1] + stage[1:]) * dt)
    J += x_hist[-1] @ Qf @ x_hist[-1]
    return J


def infinite_horizon_gain(A, B, Q, R):
    """Algebraic Riccati (validation / hover steady-state limit)."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return P, K
