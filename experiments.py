"""
Shared experiment setup for Parts I–II: ICs, time grids, and Part II tracking pipeline.
"""

import numpy as np

import constraints as cst
import dynamics as dyn
import lqr
import mission as mis
from analysis import CTRL_LABELS, STATE_LABELS

DEFAULT_X0_REG = np.array([5.0, 8.0, 2.0, -1.5, 0.08, 0.0, 0.3])
DEFAULT_X0_TRACK = np.array([-12.0, 15.0, -4.0, 3.0, -0.25, 0.4, -0.8])
DEFAULT_DELTA_X0 = np.array([8.0, 5.0, -3.0, 2.0, -0.15, 0.25, -0.5])
DEFAULT_X0_MISSION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

TRACKING_Q_DIAG = [1.0, 1.0, 0.5, 0.5, 10.0, 5.0, 0.01]
TRACKING_R_DIAG = [0.1, 1.0]
TRACKING_QF_DIAG = [10.0, 10.0, 1.0, 1.0, 50.0, 20.0, 0.1]


def default_time_grid(params, n=401):
    return np.linspace(0.0, params["tf_descent"], n)


def rollout_constraints(params, dynamics_func, *, enforce_control=True, enforce_state=True):
    """Build constraint config matched to a dynamics linearization."""
    return cst.RolloutConstraints(
        params=params,
        trim_func=cst.trim_for_dynamics(dynamics_func),
        enforce_control=enforce_control,
        enforce_state=enforce_state,
    )


def align_time(t_grid, x_hist):
    """Match time samples to a possibly truncated state history."""
    return t_grid[: len(x_hist)]


def tracking_cost_matrices():
    Q = np.diag(TRACKING_Q_DIAG)
    R = np.diag(TRACKING_R_DIAG)
    Qf = np.diag(TRACKING_QF_DIAG)
    return Q, R, Qf


def make_tracking_control(dynamics_func, P_interp, s_interp, xref_interp, Q, R, params):
    return lambda tt, xx: lqr.tracking_control(
        tt, xx, xref_interp(tt), dynamics_func, P_interp, s_interp, Q, R, params
    )[0]


def run_part2_tracking(
    params,
    t_grid=None,
    *,
    x0_ref=None,
    x0_track=None,
    dynamics_func=None,
    enforce_control=True,
    enforce_state=True,
):
    
    """
    Part II pipeline: regulation reference → tracking Riccati/feedforward → rollout.
    Returns dict with trajectories, interpolators, and constraint diagnostics.
    """
    
    if t_grid is None:
        t_grid = default_time_grid(params)
    if x0_ref is None:
        x0_ref = DEFAULT_X0_REG.copy()
    if x0_track is None:
        x0_track = DEFAULT_X0_TRACK.copy()
    if dynamics_func is None:
        dynamics_func = dyn.get_hover_dynamics

    Q, R, Qf = tracking_cost_matrices()
    ref_constraints = rollout_constraints(
        params, dynamics_func, enforce_control=enforce_control, enforce_state=False
    )
    trk_constraints = rollout_constraints(
        params, dynamics_func, enforce_control=enforce_control, enforce_state=enforce_state
    )

    A, B = dynamics_func(0.0, params)
    P_reg, _, _ = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf, params)
    reg_ctrl = lambda tt, xx: lqr.regulation_control(tt, xx, dynamics_func, P_reg, Q, R, params)
    xref, uref, ref_info = lqr.simulate_lti_closed_loop(
        A, B, t_grid, reg_ctrl, x0_ref, constraints=ref_constraints
    )
    t_ref = align_time(t_grid, xref)

    P_trk, _, _ = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, np.zeros((7, 7)), params)
    s_interp, _, _, xref_interp = lqr.solve_tracking_feedforward(
        dynamics_func, t_grid, Q, R, xref, P_trk, params
    )
    trk_ctrl = make_tracking_control(dynamics_func, P_trk, s_interp, xref_interp, Q, R, params)
    x_trk, u_trk, trk_info = lqr.simulate_lti_closed_loop(
        A, B, t_grid, trk_ctrl, x0_track, constraints=trk_constraints
    )
    t_trk = align_time(t_grid, x_trk)

    return {
        "t_grid": t_grid,
        "t_ref": t_ref,
        "t_trk": t_trk,
        "params": params,
        "Q": Q,
        "R": R,
        "Qf": Qf,
        "A": A,
        "B": B,
        "dynamics_func": dynamics_func,
        "ref_constraints": ref_constraints,
        "constraints": trk_constraints,
        "x0_ref": x0_ref,
        "x0_track": x0_track,
        "xref": xref,
        "uref": uref,
        "ref_info": ref_info,
        "x_trk": x_trk,
        "u_trk": u_trk,
        "trk_info": trk_info,
        "P_reg": P_reg,
        "P_trk": P_trk,
        "s_interp": s_interp,
        "xref_interp": xref_interp,
        "trk_ctrl": trk_ctrl,
    }


def run_part3_mission(
    params,
    *,
    x0=None,
    manifold="M1",
    Z0=None,
    verbose=False,
    multi_start=True,
):
    """
    Part III pipeline: min-time ascent + free-final-time landing (single shooting).
    Returns ``MissionSolution`` and cost matrices for Phase B.
    """
    if x0 is None:
        x0 = DEFAULT_X0_MISSION.copy()
    Q, R = mis.mission_cost_matrices()
    sol = mis.solve_mission(
        x0, params, manifold=manifold, Z0=Z0, verbose=verbose,
        multi_start=multi_start, target_tol=mis.TARGET_TOL,
    )
    return {
        "solution": sol,
        "x0": x0,
        "params": params,
        "Q": Q,
        "R": R,
        "manifold": manifold,
    }


def add_control_bound_hlines(ax, channel, t, params, trim_func, *, color="C3", alpha=0.5):
    """Plot time-varying delta_T or tau bounds on a control axis."""
    lo = np.zeros_like(t)
    hi = np.zeros_like(t)
    for k, tk in enumerate(t):
        bounds_lo, bounds_hi = cst.control_bounds(tk, params, trim_func)
        lo[k] = bounds_lo[channel]
        hi[k] = bounds_hi[channel]
    ax.plot(t, hi, color=color, ls=":", alpha=alpha, label="max bound")
    ax.plot(t, lo, color=color, ls=":", alpha=alpha, label="min bound")
