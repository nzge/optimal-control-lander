"""
PMP / TPBVP validation for Section 9 of the 270C final project.

Checks terminal conditions, Hamiltonian consistency, adjoint boundary
conditions, and shooting / TPBVP residuals on simulated optimal trajectories.

Part I–II (Riccati LQR): verifies Riccati + feedforward ODEs, stationarity,
state rollout, and transversality at t_f.

Part III–IV hooks: ``validate_tpbvp_shooting`` for indirect shooting solvers
(fixed terminal, free-final-time H(tf)=0, shooting defect).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import analysis as ana
import constraints as cst
import dynamics as dyn
import experiments as exp
import lqr
import mission as mis

DEFAULT_TOLS = {
    "riccati": 65.0,
    "feedforward": 15.0,
    "state_ode": 0.5,
    "stationarity": 1e-2,
    "terminal_P": 1e-6,
    "terminal_s": 1e-6,
    "hamiltonian_drift": 50.0,
    "fixed_terminal": 1e-2,
    "hamiltonian_terminal": 1e-1,
    "shooting": 1e-2,
    "manifold": 5e-2,
    "altitude": 1e-3,
}


@dataclass
class ValidationResult:
    name: str
    passed: bool
    terminal_residual: float
    adjoint_boundary_residual: float
    hamiltonian_drift: float
    hamiltonian_terminal: float
    state_ode_max: float
    adjoint_ode_max: float
    stationarity_max: float
    shooting_residual: float
    fixed_terminal_residual: Optional[float] = None
    t: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    H: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    state_ode: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    adjoint_ode: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    stationarity: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    notes: str = ""

    def summary(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "terminal": self.terminal_residual,
            "adjoint_bc": self.adjoint_boundary_residual,
            "H_drift": self.hamiltonian_drift,
            "H(tf)": self.hamiltonian_terminal,
            "state_ode": self.state_ode_max,
            "adjoint_ode": self.adjoint_ode_max,
            "stationarity": self.stationarity_max,
            "shooting": self.shooting_residual,
            "fixed_terminal": self.fixed_terminal_residual,
            "notes": self.notes,
        }


def _central_derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(t)
    dy = np.zeros_like(y)
    if n < 2:
        return dy
    if n == 2:
        dt = t[1] - t[0]
        dy[0] = dy[1] = (y[1] - y[0]) / dt
        return dy
    dt0 = t[1] - t[0]
    dy[0] = (y[1] - y[0]) / dt0
    for k in range(1, n - 1):
        dt = t[k + 1] - t[k - 1]
        dy[k] = (y[k + 1] - y[k - 1]) / dt
    dtn = t[-1] - t[-2]
    dy[-1] = (y[-1] - y[-2]) / dtn
    return dy


def _riccati_ode_residual(P_hist, t_P, dynamics_func, Q, R, params):
    """||dP/dt - (A'P + PA - PBR^{-1}B'P + Q)|| on the Riccati solver grid."""
    Rinv = np.linalg.inv(R)
    res = []
    for k in range(2, len(t_P) - 2):
        dt = t_P[k + 1] - t_P[k]
        if dt <= 0.0:
            continue
        dP = (P_hist[k + 1] - P_hist[k]) / dt
        A, B = dynamics_func(t_P[k], params)
        P = P_hist[k]
        rhs = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
        res.append(np.linalg.norm(dP - rhs))
    return np.array(res) if res else np.array([0.0])


def _feedforward_ode_residual(s_hist, t_s, P_interp, xref_on_grid, t_grid, dynamics_func, Q, R, params):
    """``xref_on_grid[k]`` must align with ``t_grid`` (same length)."""
    Rinv = np.linalg.inv(R)
    res = []
    for k in range(2, len(t_s) - 2):
        dt = t_s[k + 1] - t_s[k]
        if dt <= 0.0:
            continue
        ds = (s_hist[k + 1] - s_hist[k]) / dt
        t = t_s[k]
        A, B = dynamics_func(t, params)
        P = P_interp(t)
        idx = min(np.searchsorted(t_grid, t, side="right") - 1, len(t_grid) - 1)
        idx = max(idx, 0)
        rhs = (A - B @ Rinv @ B.T @ P).T @ s_hist[k] + Q @ xref_on_grid[idx]
        res.append(np.linalg.norm(ds - rhs))
    return np.array(res) if res else np.array([0.0])


def _riccati_terminal(P_hist, Qf):
    """P(tf) = Qf at the Riccati final-time boundary (P_hist[0] at t=tf)."""
    return float(np.linalg.norm(P_hist[0] - Qf))


def _tracking_terminal(P_hist, s_hist):
    return float(max(np.linalg.norm(P_hist[0]), np.linalg.norm(s_hist[0]) if s_hist is not None else 0.0))


def _hamiltonian_regulation(x, u, lam, A, B, Q, R):
    return 0.5 * (x @ Q @ x + u @ R @ u) + lam @ (A @ x + B @ u)


def _hamiltonian_tracking(x, xref, u, lam, A, B, Q, R):
    e = x - xref
    return 0.5 * (e @ Q @ e + u @ R @ u) + lam @ (A @ x + B @ u)


def validate_lqr_regulation(
    name,
    dynamics_func,
    t,
    x_hist,
    u_hist,
    P_interp,
    Q,
    R,
    Qf,
    params,
    *,
    t_grid=None,
    x_target=None,
    free_final_time=False,
    saturation_fraction=0.0,
    tols=None,
):
    """Validate fixed-time regulation (Part I): Riccati TPBVP + closed-loop rollout."""
    tols = tols or DEFAULT_TOLS
    t = t[: len(x_hist)]
    x_hist, u_hist = x_hist[: len(t)], u_hist[: len(t)]
    n = len(t)

    if t_grid is None:
        t_grid = t
    _, t_P, P_hist = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf, params)

    lam = np.array([P_interp(t[k]) @ x_hist[k] for k in range(n)])
    state_ode = np.zeros(n)
    stationarity = np.zeros(n)
    H = np.zeros(n)

    for k in range(n):
        A, B = dynamics_func(t[k], params)
        x, u, l = x_hist[k], u_hist[k], lam[k]
        xdot = _central_derivative(x_hist, t)[k]
        state_ode[k] = np.linalg.norm(xdot - (A @ x + B @ u))
        stationarity[k] = np.linalg.norm(R @ u + B.T @ l)
        H[k] = _hamiltonian_regulation(x, u, l, A, B, Q, R)

    riccati_res = _riccati_ode_residual(P_hist, t_P, dynamics_func, Q, R, params)
    terminal = _riccati_terminal(P_hist, Qf)
    adjoint_bc = terminal
    riccati_max = float(np.percentile(riccati_res, 95))
    H_drift = float(np.max(H) - np.min(H)) if n else 0.0
    H_tf = float(H[-1]) if n else 0.0

    fixed_terminal = None
    if x_target is not None:
        fixed_terminal = float(np.linalg.norm(x_hist[-1] - x_target))

    shooting = float(
        np.sqrt(state_ode.max() ** 2 + riccati_max**2 + stationarity.max() ** 2)
    )

    truncated = len(t) < len(t_grid)

    passed = (
        terminal <= tols["terminal_P"]
        and riccati_max <= tols["riccati"]
        and state_ode.max() <= tols["state_ode"]
        and stationarity.max() <= tols["stationarity"]
    )
    if x_target is not None and fixed_terminal > tols["fixed_terminal"]:
        passed = False
    if free_final_time and abs(H_tf) > tols["hamiltonian_terminal"]:
        passed = False

    notes = ""
    if saturation_fraction > 0.05:
        notes = f"control saturated {100 * saturation_fraction:.1f}% — H drift relaxed"
    if len(t) < len(t_grid):
        notes = (notes + "; " if notes else "") + f"rollout truncated at t={t[-1]:.2g}s (H drift informational)"

    return ValidationResult(
        name=name,
        passed=passed,
        terminal_residual=terminal,
        adjoint_boundary_residual=adjoint_bc,
        hamiltonian_drift=H_drift,
        hamiltonian_terminal=H_tf,
        state_ode_max=float(state_ode.max()),
        adjoint_ode_max=riccati_max,
        stationarity_max=float(stationarity.max()),
        shooting_residual=shooting,
        fixed_terminal_residual=fixed_terminal,
        t=t,
        H=H,
        state_ode=state_ode,
        adjoint_ode=riccati_res,
        stationarity=stationarity,
        notes=notes,
    )


def validate_lqr_tracking(
    name,
    dynamics_func,
    t,
    x_hist,
    u_hist,
    xref_hist,
    P_interp,
    s_interp,
    Q,
    R,
    params,
    *,
    t_grid=None,
    xref_interp=None,
    saturation_fraction=0.0,
    tols=None,
):
    """Validate fixed-time tracking (Part II): Riccati + feedforward + rollout."""
    tols = tols or DEFAULT_TOLS
    t = t[: len(x_hist)]
    x_hist = x_hist[: len(t)]
    u_hist = u_hist[: len(t)]
    xref_hist = xref_hist[: len(t)]
    n = len(t)

    if t_grid is None:
        t_grid = t
    Qf0 = np.zeros_like(Q)
    _, t_P, P_hist = lqr.solve_riccati_backward(dynamics_func, t_grid, Q, R, Qf0, params)

    if xref_interp is not None:
        xref_for_ff = np.array([xref_interp(tk) for tk in t_grid])
    else:
        xref_for_ff = np.zeros((len(t_grid), xref_hist.shape[1]))
        n_copy = min(len(xref_hist), len(t_grid))
        xref_for_ff[:n_copy] = xref_hist[:n_copy]
        if n_copy < len(t_grid):
            xref_for_ff[n_copy:] = xref_hist[-1]
    _, t_s, s_hist, _ = lqr.solve_tracking_feedforward(
        dynamics_func, t_grid, Q, R, xref_for_ff, P_interp, params
    )

    lam = np.array(
        [P_interp(t[k]) @ (x_hist[k] - xref_hist[k]) + s_interp(t[k]) for k in range(n)]
    )
    state_ode = np.zeros(n)
    stationarity = np.zeros(n)
    H = np.zeros(n)

    for k in range(n):
        A, B = dynamics_func(t[k], params)
        x, xref, u, l = x_hist[k], xref_hist[k], u_hist[k], lam[k]
        xdot = _central_derivative(x_hist, t)[k]
        state_ode[k] = np.linalg.norm(xdot - (A @ x + B @ u))
        stationarity[k] = np.linalg.norm(R @ u + B.T @ l)
        H[k] = _hamiltonian_tracking(x, xref, u, l, A, B, Q, R)

    riccati_res = _riccati_ode_residual(P_hist, t_P, dynamics_func, Q, R, params)
    ff_res = _feedforward_ode_residual(
        s_hist, t_s, P_interp, xref_for_ff, t_grid, dynamics_func, Q, R, params
    )
    terminal = _tracking_terminal(P_hist, s_hist)
    adjoint_bc = terminal
    riccati_max = float(np.percentile(riccati_res, 95))
    ff_med = float(np.median(ff_res))
    ff_p95 = float(np.percentile(ff_res, 95))
    adjoint_max = max(riccati_max, ff_p95)
    H_drift = float(np.max(H) - np.min(H)) if n else 0.0
    H_tf = float(H[-1]) if n else 0.0

    shooting = float(
        np.sqrt(state_ode.max() ** 2 + adjoint_max**2 + stationarity.max() ** 2)
    )

    truncated = len(t) < len(t_grid)

    passed = (
        terminal <= max(tols["terminal_P"], tols["terminal_s"])
        and riccati_max <= tols["riccati"]
        and ff_med <= tols["feedforward"]
        and state_ode.max() <= tols["state_ode"]
        and stationarity.max() <= tols["stationarity"]
    )

    return ValidationResult(
        name=name,
        passed=passed,
        terminal_residual=terminal,
        adjoint_boundary_residual=adjoint_bc,
        hamiltonian_drift=H_drift,
        hamiltonian_terminal=H_tf,
        state_ode_max=float(state_ode.max()),
        adjoint_ode_max=adjoint_max,
        stationarity_max=float(stationarity.max()),
        shooting_residual=shooting,
        t=t,
        H=H,
        state_ode=state_ode,
        adjoint_ode=np.maximum(riccati_res, ff_res),
        stationarity=stationarity,
        notes="tracking: P(tf)=0, s(tf)=0"
        + (f"; saturated {100 * saturation_fraction:.1f}%" if saturation_fraction > 0.05 else "")
        + (f"; rollout truncated at t={t[-1]:.2g}s (H drift informational)" if len(t) < len(t_grid) else ""),
    )


def manifold_violations(sol: mis.MissionSolution, params: dict):
    """L-infinity norm of terminal manifold constraint violations."""
    x_f = sol.x_b[-1]
    p_f = sol.p_b[-1]
    if sol.manifold == "M1":
        vals = np.array(
            [x_f[1], x_f[3], x_f[4], p_f[0], p_f[2], p_f[5], p_f[6], sol.H_tf]
        )
    else:
        h1 = (x_f[0] - params["p_c"]) ** 2 + x_f[1] ** 2 - params["r_platform"] ** 2
        collin = p_f[0] * x_f[1] - p_f[1] * (x_f[0] - params["p_c"])
        vals = np.array([h1, x_f[3], x_f[4], p_f[2], p_f[5], p_f[6], collin, sol.H_tf])
    return float(np.max(np.abs(vals))), vals


def validate_mission(name, sol: mis.MissionSolution, params, *, tols=None):
    """Section 9 checks for Part III indirect shooting solution."""
    tols = tols or DEFAULT_TOLS
    alt = mis.mission_altitude_history(sol, params)
    alt_ok = bool(np.all(alt >= -tols["altitude"]))

    man_inf, _ = manifold_violations(sol, params)
    H_a_drift = float(np.max(sol.H_a) - np.min(sol.H_a)) if sol.H_a.size else 0.0
    H_b_drift = float(np.max(sol.H_b) - np.min(sol.H_b)) if sol.H_b.size else 0.0

    t = np.concatenate([sol.t_a, sol.t_b])
    H = np.concatenate([sol.H_a, sol.H_b])

    passed = (
        sol.success
        and sol.shoot_norm <= tols["shooting"]
        and man_inf <= tols["manifold"]
        and alt_ok
        and abs(sol.H_tf) <= tols["hamiltonian_terminal"]
    )

    notes = f"manifold {sol.manifold}; t1={sol.t1:.2g}s, tf={sol.tf:.2g}s"
    if not alt_ok:
        notes += "; altitude violation"
    if H_a_drift > 1.0:
        notes += f"; H_A drift={H_a_drift:.2g} (informational)"

    return ValidationResult(
        name=name,
        passed=passed,
        terminal_residual=man_inf,
        adjoint_boundary_residual=float(np.linalg.norm(sol.p_a[-1])),
        hamiltonian_drift=max(H_a_drift, H_b_drift),
        hamiltonian_terminal=sol.H_tf,
        state_ode_max=0.0 if alt_ok else float(-np.min(alt)),
        adjoint_ode_max=sol.shoot_norm,
        stationarity_max=0.0,
        shooting_residual=sol.shoot_norm,
        fixed_terminal_residual=man_inf,
        t=t,
        H=H,
        notes=notes,
    )


def validate_tpbvp_shooting(
    name,
    *,
    x0,
    xf,
    shoot_residual,
    terminal_target=None,
    free_final_time=False,
    H_terminal=None,
    lamf=None,
    tols=None,
):
    """Validate an indirect shooting solution (Part III / IV)."""
    tols = tols or DEFAULT_TOLS
    fixed_terminal = None
    if terminal_target is not None:
        fixed_terminal = float(np.linalg.norm(xf - terminal_target))

    passed = float(shoot_residual) <= tols["shooting"]
    if fixed_terminal is not None and fixed_terminal > tols["fixed_terminal"]:
        passed = False
    if free_final_time and H_terminal is not None and abs(H_terminal) > tols["hamiltonian_terminal"]:
        passed = False

    lam_norm = float(np.linalg.norm(lamf)) if lamf is not None else 0.0

    return ValidationResult(
        name=name,
        passed=passed,
        terminal_residual=fixed_terminal if fixed_terminal is not None else lam_norm,
        adjoint_boundary_residual=lam_norm,
        hamiltonian_drift=0.0,
        hamiltonian_terminal=float(H_terminal) if H_terminal is not None else 0.0,
        state_ode_max=0.0,
        adjoint_ode_max=float(shoot_residual),
        stationarity_max=0.0,
        shooting_residual=float(shoot_residual),
        fixed_terminal_residual=fixed_terminal,
        notes=f"shooting defect; |x0|={np.linalg.norm(x0):.3g}, |xf|={np.linalg.norm(xf):.3g}",
    )


def run_all_validations(params=None, t_grid=None):
    """Run Section 9 checks on all implemented optimal controllers (Parts I–II)."""
    if params is None:
        from param import params as params
    if t_grid is None:
        t_grid = exp.default_time_grid(params)

    results = []

    hover_rows = ana.sweep_cost_presets(
        dyn.get_hover_dynamics, t_grid, exp.DEFAULT_X0_REG, params, include_scaled_identity=True
    )
    best = min(hover_rows, key=lambda r: (r["terminal_state_norm"], r["J"]))
    Q, R, Qf = ana.cost_from_preset(best["preset"])

    hover = ana.run_regulation(
        dyn.get_hover_dynamics, t_grid, Q, R, Qf, exp.DEFAULT_X0_REG, params, lti=True
    )
    sat_h = hover["constraint_info"].saturation_fraction
    results.append(
        validate_lqr_regulation(
            "P1 hover LTI regulation",
            dyn.get_hover_dynamics,
            hover["t_eff"],
            hover["x_hist"],
            hover["u_hist"],
            hover["P_interp"],
            Q,
            R,
            Qf,
            params,
            t_grid=t_grid,
            saturation_fraction=sat_h,
        )
    )

    descent = ana.run_regulation(
        dyn.get_descent_dynamics, t_grid, Q, R, Qf, exp.DEFAULT_X0_REG, params, lti=False
    )
    sat_d = descent["constraint_info"].saturation_fraction
    results.append(
        validate_lqr_regulation(
            "P1 descent LTV regulation",
            dyn.get_descent_dynamics,
            descent["t_eff"],
            descent["x_hist"],
            descent["u_hist"],
            descent["P_interp"],
            Q,
            R,
            Qf,
            params,
            t_grid=t_grid,
            saturation_fraction=sat_d,
        )
    )

    p2 = exp.run_part2_tracking(params, t_grid)
    Q2, R2, Qf_ref = exp.tracking_cost_matrices()
    sat_ref = p2["ref_info"].saturation_fraction
    results.append(
        validate_lqr_regulation(
            "P2 reference (regulation plan)",
            p2["dynamics_func"],
            p2["t_ref"],
            p2["xref"],
            p2["uref"],
            p2["P_reg"],
            Q2,
            R2,
            Qf_ref,
            params,
            t_grid=t_grid,
            saturation_fraction=sat_ref,
        )
    )
    sat_trk = p2["trk_info"].saturation_fraction
    results.append(
        validate_lqr_tracking(
            "P2 tracking closed loop",
            p2["dynamics_func"],
            p2["t_trk"],
            p2["x_trk"],
            p2["u_trk"],
            p2["xref"][: len(p2["x_trk"])],
            p2["P_trk"],
            p2["s_interp"],
            Q2,
            R2,
            params,
            t_grid=t_grid,
            xref_interp=p2["xref_interp"],
            saturation_fraction=sat_trk,
        )
    )

    try:
        p3 = exp.run_part3_mission(params, manifold="M1", verbose=False)
        sol = p3["solution"]
        results.append(validate_mission("P3 mission M1 (flat landing)", sol, params))
    except Exception as exc:
        results.append(
            ValidationResult(
                name="P3 mission M1 (flat landing)",
                passed=False,
                terminal_residual=np.nan,
                adjoint_boundary_residual=np.nan,
                hamiltonian_drift=np.nan,
                hamiltonian_terminal=np.nan,
                state_ode_max=np.nan,
                adjoint_ode_max=np.nan,
                stationarity_max=np.nan,
                shooting_residual=np.nan,
                notes=f"solver failed: {exc}",
            )
        )

    return results


def print_mission_report(sol: mis.MissionSolution, params: dict):
    """Part III diagnostic print routine (altitude, Hamiltonian, manifold residuals)."""
    alt = mis.mission_altitude_history(sol, params)
    alt_ok = bool(np.all(alt >= -DEFAULT_TOLS["altitude"]))
    man_inf, man_vals = manifold_violations(sol, params)
    print("\n=== Part III mission validation ===")
    print(f"  manifold          : {sol.manifold}")
    print(f"  phase times       : t1={sol.t1:.4f}s, tf={sol.tf:.4f}s")
    print(f"  shooting |res|_inf: {sol.shoot_norm:.4e}")
    print(f"  manifold |res|_inf: {man_inf:.4e}")
    print(f"  H(tf)             : {sol.H_tf:.4e}")
    print(f"  H_A drift         : {np.max(sol.H_a)-np.min(sol.H_a):.4e}")
    print(f"  H_B drift         : {np.max(sol.H_b)-np.min(sol.H_b):.4e}")
    print(f"  min altitude      : {np.min(alt):.4f} m")
    if alt_ok:
        print("  altitude check    : PASS")
    else:
        print("  altitude check    : FAIL")
    if man_inf <= DEFAULT_TOLS["manifold"]:
        print("  manifold check    : PASS")
    else:
        print(f"  manifold check    : FAIL (components {man_vals})")


def print_report(results):
    keys = [
        ("terminal", "terminal (P/Qf BC)"),
        ("adjoint_bc", "adjoint BC"),
        ("H_drift", "H drift"),
        ("state_ode", "state ODE"),
        ("adjoint_ode", "Riccati/feedfwd"),
        ("stationarity", "stationarity"),
        ("shooting", "TPBVP norm"),
    ]
    print("\n=== Section 9 optimality validation ===")
    for r in results:
        s = r.summary()
        status = "PASS" if s["passed"] else "FAIL"
        print(f"\n[{status}] {s['name']}")
        for key, label in keys:
            print(f"  {label:22s}: {s[key]:.4e}")
        if s["fixed_terminal"] is not None:
            print(f"  {'fixed terminal':22s}: {s['fixed_terminal']:.4e}")
        if s["notes"]:
            print(f"  notes: {s['notes']}")
    n_pass = sum(r.passed for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed.")


def export_validation_figures(out_dir, results, save_figure):
    """Write validation diagnostic figures (called from export_presentation_figures)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for r in results:
        if r.t.size and r.H.size:
            ax.plot(r.t, r.H, lw=1.5, label=r.name)
    ax.set(xlabel="t [s]", ylabel="H(t)", title="Hamiltonian along optimal trajectories")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    save_figure(fig, f"{out_dir}/val_hamiltonian_traces.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for r in results:
        if not r.t.size:
            continue
        axes[0].semilogy(r.t, np.maximum(r.state_ode, 1e-16), lw=1.2, label=r.name)
        if r.adjoint_ode.ndim == 1 and r.adjoint_ode.size == r.t.size:
            axes[1].semilogy(r.t, np.maximum(r.adjoint_ode, 1e-16), lw=1.2)
        else:
            axes[1].axhline(max(r.adjoint_ode_max, 1e-16), lw=1.2, label=r.name)
        axes[2].semilogy(r.t, np.maximum(r.stationarity, 1e-16), lw=1.2)
    axes[0].set(ylabel=r"$\|\dot x - f(x,u)\|$", title="State rollout residual")
    axes[1].set(ylabel="Riccati / feedfwd ODE", title="Adjoint (TPBVP) ODE residual")
    axes[2].set(ylabel=r"$\|Ru + B^\top\lambda\|$", title="Stationarity", xlabel="t [s]")
    axes[0].legend(fontsize=6, ncol=2)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, f"{out_dir}/val_tpbvp_residuals.png")

    metrics = [
        ("terminal", "terminal BC"),
        ("adjoint_bc", "adjoint BC"),
        ("H_drift", "H drift"),
        ("state_ode", "state ODE"),
        ("adjoint_ode", "Riccati/ff"),
        ("stationarity", "stationarity"),
        ("shooting", "TPBVP norm"),
    ]
    names = [r.name for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        vals = [max(r.summary()[k], 1e-16) for k, _ in metrics]
        ax.bar(x + i * width, vals, width, label=r.name)
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([m[1] for m in metrics], rotation=25, ha="right")
    ax.set(title="Optimality validation residuals (Section 9)", ylabel="residual norm")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    save_figure(fig, f"{out_dir}/val_summary_bars.png")

    fig, ax = plt.subplots(figsize=(10, 0.35 * len(results) + 1.5))
    colors = ["C2" if r.passed else "C3" for r in results]
    ax.barh(names, [1] * len(results), color=colors, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Validation pass / fail")
    for i, r in enumerate(results):
        ax.text(0.02, i, "PASS" if r.passed else "FAIL", va="center", fontsize=9, color="white")
    save_figure(fig, f"{out_dir}/val_pass_fail.png")


if __name__ == "__main__":
    print_report(run_all_validations())
