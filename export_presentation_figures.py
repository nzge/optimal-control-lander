"""Regenerate all figures for PRESENTATION_PARTS_I_II.md from project modules."""

import os

import matplotlib.pyplot as plt
import numpy as np

import analysis as ana
import dynamics as dyn
import experiments as exp
import lqr
import mission as mis
import nonlinear as nl
import validation as val
from param import params

OUT = "figures/presentation"


def _layout(fig, *, has_suptitle=False):
    """Reserve headroom so titles and suptitles are not clipped."""
    if has_suptitle:
        fig.tight_layout(pad=1.2)
        if fig._suptitle is not None:
            tops = [ax.get_position().y1 for ax in fig.axes if ax.axison]
            if tops:
                fig._suptitle.set_y(min(max(tops) + 0.02, 0.99))
    else:
        fig.tight_layout(pad=1.4)


def save_figure(fig, path, *, has_suptitle=False):
    _layout(fig, has_suptitle=has_suptitle)
    pad = 0.25 if has_suptitle else 0.2
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def generate_part1(t):
    """Part I figures: controllability/Gramian, cost sweep, hover-vs-descent,
    solution histories, mission-plane attitude quiver, and validation Hamiltonian."""
    with open("docs/controllability_matrices.tex", "w", encoding="utf-8") as f:
        f.write(dyn.controllability_latex_report(params, times=(0.0, 15.0)))

    rows = dyn.compare_along_trajectory(params)
    ts = [r["t"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(ts, [r["sigma_struct_W_hover"] for r in rows], "o-", label="hover (LTI)")
    axes[0].semilogy(ts, [r["sigma_struct_W_descent"] for r in rows], "s--", label="descent (LTV)")
    axes[0].set(
        xlabel="trim time t [s]",
        ylabel="6th Gramian singular value",
        title="Weakest structural mode over [t, tf]",
    )
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy(ts, [r["horiz_index_descent"] for r in rows], "s-", color="C3")
    axes[1].set(
        xlabel="trim time t [s]",
        ylabel="control energy to reach unit p_x",
        title="Horizontal controllability proxy (descent only)",
    )
    axes[1].grid(True, which="both", alpha=0.3)
    save_figure(fig, f"{OUT}/p1_gramian_remaining_horizon.png")

    hover_rows = ana.sweep_cost_presets(
        dyn.get_hover_dynamics, t, exp.DEFAULT_X0_REG, params, include_scaled_identity=True
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(hover_rows)))
    for r, c in zip(hover_rows, colors):
        t_eff = r["t_eff"]
        axes[0, 0].plot(t_eff, r["state_norm_trace"], color=c, alpha=0.8, label=r["preset"])
        axes[0, 1].plot(t_eff, r["weighted_state_trace"], color=c, alpha=0.8)
        axes[1, 0].plot(t_eff, r["control_norm_trace"], color=c, alpha=0.8)
        axes[1, 1].plot(r["x_hist"][:, 0], r["x_hist"][:, 1], color=c, alpha=0.8)
    axes[0, 0].set(title=r"$\|x(t)\|$", xlabel="t [s]")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].set(title=r"$\sqrt{x^\top Q x}$", xlabel="t [s]")
    axes[1, 0].set(title=r"$\|u(t)\|$", xlabel="t [s]")
    axes[1, 1].set(title=r"$p_x$ vs $p_z$", xlabel=r"$p_x$", ylabel=r"$p_z$")
    save_figure(fig, f"{OUT}/p1_cost_preset_sweep.png")

    best = min(hover_rows, key=lambda r: (r["terminal_state_norm"], r["J"]))
    Q, R, Qf = ana.cost_from_preset(best["preset"])
    alphas = np.linspace(0.25, 2.0, 8)
    cmp = ana.compare_linearizations(
        dyn.get_hover_dynamics,
        dyn.get_descent_dynamics,
        t,
        Q,
        R,
        Qf,
        exp.DEFAULT_X0_REG,
        params,
        ic_alphas=alphas,
        delta_x0=exp.DEFAULT_DELTA_X0,
    )
    xh, xd = cmp["hover"]["x_hist"], cmp["descent"]["x_hist"]
    th, td = cmp["hover"]["t_eff"], cmp["descent"]["t_eff"]
    mh, md = cmp["hover"]["metrics"], cmp["descent"]["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(th, mh["state_norm_trace"], "C0-", label="hover")
    axes[0, 0].plot(td, md["state_norm_trace"], "C3--", label="descent")
    axes[0, 1].plot(th, mh["weighted_state_trace"], "C0-")
    axes[0, 1].plot(td, md["weighted_state_trace"], "C3--")
    axes[1, 0].plot(th, mh["control_norm_trace"], "C0-")
    axes[1, 0].plot(td, md["control_norm_trace"], "C3--")
    axes[1, 1].plot(xh[:, 0], xh[:, 1], "C0-", lw=2, label="hover")
    axes[1, 1].plot(xd[:, 0], xd[:, 1], "C3--", lw=2, label="descent")
    axes[1, 1].scatter([exp.DEFAULT_X0_REG[0]], [exp.DEFAULT_X0_REG[1]], c="k", s=40, zorder=5)
    for ax, title in zip(
        axes.flat, [r"$\|x\|$", r"$\sqrt{x^\top Qx}$", r"$\|u\|$", r"$p_x$ vs $p_z$"]
    ):
        ax.set(title=title, xlabel="t [s]" if ax is not axes[1, 1] else None)
    axes[0, 0].legend()
    axes[1, 1].legend()
    save_figure(fig, f"{OUT}/p1_hover_vs_descent_2x2.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for i, lab in enumerate(["p_x", "p_z", "θ"]):
        axes[i].plot(th, xh[:, i], "C0-", label="hover")
        axes[i].plot(td, xd[:, i], "C3--", label="descent")
        axes[i].set(xlabel="t [s]", title=lab)
        axes[i].legend()
    fig.suptitle("Weakly controllable directions: horizontal & attitude transients")
    save_figure(fig, f"{OUT}/p1_weak_direction_transients.png", has_suptitle=True)

    ic_h, ic_d = cmp["ic_sensitivity"]["hover"], cmp["ic_sensitivity"]["descent"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(alphas, [r["terminal_state_norm"] for r in ic_h], "o-", label="hover")
    axes[0].plot(alphas, [r["terminal_state_norm"] for r in ic_d], "s--", label="descent")
    axes[1].plot(alphas, [r["control_energy"] for r in ic_h], "o-", label="hover")
    axes[1].plot(alphas, [r["control_energy"] for r in ic_d], "s--", label="descent")
    axes[2].plot(alphas, [r["J"] for r in ic_h], "o-", label="hover")
    axes[2].plot(alphas, [r["J"] for r in ic_d], "s--", label="descent")
    axes[0].set(title=r"$\|x(t_f)\|$ vs IC scale", xlabel=r"$\alpha$")
    axes[1].set(title=r"$\int \|u\|^2 dt$", xlabel=r"$\alpha$")
    axes[2].set(title="Cost J", xlabel=r"$\alpha$")
    axes[0].legend()
    save_figure(fig, f"{OUT}/p1_ic_sensitivity.png")

    # --- Required (Sec. 9): Part I solution trajectories vs time ---
    uh, ud = cmp["hover"]["u_hist"], cmp["descent"]["u_hist"]
    state_titles = [
        r"$p_x$ [m]", r"$p_z$ [m]", r"$v_x$ [m/s]", r"$v_z$ [m/s]",
        r"$\theta$ [rad]", r"$\omega$ [rad/s]", r"$m$ [kg]",
    ]
    ctrl_titles = [r"$\delta T$ [N]", r"$\tau$ [N$\cdot$m]"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i in range(7):
        ax = axes.flat[i]
        ax.plot(th, xh[:, i], "C0-", label="hover (controllable)")
        ax.plot(td, xd[:, i], "C3--", label="descent (weakly ctrl.)")
        ax.set(title=state_titles[i], xlabel="t [s]")
        ax.grid(True, alpha=0.3)
    for j in range(2):
        ax = axes.flat[7 + j]
        ax.plot(th, uh[:, j], "C0-")
        ax.plot(td, ud[:, j], "C3--")
        ax.set(title=ctrl_titles[j], xlabel="t [s]")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    save_figure(fig, f"{OUT}/p1_state_control_histories.png")

    # --- Required (Sec. 9): mission-plane (p_x vs p_z) with attitude quivers ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for x_hist, t_eff, color, lab in [
        (xh, th, "C0", "hover (controllable)"),
        (xd, td, "C3", "descent (weakly ctrl.)"),
    ]:
        ax.plot(x_hist[:, 0], x_hist[:, 1], color=color, lw=2, label=lab)
        step = max(1, len(x_hist) // 16)
        arrow = 0.7  # body-axis arrow length [m]; theta=0 points "up" (+p_z)
        ax.quiver(
            x_hist[::step, 0],
            x_hist[::step, 1],
            arrow * np.sin(x_hist[::step, 4]),
            arrow * np.cos(x_hist[::step, 4]),
            angles="xy", scale_units="xy", scale=1,
            color=color, alpha=0.5, width=0.004,
        )
    ax.scatter([xh[0, 0]], [xh[0, 1]], c="k", s=55, zorder=5, label=r"$x_0$ (start)")
    ax.scatter([0.0], [0.0], marker="*", c="g", s=140, zorder=6, label="trim target")
    ax.set(
        xlabel=r"$p_x$ deviation [m]",
        ylabel=r"$p_z$ deviation [m]",
        title=r"Part I mission plane: $p_x$ vs $p_z$ with attitude",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    save_figure(fig, f"{OUT}/p1_mission_plane_quiver.png")

    # Validation (Sec. 9): Hamiltonian consistency, reusing the rollouts above.
    p1_results = [
        val.validate_lqr_regulation(
            "P1 hover LTI regulation", dyn.get_hover_dynamics,
            cmp["hover"]["t_eff"], cmp["hover"]["x_hist"], cmp["hover"]["u_hist"],
            cmp["hover"]["P_interp"], Q, R, Qf, params, t_grid=t,
            saturation_fraction=cmp["hover"]["constraint_info"].saturation_fraction,
        ),
        val.validate_lqr_regulation(
            "P1 descent LTV regulation", dyn.get_descent_dynamics,
            cmp["descent"]["t_eff"], cmp["descent"]["x_hist"], cmp["descent"]["u_hist"],
            cmp["descent"]["P_interp"], Q, R, Qf, params, t_grid=t,
            saturation_fraction=cmp["descent"]["constraint_info"].saturation_fraction,
        ),
    ]
    val.export_part1_hamiltonian_figure(OUT, p1_results, save_figure)


def generate_part2(t):
    """Part II tracking figures + tracking-Hamiltonian validation."""
    p2 = exp.run_part2_tracking(params, t)
    xref, uref = p2["xref"], p2["uref"]
    x_trk, u_trk = p2["x_trk"], p2["u_trk"]
    t_ref, t_trk = p2["t_ref"], p2["t_trk"]
    Q2, R2 = p2["Q"], p2["R"]
    df = p2["dynamics_func"]
    P_trk = p2["P_trk"]
    s_interp = p2["s_interp"]
    trk_ctrl = p2["trk_ctrl"]
    x0_track = p2["x0_track"]
    trim_func = p2["constraints"].trim_func
    err = x_trk - xref[: len(x_trk)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i in range(7):
        ax = axes.flat[i]
        ax.plot(t_ref, xref[:, i], "C0")
        ax.set(title=exp.STATE_LABELS[i], xlabel="t [s]")
        ax.grid(True, alpha=0.3)
    for j in range(2):
        ax = axes.flat[7 + j]
        ax.plot(t_ref, uref[:, j], "C1")
        ax.set(title=exp.CTRL_LABELS[j], xlabel="t [s]")
        ax.grid(True, alpha=0.3)
    save_figure(fig, f"{OUT}/p2_reference_states_controls.png")

    h = p2["hover_alt"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xref[:, 0], xref[:, 1] + h, "C0-", lw=2, label=r"$x_{ref}$")
    step = max(1, len(t_ref) // 25)
    arrow = 0.8  # body-axis arrow [m]; theta=0 points "up" (+p_z)
    ax.quiver(
        xref[::step, 0],
        xref[::step, 1] + h,
        arrow * np.sin(xref[::step, 4]),
        arrow * np.cos(xref[::step, 4]),
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.6,
    )
    ax.axhline(0.0, color="k", lw=1.0, ls="-", alpha=0.7, label="ground ($p_z=0$)")
    ax.set(
        xlabel=r"$p_x$ [m]",
        ylabel=r"$p_z$ [m] (absolute)",
        title=f"Reference mission trajectory (hover at {h:.0f} m)",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=-1.0)
    ax.grid(True, alpha=0.3)
    save_figure(fig, f"{OUT}/p2_reference_mission_plane.png")

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for i in range(7):
        ax = axes.flat[i]
        ax.plot(t_ref, xref[:, i], "k--", alpha=0.7, label="ref")
        ax.plot(t_trk, x_trk[:, i], "C0", label="tracked")
        ax.set(title=exp.STATE_LABELS[i], xlabel="t [s]")
    axes.flat[7].axis("off")
    axes[0, 0].legend()
    fig.suptitle("Tracking performance: states vs reference")
    save_figure(fig, f"{OUT}/p2_tracking_states_4x2.png", has_suptitle=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(xref[:, 0], xref[:, 1] + h, "k--", lw=2, label="reference")
    ax.plot(x_trk[:, 0], x_trk[:, 1] + h, "C0", lw=2, label="tracked")
    for x_hist, color in [(xref, "k"), (x_trk, "C0")]:
        step = max(1, len(x_hist) // 18)
        ax.quiver(
            x_hist[::step, 0],
            x_hist[::step, 1] + h,
            0.8 * np.sin(x_hist[::step, 4]),
            0.8 * np.cos(x_hist[::step, 4]),
            angles="xy", scale_units="xy", scale=1,
            color=color, alpha=0.45, width=0.004,
        )
    ax.scatter([x0_track[0]], [x0_track[1] + h], c="C3", s=60, zorder=5, label=r"$x_0$ (tracker)")
    ax.scatter([0.0], [h], marker="*", c="g", s=150, zorder=6, label="hover target")
    ax.axhline(0.0, color="k", lw=1.0, ls="-", alpha=0.7, label="ground ($p_z=0$)")
    ax.set(
        xlabel=r"$p_x$ [m]",
        ylabel=r"$p_z$ [m] (absolute)",
        title="Mission plane: convergence toward reference",
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(bottom=-1.0)
    ax.grid(True, alpha=0.3)
    save_figure(fig, f"{OUT}/p2_mission_plane_convergence.png")

    weighted_err = np.array([np.sqrt(e @ Q2 @ e) for e in err])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_trk, weighted_err, "C0")
    axes[0].set(title="Weighted tracking error norm", xlabel="t [s]")
    for i in [0, 1, 4]:
        axes[1].plot(t_trk, err[:, i], label=exp.STATE_LABELS[i])
    axes[1].set(title="Error transients (selected states)", xlabel="t [s]")
    axes[1].legend()
    save_figure(fig, f"{OUT}/p2_error_transients.png")

    u_fb = np.zeros_like(u_trk)
    u_ff = np.zeros_like(u_trk)
    for k, tk in enumerate(t_trk):
        _, Bk = df(tk, params)
        Pk = P_trk(tk)
        Kk = np.linalg.inv(R2) @ Bk.T @ Pk
        u_fb[k] = -Kk @ (x_trk[k] - xref[k])
        u_ff[k] = -np.linalg.inv(R2) @ Bk.T @ s_interp(tk)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for j in range(2):
        axes[j].plot(t_trk, u_trk[:, j], "k", lw=2, label="total (saturated)")
        axes[j].plot(t_trk, u_fb[:, j], "C0--", label="feedback")
        axes[j].plot(t_trk, u_ff[:, j], "C1:", label="feedforward")
        exp.add_control_bound_hlines(axes[j], j, t_trk, params, trim_func)
        axes[j].set(ylabel=exp.CTRL_LABELS[j])
    axes[1].set(xlabel="t [s]")
    axes[0].legend()
    fig.suptitle("Transient control decomposition")
    save_figure(fig, f"{OUT}/p2_control_decomposition.png", has_suptitle=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for j in range(2):
        axes[j].plot(t_ref, uref[:, j], "k--", alpha=0.7, label="ref")
        axes[j].plot(t_trk, u_trk[:, j], "C0", label="track")
        exp.add_control_bound_hlines(axes[j], j, t_trk, params, trim_func)
        axes[j].set(ylabel=exp.CTRL_LABELS[j])
    axes[1].set(xlabel="t [s]")
    axes[0].legend()
    fig.suptitle("Control effort: reference vs tracking")
    save_figure(fig, f"{OUT}/p2_control_effort.png", has_suptitle=True)

    delta = x0_track - exp.DEFAULT_X0_REG
    alphas2 = np.linspace(0.2, 1.5, 8)
    A, B = p2["A"], p2["B"]
    term_err_ic, cost_ic = [], []
    for alpha in alphas2:
        xt, ut, _ = lqr.simulate_lti_closed_loop(
            A, B, t, trk_ctrl, exp.DEFAULT_X0_REG + alpha * delta, constraints=p2["constraints"]
        )
        t_eff = exp.align_time(t, xt)
        term_err_ic.append(np.linalg.norm(xt[-1] - xref[len(xt) - 1]))
        cost_ic.append(lqr.tracking_cost(xt, ut, xref[: len(xt)], Q2, R2, t_eff))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(alphas2, term_err_ic, "o-")
    axes[0].set(
        xlabel=r"IC scale $\alpha$",
        ylabel=r"$\|x(t_f)-x_{ref}(t_f)\|$",
        title="Terminal error vs initial-condition deviation",
    )
    axes[1].plot(alphas2, cost_ic, "s-", color="C1")
    axes[1].set(xlabel=r"IC scale $\alpha$", ylabel="tracking cost J", title="Cost vs IC deviation")
    save_figure(fig, f"{OUT}/p2_ic_robustness.png")

    rng = np.random.default_rng(42)
    noise_levels = [0.0, 0.05, 0.15, 0.30]
    term_err_ref, cost_ref = [], []
    for sigma in noise_levels:
        xref_noisy = xref + sigma * rng.standard_normal(xref.shape)
        s_n, _, _, xi = lqr.solve_tracking_feedforward(df, t, Q2, R2, xref_noisy, P_trk, params)
        ctrl = exp.make_tracking_control(df, P_trk, s_n, xi, Q2, R2, params)
        xn, un, _ = lqr.simulate_lti_closed_loop(
            A, B, t, ctrl, x0_track, constraints=p2["constraints"]
        )
        t_eff = exp.align_time(t, xn)
        term_err_ref.append(np.linalg.norm(xn[-1] - xref[len(xn) - 1]))
        cost_ref.append(lqr.tracking_cost(xn, un, xref[: len(xn)], Q2, R2, t_eff))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(noise_levels, term_err_ref, "o-", color="C2")
    axes[0].set(
        xlabel="reference noise σ",
        ylabel="terminal error vs true ref",
        title="Terminal error under perturbed reference plan",
    )
    axes[1].plot(noise_levels, cost_ref, "s-", color="C3")
    axes[1].set(
        xlabel="reference noise σ",
        ylabel="tracking cost vs true ref",
        title="Cost degradation under reference mismatch",
    )
    save_figure(fig, f"{OUT}/p2_reference_noise_robustness.png")

    # Validation (Sec. 9): tracking-Hamiltonian consistency, reusing the rollouts above.
    Q2f = p2["Qf"]
    p2_results = [
        val.validate_lqr_regulation(
            "P2 reference (regulation plan)", df, t_ref, xref, uref, p2["P_reg"],
            Q2, R2, Q2f, params, t_grid=t,
            saturation_fraction=p2["ref_info"].saturation_fraction,
        ),
        val.validate_lqr_tracking(
            "P2 tracking closed loop", df, t_trk, x_trk, u_trk, xref[: len(x_trk)],
            P_trk, s_interp, Q2, R2, params, t_grid=t,
            xref_interp=p2["xref_interp"],
            saturation_fraction=p2["trk_info"].saturation_fraction,
        ),
    ]
    val.export_part2_hamiltonian_figure(OUT, p2_results, save_figure)


def generate_validation(t):
    """Cross-cutting Section 9 residual/summary figures over all implemented parts."""
    val_results = val.run_all_validations(params, t)
    try:
        val.export_validation_figures(OUT, val_results, save_figure)
    except Exception as exc:
        print(f"Validation figure export skipped: {exc}")
    val.print_report(val_results)


def generate_part3(t):
    try:
        p3_m1 = exp.run_part3_mission(params, manifold="M1", verbose=True)
        p3_m2 = exp.run_part3_mission(params, manifold="M2", verbose=True)
        stats = mis.export_part3_figures(p3_m1, p3_m2, params, OUT, save_figure)
        val.print_mission_report(p3_m1["solution"], params)
        val.print_mission_report(p3_m2["solution"], params)
        print("Part III metrics:", stats)
    except Exception as exc:
        print(f"Part III figure export skipped: {exc}")
        import traceback
        traceback.print_exc()


def generate_part4(t):
    try:
        t4 = exp.part4_time_grid(params)
        p4 = exp.run_part4_nonlinear(params, t4, verbose=True)
        nl.print_part4_report(p4)
        p4_stats = nl.export_part4_figures(p4, OUT, save_figure)
        val.print_report(val.validate_part4(p4, params))
        print("Part IV metrics:", p4_stats)

        theta_scales = np.linspace(0.75, 2.5, 5)
        sens = nl.angle_sensitivity(
            params, t4, p4["Q"], p4["R"], p4["Qf"], exp.DEFAULT_X0_PART4, theta_scales,
        )
        ok = [r for r in sens if r.get("bvp_success")]
        if ok:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            th = [r["theta0"] for r in ok]
            axes[0].plot(th, [r["J_u_linear"] for r in ok], "o-", label="LQR / linear")
            axes[0].plot(th, [r["J_u_lqr_nl"] for r in ok], "s-", label="LQR / nonlinear")
            axes[0].plot(th, [r["J_u_pmp_nl"] for r in ok], "D--", label="PMP / nonlinear")
            axes[0].set(xlabel=r"$\theta_0$ scale (× nominal)", ylabel=r"$J_u$")
            axes[0].legend(fontsize=7)
            axes[0].grid(True, alpha=0.3)
            savings = [100.0 * (1.0 - r["J_u_pmp_nl"] / r["J_u_lqr_nl"]) for r in ok]
            axes[1].plot(th, savings, "D-", color="C2")
            axes[1].axhline(0, color="k", lw=0.6)
            axes[1].set(
                xlabel=r"$\theta_0$ scale (× nominal)",
                ylabel="PMP savings vs LQR/nonlinear [%]",
                title="Control effort gain vs pitch deviation",
            )
            axes[1].grid(True, alpha=0.3)
            save_figure(fig, f"{OUT}/p4_theta_sensitivity.png")
    except Exception as exc:
        print(f"Part IV figure export skipped: {exc}")
        import traceback
        traceback.print_exc()


PART_GENERATORS = {
    "1": generate_part1,
    "2": generate_part2,
    "3": generate_part3,
    "4": generate_part4,
    "validation": generate_validation,
}


def main(parts=None):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    t = exp.default_time_grid(params)

    selected = parts or list(PART_GENERATORS)
    for key in selected:
        print(f"--- generating: part {key} ---")
        PART_GENERATORS[key](t)

    print(f"Wrote {len(os.listdir(OUT))} figures to {OUT}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate presentation/report figures.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=list(PART_GENERATORS),
        help="Subset of parts to regenerate (default: all).",
    )
    args = parser.parse_args()
    main(args.parts)
