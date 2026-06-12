CONTEXT SPECIFICATION: PHASE III TWO-PHASE MISSION OPTIMIZATION

1. System State & Parameter ArchitectureThe coding agent must strictly preserve the 7-dimensional state vector and 2-dimensional control vector indices throughout all dynamics integrations and residual calculations.

1.1 State Vector Mapping ($x \in \mathbb{R}^7$)x[0] : $p_x$ (Horizontal position, m)x[1] : $p_z$ (Vertical position/Altitude, m)x[2] : $v_x$ (Horizontal velocity, m/s)x[3] : $v_z$ (Vertical velocity, m/s)x[4] : $\theta$ (Pitch angle, rad)x[5] : $\omega$ (Angular velocity, rad/s)x[6] : $m$ (Current mass, kg)1.2 Costate Vector Mapping ($p \in \mathbb{R}^7$)p[0] to p[6] correspond directly as the shadow prices/adjoint variables for states x[0] to x[6].1.3 Baseline Parameters Dictionary LayoutPythonparams = {
    'g': 9.81,          # Gravity (m/s^2)
    'I': 100.0,         # Moment of Inertia (kg*m^2)
    'alpha': 0.0005,    # Fuel burn coefficient (s/m)
    'm_dry': 1000.0,    # Minimum dry mass threshold (kg)
    'T_min': 0.0,       # Minimum thrust capability (N)
    'T_max': 300.0,     # Maximum thrust capability (N)
    'tau_min': -20.0,   # Minimum torque capability (N*m)
    'tau_max': 20.0,    # Maximum torque capability (N*m)
    'p_c': 100.0,       # Circular platform center horizontal coordinate (m)
    'r_platform': 20.0  # Circular platform radius (m)
}

2. Phase III-A: Minimum-Time Ascent Specification

2.1 Mathematical Optimization FormulationCost Functional: $J_A = \int_{0}^{t_1} 1 \, dt = t_1$Linearized Dynamics Engine: $\dot{x} = A(t)x + B(t)u$Hamiltonian Structure:$$H_A = 1 + p^\top A(t)x + p^\top B(t)u$$Adjoint Differential Equations:$$\dot{p} = -A(t)^\top p$$

2.2 Control Switching Engine Logic (Bang-Bang Control)Because the control inputs appear purely linearly within $H_A$, the stationarity rule $\frac{\partial H}{\partial u} = 0$ is singular or non-informative. The control must saturate on the physical boundaries governed by the Switching Function vector $S(t) = B(t)^\top p(t)$.Let $S_1(t) = \big(B(t)^\top p(t)\big)_1$ control Thrust deviation $\delta T$.Let $S_2(t) = \big(B(t)^\top p(t)\big)_2$ control Torque $\tau$.The Cursor agent must implement the control selection exactly as follows within the ODE right-hand side function:Pythondef get_phase_a_control(t, p, params):
    B = get_B_matrix(t, m=params['m_nominal_decay']) # Context-dependent mass evaluation
    switching_function = B.T @ p
    
    # Minimize Hamiltonian: If switching function > 0, apply minimum bound. If < 0, apply max.
    T_cmd = params['T_min'] if switching_function[0] > 0 else params['T_max']
    tau_cmd = params['tau_min'] if switching_function[1] > 0 else params['tau_max']
    
    return np.array([T_cmd, tau_cmd])

2.3 Phase A Boundary Conditions$x(0) = x_0$ (Given initial launch pad state)$x(t_1) = x_1$ (Hand-off state vector; free intermediate value to be determined by the optimizer matching Phase B entry conditions).

3. Phase III-B: Optimal Landing Specification

3.1 Mathematical Optimization FormulationCost Functional: $J_B = \int_{t_1}^{t_f} \left( x^\top Q x + u^\top R u \right) dt$ (Note: Omission of $\frac{1}{2}$ coefficients must propagate a factor of 2 into the gradient).Hamiltonian Structure:$$H_B = x^\top Q x + u^\top R u + p^\top \big( A(t)x + B(t)u \big)$$Adjoint Differential Equations:$$\dot{p} = -2Qx - A(t)^\top p$$Optimal State-Feedback Law Calculation:$$\frac{\partial H_B}{\partial u} = 2Ru + B(t)^\top p = 0 \implies u^*(t) = -\frac{1}{2}R^{-1}B(t)^\top p(t)$$The coding assistant must implement control bounding via hard saturation on this law during the forward propagation segment:Pythondef get_phase_b_control(t, x, p, params):
    B = get_B_matrix(t, m=x[6]) # Evaluate using current state mass
    u_unconstrained = -0.5 * np.linalg.inv(params['R']) @ B.T @ p
    
    # Numerically Enforce Control Constraints (Saturation wrapping)
    T_cmd = np.clip(u_unconstrained[0], params['T_min'], params['T_max'])
    tau_cmd = np.clip(u_unconstrained[1], params['tau_min'], params['tau_max'])
    
    return np.array([T_cmd, tau_cmd])

3.2 Free Final Time Transversality ConditionBecause terminal touchdown time $t_f$ is an open parameter, the agent must enforce that the final value of the Hamiltonian decays to zero:$$\text{Residual}_{H_f} = H_B\big(x(t_f), u(t_f), p(t_f)\big) = 0$$


4. Geometric Landing Manifold Boundary EquationsThe shooting solver relies on matching unique boundary conditions at $t_f$ depending on the target selection. The Cursor agent must toggle between these two manifold residual sets.

4.1 Landing Manifold 1 ($M_1$) — Flat Ground TouchdownState Space Constraints: $p_z(t_f) = 0$, $v_z(t_f) = 0$, $\theta(t_f) = 0$.Free Terminal Parameters: $p_x(t_f)$, $v_x(t_f)$, $\omega(t_f)$, $m(t_f)$ are completely unconstrained on the surface.Transversality Adjoint Boundaries: The costates corresponding to unconstrained physical dimensions must drop to zero at the boundary:$$p_{p_x}(t_f) = 0, \quad p_{v_x}(t_f) = 0, \quad p_{\omega}(t_f) = 0, \quad p_{m}(t_f) = 0$$

4.2 Landing Manifold 2 ($M_2$) — Circular PlatformPlatform Boundary Equation: $h_1(x) = (p_x - p_c)^2 + p_z^2 - r^2 = 0$Attitude Constraints: $v_z(t_f) = 0$, $\theta(t_f) = 0$.Free Terminal Parameters: $v_x(t_f)$, $\omega(t_f)$, $m(t_f)$ are free.Transversality Adjoint Boundaries: * Unconstrained variables drop to zero: $p_{v_x}(t_f) = 0, \; p_{\omega}(t_f) = 0, \; p_{m}(t_f) = 0$.The position costate vector must structurally align normal to the curved sphere surface, yielding the collinearity equation to be solved as a residual:$$\text{Residual}_{\text{manifold}} = p_{p_x}(t_f) \cdot p_z(t_f) - p_{p_z}(t_f) \cdot \big(p_x(t_f) - p_c\big) = 0$$


5. Algorithmic Numerical Solution Strategy (Single Shooting)To bypass the forward-backward coupling, guide the Cursor agent to build an Indirect Single Shooting Solver using a parameter optimization vector $Z \in \mathbb{R}^{16}$.

5.1 Optimization Vector Profile ($Z$)Z[0:7]  : Initial costate guess for Phase A: $p_A(0)$Z[7:14] : Initial costate guess for Phase B entry: $p_B(t_1)$Z[14]   : Time duration of Phase A: $t_1$Z[15]   : Total mission end time: $t_f$

5.2 Mandatory Processing Pipeline to ImplementPhase A Forward Pass: Integrate the 14-state combined vector ($\dot{x} = Ax+Bu^*$, $\dot{p} = -A^\top p$) from $t=0$ to $t=Z[14]$ using the Bang-Bang control law. Save the intermediate endpoint state $x_A(t_1)$.Phase B Entry Handoff: Initialize Phase B states using the saved state boundary: $x_B(t_1) = x_A(t_1)$. Initialize Phase B costates directly from the decision vector parameters: $p_B(t_1) = Z[7:14]$.Phase B Forward Pass: Integrate the combined state-costate ODEs forward from $t=Z[14]$ to $t=Z[15]$ using the saturated state-feedback control law. Save the final conditions $x(t_f)$ and $p(t_f)$.Compute the Residual Array: Return a 16-element array to scipy.optimize.root or scipy.optimize.minimize composed of the structural errors detailed below.

5.3 The Residual Assembly MatrixPythondef objective_residuals(Z, x0, params, manifold_type='M1'):
    # Extract times
    t1 = Z[14]
    tf = Z[15]
    
    # 1. Propagate Phase A -> Extract x_A_t1
    x_A_t1, p_A_t1 = propagate_phase_a(x0, Z[0:7], t1, params)
    
    # 2. Propagate Phase B -> Extract final state/costate pairs
    x_B_tf, p_B_tf = propagate_phase_b(x_A_t1, Z[7:14], t1, tf, params)
    
    residuals = []
    
    # --- PHASE A TERMINAL OBJECTIVE TRACKING RESIDUALS ---
    # Target state constraints for the handoff location (e.g. target altitude/position)
    # Add any phase A terminal positioning constraints here
    
    # --- PHASE B MANIFOLD GEOMETRY RESIDUALS ---
    if manifold_type == 'M1':
        residuals.append(x_B_tf[1])      # p_z = 0
        residuals.append(x_B_tf[3])      # v_z = 0
        residuals.append(x_B_tf[4])      # theta = 0
        residuals.append(p_B_tf[0])      # p_px = 0 (transversality)
        residuals.append(p_B_tf[2])      # p_vx = 0 (transversality)
        residuals.append(p_B_tf[5])      # p_omega = 0 (transversality)
        residuals.append(p_B_tf[6])      # p_m = 0 (transversality)
    elif manifold_type == 'M2':
        # Platform constraint h1(x) = 0
        h1 = (x_B_tf[0] - params['p_c'])**2 + x_B_tf[1]**2 - params['r_platform']**2
        residuals.append(h1)
        residuals.append(x_B_tf[3])      # v_z = 0
        residuals.append(x_B_tf[4])      # theta = 0
        residuals.append(p_B_tf[2])      # p_vx = 0 (transversality)
        residuals.append(p_B_tf[5])      # p_omega = 0 (transversality)
        residuals.append(p_B_tf[6])      # p_m = 0 (transversality)
        # Position costate orthogonality / collinearity rule
        collinearity = p_B_tf[0]*x_B_tf[1] - p_B_tf[1]*(x_B_tf[0] - params['p_c'])
        residuals.append(collinearity)
        
    # --- TIME OPTIMALITY TRANSVERSALITY RESIDUALS ---
    H_tf = calculate_hamiltonian_b(x_B_tf, p_B_tf, tf, params)
    residuals.append(H_tf)              # H(t_f) = 0
    
    return np.array(residuals)

6. Strict Numerical Validation RequirementsThe implementation script must output diagnostic verification steps matching the formal engineering reporting expectations. Tell Cursor to create a validation print routine validating the mathematical consistency of the generated solution files:State Constraint Sanity Check: Ensure that at no point during the forward integration phase does altitude drop below negative thresholds: assert np.all(altitude_history >= -1e-3).Hamiltonian Consistency Track: Generate a separate validation plot showing $H(t)$ across the flight envelope. For Phase A, $H_A(t)$ must show a flat line at $0$. For Phase B, $H_B(t)$ must remain constant and exactly equal to $0$ at the final boundary point $t_f$.Shooting Residual Precision: Output a text log summary explicitly detailing the $\mathcal{L}_\infty$ norm of the terminal manifold constraint violations to ensure accuracy below a tolerance threshold of $10^{-5}$.