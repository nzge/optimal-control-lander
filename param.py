
g = 9.8 #m/s
I = 10 #kg-m/s^2
m0 = 20 #kg
m_dry = 8 #kg, propellant depleted mass floor
T_min = 0 #N
T_max = 300 #N
α = 0.005 #kg/(N-s)
τ_min = -30 #N-m
τ_max = 30 #N-m

# Part I.5.2 descent trim (weak horizontal coupling regime)
tf_descent = 20.0  # s, mission end time
vx_descent = 40.0  # m/s, large horizontal speed
vz_descent = -8.0  # m/s, descending
theta_descent = 0.0  # rad, nearly vertical

# Part IV nonlinear near-hover horizon (short maneuver: keeps the linearization
# valid and mass well above m_dry, since hover burns T=mg and depletes ~alpha*m*g*t).
tf_part4 = 4.0  # s

# Part III mission geometry (hover-trim deviation coords)
pz_sky = 8.0  # m — Phase A rendezvous altitude above the launch site
# Circular platform (manifold M2) centred at (p_c, 0); apex (p_c, r_platform).
# p_c = 0 places the platform under the vertical ascent so M1 (ground, (0,0))
# and M2 (apex, (0, r_platform)) share the same horizontal footprint.
p_c = 0.0  # m
r_platform = 5.0  # m
tf_mission_max = 60.0  # s, upper bound on total mission time

params = {
    "g": g,
    "I": I,
    "m0": m0,
    "m_dry": m_dry,
    "T_min": T_min,
    "T_max": T_max,
    "α": α,
    "τ_min": τ_min,
    "τ_max": τ_max,
    "tf_descent": tf_descent,
    "tf_part4": tf_part4,
    "vx_descent": vx_descent,
    "vz_descent": vz_descent,
    "theta_descent": theta_descent,
    "pz_sky": pz_sky,
    "p_c": p_c,
    "r_platform": r_platform,
    "tf_mission_max": tf_mission_max,
}