
g = 9.8 #m/s
I = 10 #kg-m/s^2
m0 = 20 #kg
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

params = {
    "g": g,
    "I": I,
    "m0": m0,
    "T_min": T_min,
    "T_max": T_max,
    "α": α,
    "τ_min": τ_min,
    "τ_max": τ_max,
    "tf_descent": tf_descent,
    "vx_descent": vx_descent,
    "vz_descent": vz_descent,
    "theta_descent": theta_descent,
}