#!/usr/bin/env python3
"""
Correct noise calculation for integrated white noise
"""

import numpy as np

# From our previous analysis
avg_separation = 2.03e-08  # Conductance separation
dt = 0.5e-9  # Time step
n_points = 7940  # Number of integration points
t_end = n_points * dt

print("Correct Noise Calculation Analysis")
print("=" * 40)

# For a given SNR, let's calculate what the noise parameters should be
snr = 1.0  # Example SNR

# The white noise formula: Y = (<d_ij> / snr)^2 / dt
Y = (avg_separation / snr)**2 / dt
print(f"For SNR = {snr}:")
print(f"  Y parameter: {Y:.2e}")
print(f"  Noise std per time step: {np.sqrt(Y * dt):.2e}")

# After integration over n_points time steps:
# The integrated noise has variance = Y * dt * n_points
# So the std of integrated noise = sqrt(Y * dt * n_points)
integrated_noise_std = np.sqrt(Y * dt * n_points)
print(f"  Integrated noise std: {integrated_noise_std:.2e}")

# The signal separation after integration should be:
# avg_separation * n_points (since signal grows linearly with time)
integrated_signal_separation = avg_separation * n_points
print(f"  Integrated signal separation: {integrated_signal_separation:.2e}")

# The effective SNR after integration should be:
effective_snr = integrated_signal_separation / integrated_noise_std
print(f"  Effective SNR: {effective_snr:.3f}")

print(f"\nThe issue with the empirical calculation:")
print(f"The empirical code calculates:")
print(f"  noise_level = sqrt(var(I_integrated) + var(Q_integrated))")
print(f"  where I_integrated and Q_integrated are the final integrated values")
print(f"")
print(f"This is WRONG because:")
print(f"1. It calculates the variance of the integrated signal across realizations")
print(f"2. But the integrated signal includes both the deterministic part and the noise")
print(f"3. The variance of the integrated signal is NOT the same as the noise level")
print(f"")
print(f"The correct approach would be:")
print(f"1. Calculate the noise component separately from the signal")
print(f"2. Or use the theoretical noise level: sqrt(Y * dt * n_points)")
print(f"3. Or calculate the noise from the variance of the noise increments")

# Let's check what the empirical calculation would give
print(f"\nWhat the empirical calculation gives:")
print(f"If we assume the empirical calculation gives approximately the integrated noise std,")
print(f"then the empirical SNR would be:")
print(f"  empirical_snr = integrated_signal_separation / empirical_noise_level")
print(f"")
print(f"If empirical_noise_level ≈ integrated_noise_std, then empirical_snr ≈ {effective_snr:.3f}")
print(f"But if empirical_noise_level is calculated incorrectly, it could be much smaller")
print(f"leading to a much larger empirical SNR")

# The key insight: the empirical calculation might be calculating the variance
# of the integrated signal, which includes both signal and noise components
# This would give a smaller "noise level" than the actual noise level

print(f"\nPotential fix:")
print(f"The empirical noise level should be calculated as:")
print(f"  noise_level = sqrt(Y * dt * n_points)")
print(f"where Y is the white noise parameter from the simulation")
print(f"")
print(f"This would give the correct effective SNR that matches the input SNR")