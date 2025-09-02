#!/usr/bin/env python3
"""
Analyze the effect of integration on SNR calculation
"""

import numpy as np
import jax.numpy as jnp

# From the debug output
avg_separation = 2.03e-08  # Conductance separation
dt = 0.5e-9  # Time step
t_end = 1000 * 3.97e-9  # 1000 resonant periods
n_points = int(t_end / dt)

print("Integration Effect Analysis")
print("=" * 40)
print(f"Average conductance separation: {avg_separation:.2e}")
print(f"Time step: {dt:.1e} s")
print(f"Total time: {t_end:.1e} s")
print(f"Number of integration points: {n_points}")

# The key insight: integration amplifies the signal
# If we have a constant signal S, then integrated signal = S * t
# If we have noise with std σ, then integrated noise std = σ * sqrt(t)

print(f"\nIntegration amplification:")
print(f"Signal amplification factor: {n_points} (linear in time)")
print(f"Noise amplification factor: {np.sqrt(n_points):.1f} (square root of time)")

# This means the SNR after integration should be:
# SNR_integrated = (S * t) / (σ * sqrt(t)) = (S / σ) * sqrt(t)
# So SNR_integrated = SNR_instantaneous * sqrt(t)

sqrt_n_points = np.sqrt(n_points)
print(f"SNR amplification factor: {sqrt_n_points:.1f}")

# Let's check if this explains the 60x factor
print(f"\nChecking if this explains the 60x factor:")
print(f"sqrt(n_points) = {sqrt_n_points:.1f}")
print(f"Is this close to 60? {abs(sqrt_n_points - 60) < 5}")

# Let's also check the time scaling
print(f"\nTime scaling analysis:")
print(f"Total integration time: {t_end*1e6:.2f} μs")
print(f"Resonant period: {3.97e-9*1e9:.2f} ns")
print(f"Number of resonant periods: {t_end / 3.97e-9:.0f}")

# The issue might be in the noise calculation
# In the empirical calculation, they calculate noise as:
# noise_level = sqrt(var(I_integrated) + var(Q_integrated))
# But this might not be the right way to calculate noise after integration

print(f"\nNoise calculation issue:")
print(f"The empirical calculation uses the variance of integrated signals")
print(f"But for white noise, the variance of integrated white noise grows as t")
print(f"So if the noise variance is calculated incorrectly, it could be off by a factor")

# Let's think about the correct way to calculate noise after integration
# For white noise with variance σ² per time step, after integration:
# - The integrated signal has variance σ² * t
# - But the standard deviation is σ * sqrt(t)
# - However, if we're looking at the variance across realizations, we need to be careful

print(f"\nCorrect noise calculation for integrated white noise:")
print(f"If white noise has std σ per time step, then:")
print(f"- Integrated noise std = σ * sqrt(t)")
print(f"- But if we calculate variance across realizations of integrated signals,")
print(f"  we get the variance of the final integrated value")
print(f"- This variance should be σ² * t, so std = σ * sqrt(t)")

# The 60x factor suggests there might be a unit mismatch or calculation error
print(f"\nPotential issues:")
print(f"1. Unit mismatch between conductance and IQ units")
print(f"2. Incorrect noise variance calculation")
print(f"3. Missing factor in the integration process")
print(f"4. The empirical calculation might be using the wrong metric")

# Let's check if there's a factor of sqrt(2) or similar missing
print(f"\nChecking for missing factors:")
print(f"sqrt(2) = {np.sqrt(2):.3f}")
print(f"sqrt(60) = {np.sqrt(60):.3f}")
print(f"60 / sqrt(n_points) = {60 / sqrt_n_points:.3f}")