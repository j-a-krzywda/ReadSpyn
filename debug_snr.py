#!/usr/bin/env python3
"""
Debug script to understand the SNR issue in white noise example
"""

import numpy as np
import jax.numpy as jnp

# From the white_noise_example.py
Cdd = jnp.array([[1.0, 0.1], [0.1, 1.0]])  # 2x2 dot-dot capacitance matrix
Cds = jnp.array([[0.6], [0.3]])  # 2x1 dot-sensor coupling matrix

# Calculate Cdd_inv
Cdd_inv = jnp.linalg.inv(Cdd)
print("Cdd_inv:")
print(Cdd_inv)

# Calculate coupling matrix
coupling_matrix = Cds.T @ Cdd_inv
print("\nCoupling matrix (Cds.T @ Cdd_inv):")
print(coupling_matrix)

# Define charge states
charge_states = jnp.array([
    [0, 0],  # Both dots empty
    [1, 0],  # First dot occupied
    [0, 1],  # Second dot occupied
    [1, 1]   # Both dots occupied
])

print("\nCharge states:")
for i, state in enumerate(charge_states):
    print(f"  State {i}: {state}")

# Calculate energy offsets for each charge state
# energy_offset = (self.Cds.T @ self.Cdd_inv @ charge_state + eps0)
# with eps0 = 0.0 and sensor_voltages = zeros
eps0 = 0.0
sensor_voltages = jnp.zeros(1)  # 1 sensor

print("\nEnergy offsets for each charge state:")
energy_offsets = []
for i, charge_state in enumerate(charge_states):
    energy_offset = coupling_matrix @ charge_state + eps0
    energy_offsets.append(energy_offset[0])  # First (and only) sensor
    print(f"  State {i}: {float(energy_offset[0]):.6f}")

# Calculate conductance values
# From the code: conductance_fun(eps) = 2 * cosh(2 * eps / eps_w)^(-2) / R0
eps_w = 1.0  # Energy width
R0 = 50e6    # Base resistance (from g0 = 1/50)

print(f"\nConductance calculation parameters:")
print(f"  eps_w: {eps_w}")
print(f"  R0: {R0}")

print("\nConductance values for each charge state:")
conductances = []
for i, eps in enumerate(energy_offsets):
    conductance = 2 * jnp.cosh(2 * eps / eps_w)**(-2) / R0
    conductances.append(float(conductance))
    print(f"  State {i}: {conductance:.2e}")

# Calculate separations between charge states
print("\nConductance separations between charge state pairs:")
separations = []
for i in range(len(charge_states)):
    for j in range(i + 1, len(charge_states)):
        separation = abs(conductances[i] - conductances[j])
        separations.append(separation)
        print(f"  States {i} vs {j}: {separation:.2e}")

avg_separation = np.mean(separations)
print(f"\nAverage conductance separation: {avg_separation:.2e}")

# Now let's understand the white noise formula
print(f"\nWhite noise formula analysis:")
print(f"Formula: Y = (<d_ij> / snr)^2 / dt")
print(f"Where:")
print(f"  <d_ij> = average separation = {avg_separation:.2e}")
print(f"  dt = time step = 0.5e-9 s")
print(f"  snr = user-specified SNR")

# For different SNR values
snr_values = [0.1, 0.2, 0.5, 1.0]
dt = 0.5e-9

print(f"\nY parameter for different SNR values:")
for snr in snr_values:
    Y = (avg_separation / snr)**2 / dt
    print(f"  SNR = {snr}: Y = {Y:.2e}")

# The issue: if avg_separation is very small, then Y becomes very small
# This means the white noise amplitude becomes very small
# But the effective SNR calculation might be wrong

print(f"\nExpected noise standard deviation:")
for snr in snr_values:
    Y = (avg_separation / snr)**2 / dt
    expected_noise_std = np.sqrt(Y * dt)
    print(f"  SNR = {snr}: expected noise std = {expected_noise_std:.2e}")

print(f"\nExpected effective SNR (should equal input SNR):")
for snr in snr_values:
    Y = (avg_separation / snr)**2 / dt
    expected_noise_std = np.sqrt(Y * dt)
    expected_effective_snr = avg_separation / expected_noise_std
    print(f"  SNR = {snr}: expected effective SNR = {expected_effective_snr:.3f}")

# The problem might be in how the effective SNR is calculated
print(f"\nPotential issue analysis:")
print(f"If the empirical SNR is 60x too large, this suggests:")
print(f"1. The noise level calculation is wrong (too small)")
print(f"2. The signal separation calculation is wrong (too large)")
print(f"3. There's a unit mismatch somewhere")
print(f"4. The integration process is not accounted for correctly")