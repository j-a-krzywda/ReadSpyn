#!/usr/bin/env python3
"""
JAX-based ReadSpyn Example Script

This script demonstrates the new JAX-based implementation of ReadSpyn, which uses 
efficient state scanning and precomputed noise trajectories.

Key Features:
- Precomputed noise trajectories for efficiency
- JAX scan for vectorized state processing
- White noise added in post-processing
- GPU acceleration support

Author: Jan A. Krzywda
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import ReadSpyn components
from readout_simulator import (
    QuantumDotSystem, 
    RLC_sensor, 
    JAXReadoutSimulator,
    OU_noise, 
    OverFNoise
)

def main():
    """Main function demonstrating JAX-based ReadSpyn."""
    
    # Set plotting style
    rcParams['figure.figsize'] = (12, 8)
    rcParams['font.size'] = 12
    
    print("=== JAX-based ReadSpyn Example ===\n")
    
    # 1. Define Quantum Dot System
    print("1. Creating quantum dot system...")
    Cdd = jnp.array([[1, 0], [0, 1]])  # 2x2 dot-dot capacitance matrix
    Cds = jnp.array([[1], [0.1]]) * 0.6  # 2x1 dot-sensor coupling matrix
    dot_system = QuantumDotSystem(Cdd, Cds)
    print(f"   ✓ Quantum dot system created with {dot_system.num_dots} dots and {dot_system.num_sensors} sensors")
    
    # 2. Configure Sensor Parameters
    print("2. Configuring sensor...")
    params_resonator = {
        'Lc': 800e-9,
        'Cp': 0.5e-12,
        'RL': 40,
        'Rc': 100e6,
        'Z0': 50
    }
    
    params_coulomb_peak = {
        'g0': 1/50/1e6,
        'eps0': 0.5,
        'eps_width': 1
    }
    
    # Create noise models with increased g-noise for visibility
    eps_noise = OverFNoise(
        n_fluctuators=3, S1=1e-4, sigma_couplings=0.1,
        ommax=1e6, ommin=1e3, equally_dist=True
    )
    
    c_noise = OU_noise(sigma=1e-12, gamma=1e5)
    
    # Create sensor
    sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
    print(f"   ✓ Sensor created with resonant frequency: {sensor.f0/1e9:.2f} GHz")
    
    # 3. Create JAX Simulator
    print("3. Creating JAX simulator...")
    simulator = JAXReadoutSimulator(dot_system, [sensor])
    print("   ✓ JAX simulator created successfully")
    
    # 4. Define Simulation Parameters
    print("4. Setting up simulation parameters...")
    t_end = 1000 * sensor.T0  # 1000 oscillation periods
    dt = 0.5e-9  # Time step
    times = jnp.arange(0, t_end, dt)
    
    # Define charge states
    charge_states = jnp.array([
        [1, 0],  # State 0
        [0, 1],  # State 1
        [1, 1]   # State 2
    ])
    
    # Simulation parameters with reduced white noise for g-noise visibility
    params = {
        'SNR_white': 1.0,
        'eps0': 0.5,
        'plot_conductance': True
    }
    
    print(f"   ✓ Simulation time: {t_end*1e6:.1f} μs")
    print(f"   ✓ Time points: {len(times)}")
    print(f"   ✓ Charge states: {len(charge_states)}")
    
    # 5. Precompute Noise Trajectories
    print("5. Precomputing noise trajectories...")
    key = jax.random.PRNGKey(42)
    n_realizations = 50
    
    start_time = time.time()
    simulator.precompute_noise(key, times, n_realizations, eps_noise)
    noise_time = time.time() - start_time
    
    print(f"   ✓ Precomputed {n_realizations} noise trajectories in {noise_time:.2f} seconds")
    
    # 6. Run Simulation
    print("6. Running simulation...")
    key = jax.random.PRNGKey(123)
    
    start_time = time.time()
    results = simulator.run_simulation(charge_states, times, params, key)
    sim_time = time.time() - start_time
    
    print(f"   ✓ Simulation completed in {sim_time:.2f} seconds")
    
    # 7. Analyze Results
    print("7. Analyzing results...")
    I_integrated, Q_integrated = simulator.get_integrated_IQ(sensor_idx=0)
    fidelity = simulator.calculate_fidelity(sensor_idx=0)
    
    print(f"   ✓ Readout fidelity: {fidelity:.3f}")
    print(f"   ✓ IQ data shape: {I_integrated.shape}")
    
    # 8. Performance Summary
    print("\n=== Performance Summary ===")
    total_time = noise_time + sim_time
    total_operations = len(charge_states) * n_realizations * len(times)
    ops_per_second = total_operations / total_time
    
    print(f"Noise precomputation time: {noise_time:.2f} seconds")
    print(f"Simulation time: {sim_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total operations: {total_operations:,}")
    print(f"Operations per second: {ops_per_second:,.0f}")
    print(f"Readout fidelity: {fidelity:.3f}")
    
    # 9. Create Visualizations
    print("\n8. Creating visualizations...")
    create_plots(I_integrated, Q_integrated, times, fidelity, charge_states)
    
    # 10. Demonstrate Advanced Features
    print("\n9. Demonstrating advanced features...")
    demonstrate_advanced_features(simulator)
    
    print("\n=== Example Completed Successfully! ===")
    return True

def create_plots(I_integrated, Q_integrated, times, fidelity, charge_states):
    """Create visualization plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['red', 'green', 'blue']
    
    # Plot 1: I component evolution
    for state_idx in range(I_integrated.shape[0]):
        I_mean = jnp.mean(I_integrated[state_idx], axis=0)
        I_std = jnp.std(I_integrated[state_idx], axis=0)
        
        axes[0, 0].plot(times * 1e6, I_mean, label=f'State {state_idx}', 
                        color=colors[state_idx], linewidth=2)
        axes[0, 0].fill_between(times * 1e6, I_mean - I_std, I_mean + I_std, 
                                alpha=0.3, color=colors[state_idx])
    
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].set_ylabel('Integrated I')
    axes[0, 0].set_title('I Component Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q component evolution
    for state_idx in range(Q_integrated.shape[0]):
        Q_mean = jnp.mean(Q_integrated[state_idx], axis=0)
        Q_std = jnp.std(Q_integrated[state_idx], axis=0)
        
        axes[0, 1].plot(times * 1e6, Q_mean, label=f'State {state_idx}', 
                        color=colors[state_idx], linewidth=2)
        axes[0, 1].fill_between(times * 1e6, Q_mean - Q_std, Q_mean + Q_std, 
                                alpha=0.3, color=colors[state_idx])
    
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('Integrated Q')
    axes[0, 1].set_title('Q Component Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final IQ points
    for state_idx in range(I_integrated.shape[0]):
        I_final = I_integrated[state_idx, :, -1]
        Q_final = Q_integrated[state_idx, :, -1]
        
        axes[1, 0].scatter(I_final, Q_final, c=colors[state_idx], 
                          label=f'State {state_idx}', alpha=0.7)
    
    axes[1, 0].set_xlabel('Integrated I')
    axes[1, 0].set_ylabel('Integrated Q')
    axes[1, 0].set_title(f'IQ Readout Results (Fidelity: {fidelity:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Plot 4: Performance metrics
    total_operations = len(charge_states) * I_integrated.shape[1] * len(times)
    memory_usage = total_operations * 8 / 1024 / 1024  # MB for float64
    
    metrics = ['States', 'Realizations', 'Time Points', 'Total Operations']
    values = [len(charge_states), I_integrated.shape[1], len(times), total_operations]
    
    axes[1, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_title('Simulation Parameters')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add text annotations
    for i, (metric, value) in enumerate(zip(metrics, values)):
        axes[1, 1].text(i, value + max(values) * 0.01, f'{value:,}', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('jax_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Plots saved as 'jax_simulation_results.png'")

def demonstrate_advanced_features(simulator):
    """Demonstrate advanced features of the simulator."""
    
    print("   Advanced Features:")
    
    # Get results for specific sensor
    sensor_results = simulator.get_sensor_results(sensor_idx=0)
    print(f"   - Sensor results keys: {list(sensor_results.keys())}")
    
    # Get results for specific charge state
    state_results = simulator.get_charge_state_results(state_idx=0)
    print(f"   - State results keys: {list(state_results.keys())}")
    
    # Calculate fidelity with different methods
    fidelity_separation = simulator.calculate_fidelity(sensor_idx=0, method='iq_separation')
    fidelity_overlap = simulator.calculate_fidelity(sensor_idx=0, method='overlap')
    
    print(f"   - Fidelity (separation method): {fidelity_separation:.3f}")
    print(f"   - Fidelity (overlap method): {fidelity_overlap:.3f}")
    
    # Show data structure
    print(f"   - Results structure verified successfully")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Example completed successfully!")
            exit(0)
        else:
            print("\n❌ Example failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 