#!/usr/bin/env python3
"""
Simple Two-Sensor Example

A simplified example showing a quantum dot system with 2 dots and 2 sensors.
This example focuses on the essential concepts without complex analysis.

Author: Jan A. Krzywda
Email: j.a.krzywda@liacs.leidenuniv.nl
"""

import numpy as np
import matplotlib.pyplot as plt

# Import ReadSpyn components
from readout_simulator import (
    QuantumDotSystem, 
    RLC_sensor, 
    ReadoutSimulator,
    OverFNoise, 
    OU_noise
)

def main():
    """Simple two-sensor simulation."""
    
    print("=== ReadSpyn Simple Two-Sensor Example ===\n")
    
    # 1. Create quantum dot system: 2 dots, 2 sensors
    print("1. Creating quantum dot system...")
    
    # Dot-dot capacitance matrix
    Cdd = np.array([
        [1.0, 0.2],  # Dot 1: self-capacitance and coupling to dot 2
        [0.2, 1.0]   # Dot 2: self-capacitance and coupling to dot 1
    ])
    
    # Dot-sensor capacitance matrix
    Cds = np.array([
        [0.8, 0.1],  # Dot 1: coupling to sensor 1 and sensor 2
        [0.1, 0.8]   # Dot 2: coupling to sensor 1 and sensor 2
    ]) * 0.5
    
    dot_system = QuantumDotSystem(Cdd, Cds)
    
    # 2. Define charge states
    print("2. Defining charge states...")
    
    # Four possible charge configurations
    charge_states = [
        np.array([0, 0]),  # Both dots empty
        np.array([1, 0]),  # Only dot 1 occupied
        np.array([0, 1]),  # Only dot 2 occupied
        np.array([1, 1])   # Both dots occupied
    ] * 20  # 20 samples per configuration
    
    print(f"   Total charge states: {len(charge_states)}")
    print(f"   Charge configurations: {len(charge_states)//20}")
    
    # 3. Create two different sensors
    print("3. Creating two sensors...")
    
    # Sensor 1: Higher frequency resonator
    sensor_1 = RLC_sensor(
        params_resonator={
            'Lc': 500e-9,      # Lower inductance = higher frequency
            'Cp': 0.3e-12,     # Lower capacitance
            'RL': 35,           # Load resistance
            'Rc': 100e6,       # Coupling resistance
            'Z0': 50
        },
        params_coulomb_peak={
            'g0': 1/50/1e6,    # Conductance
            'eps0': 0.5,       # Operating point
            'eps_width': 1
        },
        c_noise_model=OU_noise(sigma=1e-13, gamma=1e5),
        eps_noise_model=OverFNoise(
            n_fluctuators=5, s1=1e-3, sigma_couplings=1e-99,
            ommax=1, ommin=0.2, dt=1, equally_dist=True
        )
    )
    
    # Sensor 2: Lower frequency resonator
    sensor_2 = RLC_sensor(
        params_resonator={
            'Lc': 1000e-9,     # Higher inductance = lower frequency
            'Cp': 0.6e-12,     # Higher capacitance
            'RL': 45,           # Load resistance
            'Rc': 120e6,       # Coupling resistance
            'Z0': 50
        },
        params_coulomb_peak={
            'g0': 1/60/1e6,    # Conductance
            'eps0': 0.5,       # Operating point
            'eps_width': 1
        },
        c_noise_model=OU_noise(sigma=5e-14, gamma=1e5),
        eps_noise_model=OverFNoise(
            n_fluctuators=3, s1=5e-4, sigma_couplings=1e-99,
            ommax=1, ommin=0.2, dt=1, equally_dist=True
        )
    )
    
    # 4. Create simulator
    print("4. Creating simulator...")
    simulator = ReadoutSimulator(dot_system, [sensor_1, sensor_2])
    
    # 5. Run simulation
    print("5. Running simulation...")
    params = {'SNR_white': 1e12, 'eps0': 0.5}
    nperiods = 1000
    simulator.run_simulation(charge_states, nperiods, params)  # 1000 periods
    
    # 6. Get results
    print("6. Extracting results...")
    IQ_data, times = simulator.get_int_IQ()
    
    # 7. Calculate fidelity for each sensor
    print("7. Calculating performance...")
    fidelity_1 = simulator.calculate_fidelity(sensor_index=0)
    fidelity_2 = simulator.calculate_fidelity(sensor_index=1)
    
    print(f"\nResults:")
    print(f"  Sensor 1 fidelity: {fidelity_1:.3f}")
    print(f"  Sensor 2 fidelity: {fidelity_2:.3f}")
    print(f"  Simulation time: {times[-1]:.2e} seconds")
    
    # 8. Create simple visualization
    print("8. Creating visualization...")
    plot_simple_results(IQ_data, times, fidelity_1, fidelity_2)
    
    print("\n=== Example completed! ===")

def plot_simple_results(IQ_data, times, fidelity_1, fidelity_2):
    """Create simple visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Two-Sensor System Results', fontsize=14, fontweight='bold')
    
    # 1. Sensor 1 IQ Plot
    ax = axes[0, 0]
    plot_sensor_iq_simple(ax, IQ_data, 0, 'Sensor 1')
    
    # 2. Sensor 2 IQ Plot
    ax = axes[0, 1]
    plot_sensor_iq_simple(ax, IQ_data, 1, 'Sensor 2')
    
    # 3. Fidelity Comparison
    ax = axes[1, 0]
    fidelities = [fidelity_1, fidelity_2]
    labels = ['Sensor 1', 'Sensor 2']
    colors = ['blue', 'red']
    bars = ax.bar(labels, fidelities, color=colors, alpha=0.7)
    ax.set_ylabel('Readout Fidelity')
    ax.set_title('Performance Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, fidelity in zip(bars, fidelities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{fidelity:.3f}', ha='center', va='bottom')
    
    # 4. Time evolution (simplified)
    ax = axes[1, 1]
    # Plot a few trajectories for sensor 1
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(min(8, len(IQ_data))):
        if i < len(IQ_data):
            data = IQ_data[i]
            color = colors[i % 4]
            ax.plot(times, data['I'], color=color, alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('I Component')
    ax.set_title('Time Evolution (Sensor 1)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_sensor_iq_simple(ax, IQ_data, sensor_idx, title):
    """Plot simple IQ data for a sensor."""
    
    # Get data for this sensor (assuming alternating sensors)
    sensor_data = []
    for i in range(len(IQ_data)):
        if i % 2 == sensor_idx:
            sensor_data.append(IQ_data[i])
    
    if sensor_data:
        # Plot final I and Q values
        I_final = [data['I'][-1] for data in sensor_data]
        Q_final = [data['Q'][-1] for data in sensor_data]
        
        # Color by charge state (assuming 4 configurations)
        colors = ['red', 'blue', 'green', 'orange']
        samples_per_config = len(sensor_data) // 4
        
        for config in range(4):
            start_idx = config * samples_per_config
            end_idx = start_idx + samples_per_config
            
            if start_idx < len(I_final):
                I_vals = I_final[start_idx:end_idx]
                Q_vals = Q_final[start_idx:end_idx]
                
                ax.scatter(I_vals, Q_vals, color=colors[config], alpha=0.6, s=20)
    
    ax.set_xlabel('I Component')
    ax.set_ylabel('Q Component')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    main() 