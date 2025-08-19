#!/usr/bin/env python3
"""
Basic Simulation Example for ReadSpyn

This script demonstrates the basic usage of the ReadSpyn simulator
for quantum dot readout systems.

Author: Jan A. Krzywda
Email: j.a.krzywda@liacs.leidenuniv.nl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Import ReadSpyn components
from readout_simulator import (
    QuantumDotSystem, 
    RLC_sensor, 
    ReadoutSimulator,
    OverFNoise, 
    OU_noise
)

def main():
    """Main simulation function."""
    
    # Set plotting style
    rcParams['figure.figsize'] = (12, 8)
    rcParams['font.size'] = 12
    
    print("=== ReadSpyn Basic Simulation Example ===\n")
    
    # 1. Define Quantum Dot System
    print("1. Creating quantum dot system...")
    Cdd = np.array([[1, 0], [0, 1]])  # 2x2 dot-dot capacitance matrix
    Cds = np.array([[1], [0.1]]) * 0.6  # 2x1 dot-sensor coupling matrix
    dot_system = QuantumDotSystem(Cdd, Cds)
    
    # 2. Configure Parameters
    print("2. Configuring simulation parameters...")
    params = {
        'SNR_white': 1e12, # TODO: Interpret this parameter
        'eps0': 0.5,
        'plot_conductance': True
    }
    
    # Simulation settings
    nT_end = 1000  # Reduced for faster execution
    samples = 25    # Reduced for faster execution
    charge_states = [np.array([1, 0]), np.array([0, 1])] * samples
    
    # 3. Configure Resonator
    print("3. Configuring RLC resonator...")
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
    
    # 4. Create Noise Models
    print("4. Creating noise models...")
    eps_noise = OverFNoise(
        n_fluctuators=3, s1=1e-3, sigma_couplings=1e-99,
        ommax=1, ommin=0.2, dt=1, equally_dist=True
    )
    
    c_noise = OU_noise(sigma=1e-13, gamma=1e5)
    
    # 5. Create Sensor and Simulator
    print("5. Creating sensor and simulator...")
    sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
    simulator = ReadoutSimulator(dot_system, [sensor])
    
    # 6. Run Simulation
    print("6. Running simulation...")
    simulator.run_simulation(charge_states, nT_end, params)
    
    # 7. Extract Results
    print("7. Extracting results...")
    IQ_data, times = simulator.get_int_IQ()
    fidelity = simulator.calculate_fidelity()
    
    print(f"\nSimulation completed!")
    print(f"Readout fidelity: {fidelity:.3f}")
    
    # 8. Visualize Results
    print("8. Creating visualizations...")
    plot_results(IQ_data, times, fidelity)
    
    print("\n=== Example completed successfully! ===")

def plot_results(IQ_data, times, fidelity):
    """Create visualization plots."""
    
    # Calculate means for each charge state
    state_0_data = [IQ_data[i] for i in range(0, len(IQ_data), 2)]
    state_1_data = [IQ_data[i] for i in range(1, len(IQ_data), 2)]
    
    I_mean_0 = np.mean([data['I'] for data in state_0_data], axis=0)
    Q_mean_0 = np.mean([data['Q'] for data in state_0_data], axis=0)
    I_mean_1 = np.mean([data['I'] for data in state_1_data], axis=0)
    Q_mean_1 = np.mean([data['Q'] for data in state_1_data], axis=0)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ReadSpyn Basic Simulation Results', fontsize=16, fontweight='bold')
    
    # 1. IQ Plot
    ax = axes[0, 0]
    for n, IQ in enumerate(IQ_data):
        if n % 2 == 0:
            ax.scatter(IQ['I'][-1] - I_mean_0[-1], IQ['Q'][-1] - Q_mean_0[-1], 
                       color='blue', alpha=0.6, s=30)
        else:
            ax.scatter(IQ['I'][-1] - I_mean_1[-1], IQ['Q'][-1] - Q_mean_1[-1], 
                       color='red', alpha=0.6, s=30)
    ax.set_xlabel('I - mean I')
    ax.set_ylabel('Q - mean Q')
    ax.set_title('IQ Readout Results')
    ax.grid(True, alpha=0.3)
    ax.legend(['State 0', 'State 1'])
    
    # 2. Time Evolution - I component
    ax = axes[0, 1]
    Nplot = min(10, len(IQ_data))
    colors = ['blue', 'red']
    for k in range(Nplot):
        color = colors[k % 2]
        if k % 2 == 0:
            mean_val = I_mean_0
        else:
            mean_val = I_mean_1
        ax.plot(times, IQ_data[k]['I'] - mean_val, color=color, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('I - I_mean')
    ax.set_title('I Component Time Evolution')
    ax.grid(True, alpha=0.3)
    
    # 3. Time Evolution - Q component
    ax = axes[1, 0]
    for k in range(Nplot):
        color = colors[k % 2]
        if k % 2 == 0:
            mean_val = Q_mean_0
        else:
            mean_val = Q_mean_1
        ax.plot(times, IQ_data[k]['Q'] - mean_val, color=color, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Q - Q_mean')
    ax.set_title('Q Component Time Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Fidelity display
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'Readout Fidelity: {fidelity:.3f}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Summary')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 