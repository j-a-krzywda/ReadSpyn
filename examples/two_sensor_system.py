#!/usr/bin/env python3
"""
Two-Sensor Quantum Dot System Example

This example demonstrates a quantum dot system with 2 dots and 2 sensors,
showing how to configure and analyze multi-sensor readout systems.

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
    """Main simulation function for two-sensor system."""
    
    # Set plotting style
    rcParams['figure.figsize'] = (15, 10)
    rcParams['font.size'] = 12
    
    print("=== ReadSpyn Two-Sensor System Example ===\n")
    
    # 1. Define Quantum Dot System with 2 dots and 2 sensors
    print("1. Creating quantum dot system with 2 dots and 2 sensors...")
    
    # Dot-dot capacitance matrix (2x2 for 2 dots)
    # Diagonal: self-capacitances, Off-diagonal: mutual capacitances
    Cdd = np.array([
        [1.0, 0.3],  # Dot 1 self-capacitance and coupling to dot 2
        [0.3, 1.2]   # Dot 2 self-capacitance and coupling to dot 1
    ])
    
    # Dot-sensor capacitance matrix (2x2 for 2 dots, 2 sensors)
    # Each row represents a dot, each column represents a sensor
    Cds = np.array([
        [0.8, 0.2],  # Dot 1 coupling to sensor 1 and sensor 2
        [0.1, 0.9]   # Dot 2 coupling to sensor 1 and sensor 2
    ]) * 0.6  # Scale factor
    
    dot_system = QuantumDotSystem(Cdd, Cds)
    print(f"Created system: {dot_system}")
    
    # 2. Configure Simulation Parameters
    print("\n2. Configuring simulation parameters...")
    params = {
        'SNR_white': 1e14,
        'eps0': 0.5,
        'plot_conductance': True
    }
    
    # Simulation settings
    nT_end = 1500  # Number of oscillation periods
    samples = 30    # Number of samples per charge state
    
    # Define charge states: [dot1, dot2]
    charge_states = [
        np.array([0, 0]),  # Both dots empty
        np.array([1, 0]),  # Only dot 1 occupied
        np.array([0, 1]),  # Only dot 2 occupied
        np.array([1, 1])   # Both dots occupied
    ] * samples
    
    print(f"Simulation parameters:")
    print(f"  SNR: {params['SNR_white']:.2e}")
    print(f"  Operating point: {params['eps0']}")
    print(f"  Duration: {nT_end} periods")
    print(f"  Total charge states: {len(charge_states)}")
    print(f"  Charge configurations: {len(charge_states)//samples}")
    
    # 3. Configure Two Different Resonators
    print("\n3. Configuring two RLC resonators with different parameters...")
    
    # Resonator 1: Higher frequency, lower Q
    params_resonator_1 = {
        'Lc': 600e-9,      # Lower inductance = higher frequency
        'Cp': 0.4e-12,     # Lower capacitance
        'RL': 30,           # Lower resistance = lower Q
        'Rc': 80e6,        # Lower coupling resistance
        'Z0': 50
    }
    
    # Resonator 2: Lower frequency, higher Q
    params_resonator_2 = {
        'Lc': 1000e-9,     # Higher inductance = lower frequency
        'Cp': 0.6e-12,     # Higher capacitance
        'RL': 50,           # Higher resistance = higher Q
        'Rc': 120e6,       # Higher coupling resistance
        'Z0': 50
    }
    
    # Coulomb peak parameters (can be different for each sensor)
    params_coulomb_peak_1 = {
        'g0': 1/40/1e6,    # Higher conductance
        'eps0': 0.4,       # Different operating point
        'eps_width': 1
    }
    
    params_coulomb_peak_2 = {
        'g0': 1/60/1e6,    # Lower conductance
        'eps0': 0.6,       # Different operating point
        'eps_width': 1
    }
    
    print("Resonator configurations:")
    print(f"  Resonator 1: L={params_resonator_1['Lc']:.1e}H, C={params_resonator_1['Cp']:.1e}F, RL={params_resonator_1['RL']}Ω")
    print(f"  Resonator 2: L={params_resonator_2['Lc']:.1e}H, C={params_resonator_2['Cp']:.1e}F, RL={params_resonator_2['RL']}Ω")
    
    # 4. Create Different Noise Models for Each Sensor
    print("\n4. Creating noise models for each sensor...")
    
    # Sensor 1: Higher noise (more fluctuators)
    eps_noise_1 = OverFNoise(
        n_fluctuators=8, s1=2e-3, sigma_couplings=1e-99,
        ommax=1, ommin=0.2, dt=1, equally_dist=True
    )
    
    c_noise_1 = OU_noise(sigma=2e-13, gamma=1e5)
    
    # Sensor 2: Lower noise (fewer fluctuators)
    eps_noise_2 = OverFNoise(
        n_fluctuators=3, s1=5e-4, sigma_couplings=1e-99,
        ommax=1, ommin=0.2, dt=1, equally_dist=True
    )
    
    c_noise_2 = OU_noise(sigma=5e-14, gamma=1e5)
    
    print("Noise configurations:")
    print(f"  Sensor 1: {eps_noise_1.noise_generator.n_telegraphs} fluctuators, σ={c_noise_1.sigma:.1e}")
    print(f"  Sensor 2: {eps_noise_2.noise_generator.n_telegraphs} fluctuators, σ={c_noise_2.sigma:.1e}")
    
    # 5. Create Two Sensors
    print("\n5. Creating two RLC sensors...")
    
    sensor_1 = RLC_sensor(
        params_resonator_1, 
        params_coulomb_peak_1, 
        c_noise_1, 
        eps_noise_1
    )
    
    sensor_2 = RLC_sensor(
        params_resonator_2, 
        params_coulomb_peak_2, 
        c_noise_2, 
        eps_noise_2
    )
    
    # 6. Create Simulator with Both Sensors
    print("\n6. Creating simulator with two sensors...")
    simulator = ReadoutSimulator(dot_system, [sensor_1, sensor_2])
    
    # 7. Run Simulation
    print("\n7. Running simulation...")
    simulator.run_simulation(charge_states, nT_end, params)
    
    # 8. Extract Results
    print("\n8. Extracting results...")
    IQ_data, times = simulator.get_int_IQ()
    
    # Handle raw signals carefully for multi-sensor systems
    try:
        raw_signals = simulator.get_raw_signal()
        print(f"  Raw signals extracted successfully")
    except Exception as e:
        print(f"  Note: Raw signal extraction had issues: {e}")
        raw_signals = None
    
    print(f"Results extracted:")
    print(f"  Total data points: {len(IQ_data)}")
    print(f"  Time array length: {len(times)}")
    print(f"  Time range: {times[0]:.2e} to {times[-1]:.2e} seconds")
    
    # 9. Analyze Results for Each Sensor
    print("\n9. Analyzing results for each sensor...")
    
    # Calculate fidelity for each sensor
    fidelity_1 = simulator.calculate_fidelity(sensor_index=0)
    fidelity_2 = simulator.calculate_fidelity(sensor_index=1)
    
    print(f"Performance Analysis:")
    print(f"  Sensor 1 fidelity: {fidelity_1:.3f}")
    print(f"  Sensor 2 fidelity: {fidelity_2:.3f}")
    
    # 10. Visualize Results
    print("\n10. Creating comprehensive visualizations...")
    plot_two_sensor_results(IQ_data, times, fidelity_1, fidelity_2, charge_states, samples)
    
    # 11. Advanced Analysis
    print("\n11. Performing advanced analysis...")
    analyze_sensor_comparison(IQ_data, times, charge_states, samples)
    
    print("\n=== Two-Sensor Example Completed Successfully! ===")

def plot_two_sensor_results(IQ_data, times, fidelity_1, fidelity_2, charge_states, samples):
    """Create comprehensive visualization for two-sensor system."""
    
    # Separate data by sensor and charge state
    num_charge_configs = len(charge_states) // samples
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('ReadSpyn Two-Sensor System Results', fontsize=16, fontweight='bold')
    
    # 1. Sensor 1 IQ Plot
    ax = axes[0, 0]
    plot_sensor_iq(ax, IQ_data, 0, num_charge_configs, samples, 'Sensor 1 IQ Readout')
    
    # 2. Sensor 2 IQ Plot
    ax = axes[0, 1]
    plot_sensor_iq(ax, IQ_data, 1, num_charge_configs, samples, 'Sensor 2 IQ Readout')
    
    # 3. Fidelity Comparison
    ax = axes[0, 2]
    fidelities = [fidelity_1, fidelity_2]
    labels = ['Sensor 1', 'Sensor 2']
    colors = ['blue', 'red']
    bars = ax.bar(labels, fidelities, color=colors, alpha=0.7)
    ax.set_ylabel('Readout Fidelity')
    ax.set_title('Sensor Performance Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, fidelity in zip(bars, fidelities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{fidelity:.3f}', ha='center', va='bottom')
    
    # 4. Time Evolution - Sensor 1 I component
    ax = axes[1, 0]
    plot_sensor_time_evolution(ax, IQ_data, times, 0, 'I', 'Sensor 1 I Component')
    
    # 5. Time Evolution - Sensor 1 Q component
    ax = axes[1, 1]
    plot_sensor_time_evolution(ax, IQ_data, times, 0, 'Q', 'Sensor 1 Q Component')
    
    # 6. Time Evolution - Sensor 2 I component
    ax = axes[1, 2]
    plot_sensor_time_evolution(ax, IQ_data, times, 1, 'I', 'Sensor 2 I Component')
    
    # 7. Time Evolution - Sensor 2 Q component
    ax = axes[2, 0]
    plot_sensor_time_evolution(ax, IQ_data, times, 1, 'Q', 'Sensor 2 Q Component')
    
    # 8. Charge State Separation Analysis
    ax = axes[2, 1]
    plot_charge_state_separation(ax, IQ_data, num_charge_configs, samples)
    
    # 9. Sensor Correlation
    ax = axes[2, 2]
    plot_sensor_correlation(ax, IQ_data, num_charge_configs, samples)
    
    plt.tight_layout()
    plt.show()

def plot_sensor_iq(ax, IQ_data, sensor_index, num_charge_configs, samples, title):
    """Plot IQ data for a specific sensor."""
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']
    
    for config in range(num_charge_configs):
        # Get data for this charge configuration
        start_idx = config * samples
        end_idx = start_idx + samples
        
        # Find data for this sensor
        sensor_data = []
        for i in range(start_idx, end_idx):
            if i < len(IQ_data):
                sensor_data.append(IQ_data[i])
        
        if sensor_data:
            # Calculate mean for this configuration
            I_vals = [data['I'][-1] for data in sensor_data]
            Q_vals = [data['Q'][-1] for data in sensor_data]
            
            # Center around mean
            I_mean = np.mean(I_vals)
            Q_mean = np.mean(Q_vals)
            
            ax.scatter(np.array(I_vals) - I_mean, np.array(Q_vals) - Q_mean, 
                      color=colors[config], alpha=0.6, s=30, label=labels[config])
    
    ax.set_xlabel('I - mean I')
    ax.set_ylabel('Q - mean Q')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

def plot_sensor_time_evolution(ax, IQ_data, times, sensor_index, component, title):
    """Plot time evolution for a specific sensor and component."""
    
    colors = ['red', 'blue', 'green', 'orange']
    Nplot = min(15, len(IQ_data) // 4)  # Plot subset of trajectories
    
    for config in range(4):  # 4 charge configurations
        color = colors[config]
        for i in range(Nplot):
            idx = config * Nplot + i
            if idx < len(IQ_data):
                data = IQ_data[idx]
                if component == 'I':
                    values = data['I']
                else:
                    values = data['Q']
                
                # Center around mean
                mean_val = np.mean(values)
                ax.plot(times, values - mean_val, color=color, alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{component} - mean {component}')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def plot_charge_state_separation(ax, IQ_data, num_charge_configs, samples):
    """Plot charge state separation analysis."""
    
    # Calculate separation between different charge states
    separations = []
    labels = []
    
    for i in range(num_charge_configs):
        for j in range(i+1, num_charge_configs):
            # Get centroids for each charge state
            centroid_i = calculate_centroid(IQ_data, i, samples)
            centroid_j = calculate_centroid(IQ_data, j, samples)
            
            if centroid_i is not None and centroid_j is not None:
                separation = np.linalg.norm(centroid_i - centroid_j)
                separations.append(separation)
                labels.append(f'{i} vs {j}')
    
    if separations:
        bars = ax.bar(range(len(separations)), separations, alpha=0.7)
        ax.set_xlabel('Charge State Pairs')
        ax.set_ylabel('Separation Distance')
        ax.set_title('Charge State Separation')
        ax.set_xticks(range(len(separations)))
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, sep in zip(bars, separations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{sep:.2f}', ha='center', va='bottom')

def plot_sensor_correlation(ax, IQ_data, num_charge_configs, samples):
    """Plot correlation between sensors."""
    
    # Calculate correlation between sensor readings
    correlations = []
    
    for config in range(num_charge_configs):
        # Get data for this charge configuration
        start_idx = config * samples
        end_idx = start_idx + samples
        
        sensor1_data = []
        sensor2_data = []
        
        for i in range(start_idx, end_idx):
            if i < len(IQ_data):
                # For simplicity, assume alternating sensors
                if i % 2 == 0:
                    sensor1_data.append(IQ_data[i])
                else:
                    sensor2_data.append(IQ_data[i])
        
        if sensor1_data and sensor2_data:
            # Calculate correlation (simplified)
            I1_vals = [data['I'][-1] for data in sensor1_data]
            I2_vals = [data['I'][-1] for data in sensor2_data]
            
            if len(I1_vals) == len(I2_vals) and len(I1_vals) > 1:
                corr = np.corrcoef(I1_vals, I2_vals)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
    
    if correlations:
        ax.hist(correlations, bins=10, alpha=0.7, color='purple')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Sensor Correlation Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        ax.legend()

def calculate_centroid(IQ_data, config_idx, samples):
    """Calculate centroid for a specific charge configuration."""
    
    start_idx = config_idx * samples
    end_idx = start_idx + samples
    
    config_data = []
    for i in range(start_idx, end_idx):
        if i < len(IQ_data):
            config_data.append(IQ_data[i])
    
    if config_data:
        I_vals = [data['I'][-1] for data in config_data]
        Q_vals = [data['Q'][-1] for data in config_data]
        
        I_mean = np.mean(I_vals)
        Q_mean = np.mean(Q_vals)
        
        return np.array([I_mean, Q_mean])
    
    return None

def analyze_sensor_comparison(IQ_data, times, charge_states, samples):
    """Perform detailed analysis comparing the two sensors."""
    
    print("\n=== Sensor Comparison Analysis ===")
    
    # Analyze each sensor separately
    for sensor_idx in range(2):
        print(f"\nSensor {sensor_idx + 1} Analysis:")
        
        # Get data for this sensor
        sensor_data = []
        for i in range(len(IQ_data)):
            if i % 2 == sensor_idx:  # Assuming alternating sensors
                sensor_data.append(IQ_data[i])
        
        if sensor_data:
            # Calculate statistics
            I_final = [data['I'][-1] for data in sensor_data]
            Q_final = [data['Q'][-1] for data in sensor_data]
            
            print(f"  Final I range: {np.min(I_final):.3e} to {np.max(I_final):.3e}")
            print(f"  Final Q range: {np.min(Q_final):.3e} to {np.max(Q_final):.3e}")
            print(f"  I standard deviation: {np.std(I_final):.3e}")
            print(f"  Q standard deviation: {np.std(Q_final):.3e}")
    
    # Analyze charge state discrimination
    print(f"\nCharge State Discrimination:")
    unique_states = np.unique(charge_states, axis=0)
    print(f"  Number of unique charge states: {len(unique_states)}")
    
    for i, state in enumerate(unique_states):
        print(f"  State {i}: {state}")
    
    # Time evolution analysis
    print(f"\nTime Evolution Analysis:")
    print(f"  Simulation duration: {times[-1] - times[0]:.3e} seconds")
    print(f"  Number of time points: {len(times)}")
    print(f"  Time step: {times[1] - times[0]:.3e} seconds")

if __name__ == "__main__":
    main() 