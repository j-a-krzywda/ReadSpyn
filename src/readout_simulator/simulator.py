"""
Readout Simulator Module

This module provides the main ReadoutSimulator class for simulating quantum dot
readout systems with multiple sensors and noise models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .sensor_backend import RLC_sensor
from .quantum_dot_system import QuantumDotSystem


class ReadoutSimulator:
    """
    Manages the readout simulation for a quantum dot system.
    
    This class orchestrates simulations across multiple sensors, handling
    charge state evolution, noise generation, and signal processing.
    
    Attributes:
        dot_system (QuantumDotSystem): The quantum dot system to simulate
        sensors (List[RLC_sensor]): List of sensor objects
        results (List[Dict]): Simulation results for each sensor and charge state
    """
    
    def __init__(self, dot_system: QuantumDotSystem, sensors: Optional[List[RLC_sensor]] = None):
        """
        Initialize the ReadoutSimulator.
        
        Args:
            dot_system: The quantum dot system to simulate
            sensors: List of sensor objects. If None, default sensors will be created
        """
        if sensors and len(sensors) != dot_system.num_sensors:
            raise ValueError(f"Number of provided sensors ({len(sensors)}) must match "
                           f"the Cds matrix dimension ({dot_system.num_sensors}).")
        
        self.dot_system = dot_system
        self.sensors = sensors or []
        self.results = []
        
        # Validate sensor configuration
        if not self.sensors:
            raise ValueError("At least one sensor must be provided.")

    def run_simulation(self, charge_states: List[np.ndarray], t_end: float, params: Dict[str, Any]):
        """
        Run the simulation for a list of charge states across all sensors.
        
        Args:
            charge_states: List of charge state arrays to simulate
            t_end: End time in units of sensor oscillation periods
            params: Dictionary containing simulation parameters
                   - SNR_white: White noise signal-to-noise ratio
                   - eps0: Nominal position on Coulomb peak in units of width
        """
        self.results = []
        
        # Loop over sensors
        for sensor_index, sensor in enumerate(self.sensors):
            dt = 0.5e-9  # Time step in seconds
            times = np.arange(0, t_end * sensor.T0, dt)
            num_points = len(times)
            
            # Calculate conductance function
            def conductance_fun(eps):
                return np.cosh(eps / sensor.eps_w)**(-2) / sensor.R0
            
            # Calculate energy offsets for each charge state
            energy_offsets = [
                self.dot_system.get_energy_offset(cs, np.zeros(self.dot_system.num_sensors), sensor.eps0)[sensor_index] 
                for cs in charge_states
            ]
            
            # Calculate conductance values
            g_values = [conductance_fun(eo) for eo in energy_offsets]
   
            
            # Adjust SNR based on conductance variation
            params['SNR_eff'] = params['SNR_white'] * (np.max(g_values) - np.min(g_values)) / np.mean(g_values)

       
            
            
            # Get unique charge states for labeling
            unique_states = np.unique(charge_states, axis=0)
            
            # Optional conductance plotting
            if params.get('plot_conductance', False):
                self._plot_conductance(sensor, energy_offsets, conductance_fun)
            

           
            # Simulate each charge state
            for i, charge_state in enumerate(tqdm(charge_states, desc=f"Sensor {sensor_index}")):
                # Generate noise trajectory
                 #faster sim
                noise_trajectory = sensor.eps_noise_model.generate_trajectory(num_points) 
                # Get sensor signal
                I, Q, V_refl_t, times = sensor.get_signal(
                    times, self.dot_system, charge_state, sensor_index, params, noise_trajectory
                )
                
                # Store results
                charge_state_id = np.where(np.all(unique_states == charge_state, axis=1))[0][0]
                self.results.append({
                    'sensor_index': sensor_index,
                    'charge_state_id': charge_state_id,
                    'charge_state': charge_state,
                    'I': I, 
                    'Q': Q,
                    'raw_signal': np.array([V_refl_t]),
                    'times': times,
                    'noise_trajectory': noise_trajectory,
                    'energy_offset': energy_offsets[i]
                })

    def _plot_conductance(self, sensor, energy_offsets, conductance_fun):
        """Helper method to plot conductance characteristics."""
        epses = sensor.eps_w * np.linspace(0, 2, 51)
        plt.figure(figsize=(8, 6))
        plt.plot(epses, conductance_fun(epses), label='Conductance')
        plt.vlines(x=sensor.eps0, ymin=0, ymax=conductance_fun(sensor.eps0), 
                  label='Operating Point', linestyle='--', color='black')
        
        # Plot energy offsets for different charge states
        colors = ['green', 'red']
        for i, eo in enumerate(energy_offsets[:2]):  # Show first two states
            plt.vlines(x=eo, ymin=0, ymax=conductance_fun(eo), 
                      label=f'Charge State {i}', linestyle='--', color=colors[i])
        
        plt.xlabel('Energy Offset (eV)')
        plt.ylabel('Conductance (S)')
        plt.title('Sensor Conductance Characteristics')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_results(self):
        """Plot the integrated IQ results for each sensor."""
        if not self.results:
            print("No simulation results to plot. Run simulation first.")
            return
            
        colors = ["red", "green", "blue", "orange", "purple"]
        
        for sensor_index in range(len(self.sensors)):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for result in self.results:
                if result['sensor_index'] == sensor_index:
                    I_int = np.cumsum(result['I']) / np.arange(1, len(result['I']) + 1)
                    Q_int = np.cumsum(result['Q']) / np.arange(1, len(result['Q']) + 1)
                    color = colors[result['charge_state_id'] % len(colors)]
                    
                    ax.scatter(I_int[-1], Q_int[-1], c=color, 
                             label=f"Charge State {result['charge_state_id']}")
            
            ax.set_xlabel("Integrated I")
            ax.set_ylabel("Integrated Q")
            ax.set_title(f"IQ Readout Results for Sensor {sensor_index}")
            ax.axis('equal')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def get_int_IQ(self):
        """
        Get integrated IQ data for all sensors and charge states.
        
        Returns:
            tuple: (integrated_IQ_data, times)
                   - integrated_IQ_data: Array of dictionaries with 'I' and 'Q' keys
                   - times: Time array
        """
        if not self.results:
            return np.array([]), np.array([])
            
        int_IQ_sensors = []
        for result in self.results:
            I_int = np.cumsum(result['I']) / np.arange(1, len(result['I']) + 1)
            Q_int = np.cumsum(result['Q']) / np.arange(1, len(result['Q']) + 1)
            int_IQ_sensors.append({"I": I_int, "Q": Q_int})
        
        times = self.results[0]['times'] if self.results else np.array([])
        return np.array(int_IQ_sensors), times

    def get_raw_signal(self):
        """
        Get raw signal data for all simulations.
        
        Returns:
            np.ndarray: Array of raw signal data
        """
        if not self.results:
            return np.array([])
            
        raw_signals = [result['raw_signal'] for result in self.results]
        return np.array(raw_signals)
    
    def get_sensor_results(self, sensor_index: int):
        """
        Get results for a specific sensor.
        
        Args:
            sensor_index: Index of the sensor
            
        Returns:
            List[Dict]: Results for the specified sensor
        """
        return [result for result in self.results if result['sensor_index'] == sensor_index]
    
    def get_charge_state_results(self, charge_state_id: int):
        """
        Get results for a specific charge state.
        
        Args:
            charge_state_id: ID of the charge state
            
        Returns:
            List[Dict]: Results for the specified charge state
        """
        return [result for result in self.results if result['charge_state_id'] == charge_state_id]
    
    def calculate_fidelity(self, sensor_index: int = 0):
        """
        Calculate readout fidelity for a specific sensor.
        
        Args:
            sensor_index: Index of the sensor to analyze
            
        Returns:
            float: Readout fidelity (0-1)
        """
        sensor_results = self.get_sensor_results(sensor_index)
        if not sensor_results:
            return 0.0
            
        # Group by charge state
        charge_states = {}
        for result in sensor_results:
            cs_id = result['charge_state_id']
            if cs_id not in charge_states:
                charge_states[cs_id] = []
            charge_states[cs_id].append(result)
        
        # Calculate fidelity based on IQ separation
        if len(charge_states) < 2:
            return 0.0
            
        # Use final integrated values
        centroids = []
        for cs_id, results in charge_states.items():
            I_final = np.mean([result['I'][-1] for result in results])
            Q_final = np.mean([result['Q'][-1] for result in results])
            centroids.append(np.array([I_final, Q_final]))
        

        # TODO: PERFORM ANALYSIS
        # Calculate separation and overlap
        if len(centroids) == 2:
            separation = np.linalg.norm(centroids[0] - centroids[1])
            # Simple fidelity estimate based on separation
            # This could be improved with more sophisticated analysis
            return min(1.0, separation / 10.0)  # Normalized by typical scale
        
        return 0.0