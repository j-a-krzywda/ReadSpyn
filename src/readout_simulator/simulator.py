# src/readout_simulator/simulator.py

import numpy as np
import matplotlib.pyplot as plt
from .sensor import Resonator
from .noise_models import OverFNoise
from .quantum_dot_system import QuantumDotSystem

class ReadoutSimulator:
    """
    Manages the readout simulation for a quantum dot system.
    """
    def __init__(self, dot_system: QuantumDotSystem, sensors: list[Resonator] = None, 
                 noise_model: OverFNoise = None):
        if sensors and len(sensors) != dot_system.num_sensors:
            raise ValueError("Number of provided sensors must match the Cds matrix dimension.")
        
        self.dot_system = dot_system
        self.sensors = sensors or [Resonator() for _ in range(dot_system.num_sensors)]
        
        if noise_model is None:
            default_resonator = self.sensors[0]
            dt = default_resonator.T0 / 5
            self.noise_model = OverFNoise(
                n_fluctuators=8, s1=10 * 1e-6, sigma_couplings=1e-99,
                ommax=1 / dt, ommin=1 / (100 * 25 * default_resonator.T0),
                dt=dt
            )
        else:
            self.noise_model = noise_model
            
        self.results = []

    def run_simulation(self, charge_states: list, t_end: float, params: dict):
        """Runs the simulation for a list of charge states across all sensors."""
        self.results = []
        
        for sensor_index, sensor in enumerate(self.sensors):
            dt = sensor.T0 / 5
            times = np.arange(0, t_end * sensor.T0, dt)
            num_points = len(times)
            
            # Calculate a more physically meaningful SNR
            def conductance_fun(eps):
                return 2 * np.cosh(2 * eps / sensor.eps_w)**(-2) / sensor.R0
            
            energy_offsets = [sensor.get_energy_offset(cs, np.zeros(self.dot_system.num_sensors))[sensor_index] for cs in charge_states]
            g_values = [conductance_fun(eo) for eo in energy_offsets]
            
            params['SNR_eff'] = params['SNR_white'] * np.abs(g_values[0] - g_values[1]) / np.mean(g_values)
            
            for i, charge_state in enumerate(charge_states):
                noise_trajectory = self.noise_model.generate_trajectory(num_points)
                I, Q = sensor.get_iq_signal(times, self.dot_system, charge_state, sensor_index, params, noise_trajectory)
                self.results.append({
                    'sensor_index': sensor_index,
                    'charge_state_id': i,
                    'I': I, 'Q': Q,
                    'times': times
                })

    def plot_results(self):
        """Plots the integrated IQ results for each sensor."""
        for sensor_index in range(len(self.sensors)):
            fig, ax = plt.subplots(figsize=(8, 8))
            for result in self.results:
                if result['sensor_index'] == sensor_index:
                    I_int = np.cumsum(result['I']) / np.arange(1, len(result['I']) + 1)
                    Q_int = np.cumsum(result['Q']) / np.arange(1, len(result['Q']) + 1)
                    color = 'blue' if result['charge_state_id'] == 0 else 'red'
                    ax.scatter(I_int[-1], Q_int[-1], c=color, label=f"Charge State {result['charge_state_id']}")
            
            ax.set_xlabel("Integrated I")
            ax.set_ylabel("Integrated Q")
            ax.set_title(f"IQ Readout Results for Sensor {sensor_index}")
            ax.axis('equal')
            plt.show()