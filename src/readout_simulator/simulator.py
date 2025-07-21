# src/readout_simulator/simulator.py

import numpy as np
import matplotlib.pyplot as plt
from sensor_backend import RLC_sensor
from noise_models import OverFNoise
from quantum_dot_system import QuantumDotSystem
from tqdm import tqdm
from noise import OU_noise # Added

class ReadoutSimulator:
    """
    Manages the readout simulation for a quantum dot system.
    """
    def __init__(self, dot_system: QuantumDotSystem, sensors: list[RLC_sensor] = None): # Added L_noise_model


        if sensors and len(sensors) != dot_system.num_sensors:
            raise ValueError("Number of provided sensors must match the Cds matrix dimension.")
        self.dot_system = dot_system
        self.sensors = sensors or [Resonator(C_noise_model=C_noise_model) for _ in range(dot_system.num_sensors)] # create default sensors if none provide TODO: Add_interface to change parameters easy

        self.results = []

    def run_simulation(self, charge_states: list, t_end: float, params: dict):
        """Runs the simulation for a list of charge states across all sensors."""
        self.results = []
        

        # LOOP OVER SENSORS-----------
        for sensor_index, sensor in enumerate(self.sensors):
            dt = sensor.T0 / 5
            times = np.arange(0, t_end * sensor.T0, dt)
            num_points = len(times)
            
            def conductance_fun(eps):
                return np.cosh( eps / sensor.eps_w)**(-2) / sensor.R0
            energy_offsets = [self.dot_system.get_energy_offset(cs, np.zeros(self.dot_system.num_sensors), sensor.eps0)[sensor_index] for cs in charge_states]
            g_values = [conductance_fun(eo) for eo in energy_offsets]
            print("Conductance per state (g/g0) : ", np.array(g_values)*sensor.R0)
            params['SNR_eff'] = params['SNR_white'] * (np.max(g_values) - np.min(g_values))/np.mean(g_values)  # Adjust SNR based on conductance

            unique_states = np.unique(charge_states, axis=0)

            plot_conductance = True
            if plot_conductance:
                epses = sensor.eps_w * np.linspace(0, 2, 51)
                plt.plot(epses, conductance_fun(epses), label='Conductance')
                plt.vlines(x=sensor.eps0, ymin=0, ymax=conductance_fun(sensor.eps0), label='SNR Adjusted Conductance', linestyle='--')
                plt.vlines(x=np.array(energy_offsets[0]), ymin=0, ymax=conductance_fun(sensor.eps0), label='SNR Adjusted Conductance', linestyle='--', color='green')
                plt.vlines(x=np.array(energy_offsets[1]), ymin=0, ymax=conductance_fun(sensor.eps0), label='SNR Adjusted Conductance', linestyle='--', color="r")

            for i, charge_state in enumerate(tqdm(charge_states, desc=f"Sensor {sensor_index}")):
                noise_trajectory = sensor.eps_noise_model.generate_trajectory(num_points)
                I, Q, V_refl_t, times = sensor.get_signal(times, self.dot_system, charge_state, sensor_index, params, noise_trajectory)
                self.results.append({
                    'sensor_index': sensor_index,
                    'charge_state_id': np.where(np.all(unique_states == charge_state, axis=1))[0][0],
                    'I': I, 'Q': Q,
                    "raw_signal": np.array([V_refl_t]),
                    'times': times,
                    'noise_trajectory': noise_trajectory
                })

    def plot_results(self):
        """Plots the integrated IQ results for each sensor."""
        colors = ["r","g","b"]
        for sensor_index in range(len(self.sensors)):
            fig, ax = plt.subplots(figsize=(8, 8))
            for result in self.results:
                if result['sensor_index'] == sensor_index:
                    I_int = np.cumsum(result['I']) / np.arange(1, len(result['I']) + 1)
                    Q_int = np.cumsum(result['Q']) / np.arange(1, len(result['Q']) + 1)
                    color = colors[result['charge_state_id'] ]
                    
                    ax.scatter(I_int[-1], Q_int[-1], c=color, label=f"Charge State {result['charge_state_id']}")
            
            ax.set_xlabel("Integrated I")
            ax.set_ylabel("Integrated Q")
            ax.set_title(f"IQ Readout Results for Sensor {sensor_index}")
            ax.axis('equal')


    def get_int_IQ(self):
        int_IQ_sensors = []
        for sensor_index in range(len(self.sensors)):
            fig, ax = plt.subplots(figsize=(8, 8))
            for result in self.results:
                if result['sensor_index'] == sensor_index:
                    I_int = np.cumsum(result['I']) / np.arange(1, len(result['I']) + 1)
                    Q_int = np.cumsum(result['Q']) / np.arange(1, len(result['Q']) + 1)
                    int_IQ_sensors.append({"I": I_int, "Q": Q_int})
        times = self.results[0]['times'] if self.results else np.array([])
        return np.array(int_IQ_sensors), times


    def get_raw_signal(self):
        raw_signals = []
        for result in self.results:
            raw_signals.append(result['raw_signal'])
        return np.array(raw_signals)