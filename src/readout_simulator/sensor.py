# src/readout_simulator/sensor.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from numba import njit
from quantum_dot_system import QuantumDotSystem

@njit
def conductance_fun_numba(eps, w0, R0_val):
    return 2 * np.cosh(2 * eps / w0)**(-2) / R0_val

@njit
def rlc_ode_system_numba(t, y, Lc, Cp, RL_eff, Rc, eps_interp_val, V_s_source_val, eps_w, R0):
    v_Cp, i_L = y
    G_s_t = conductance_fun_numba(eps_interp_val, eps_w, R0)
    
    v_A = (1 / (1 + Rc * G_s_t)) * (Rc * i_L + v_Cp) if np.abs(1 + Rc * G_s_t) > 1e-9 else 0
    dv_Cp_dt = G_s_t * (v_A - v_Cp)
    
    di_L_dt = (V_s_source_val - RL_eff * i_L - v_A) / Lc
    
    return np.array([dv_Cp_dt, di_L_dt])

class Resonator:
    """Represents a resonator sensor with an RLC circuit."""
    def __init__(self, Lc: float = 800e-9, Cp: float = 0.6e-12, RL: float = 40, 
                 Rc: float = 100e6, Z0: float = 50.0, R0: float = 50e3, 
                 eps_w: float = 500e-6, self_capacitance: float = 1e-10):
        self.Lc, self.Cp, self.RL, self.Rc, self.Z0, self.R0, self.eps_w = Lc, Cp, RL, Rc, Z0, R0, eps_w
        self.self_capacitance = self_capacitance
        self.omega0 = 1 / np.sqrt(self.Lc * self.Cp)
        self.f0 = self.omega0 / (2 * np.pi)
        self.T0 = 1 / self.f0

    def get_iq_signal(self, times: np.ndarray, dot_system: QuantumDotSystem, 
                      charge_state: np.ndarray, sensor_index: int, params: dict, 
                      noise_trajectory: np.ndarray = None):
        """
        Simulates the IQ signal for a given charge state and noise trajectory.
        """
        eps0 = params.get('eps0', 0.0) * self.eps_w
        sensor_voltages = np.zeros(dot_system.Cds.shape[1])
        energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, eps0)[sensor_index]
        
        SNR_white = params.get('SNR_white', 1.0)
        SNR_eff = params.get('SNR_eff', SNR_white)
        t_end = times[-1]
        
        eps_values = (noise_trajectory if noise_trajectory is not None else 0) + energy_offset

        eps_interpolator = interp1d(times, eps_values, kind='cubic', fill_value="extrapolate", bounds_error=False)
        
        y0, RL_effective = [0.0, 0.0], self.RL + self.Z0
        
        def v_s_source_func(t):
            return 1.0 * np.sin(self.omega0 * t)

        def ode_wrapper(t, y):
            eps_val = eps_interpolator(t)
            v_s_val = v_s_source_func(t)
            return rlc_ode_system_numba(t, y, self.Lc, self.Cp, RL_effective, self.Rc, eps_val, v_s_val, self.eps_w, self.R0)

        sol = solve_ivp(ode_wrapper, [0, t_end], y0, t_eval=times, method='Radau', rtol=1e-4, atol=1e-7)

        i_L_sim = sol.y[1, :]
        V_s_t = v_s_source_func(sol.t)
        V_refl_t = V_s_t - self.Z0 * i_L_sim - (V_s_t / 2.0)
        V_refl_phasor = hilbert(V_refl_t) * np.exp(-1j * self.omega0 * sol.t)

        amp = np.std(V_refl_phasor) / SNR_eff
        I = np.real(V_refl_phasor) + np.random.normal(0, amp, len(V_refl_phasor))
        Q = np.imag(V_refl_phasor) + np.random.normal(0, amp, len(V_refl_phasor))

        return I, Q