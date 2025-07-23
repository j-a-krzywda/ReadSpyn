
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from numba import njit
from quantum_dot_system import QuantumDotSystem
from noise import OU_noise

# Numba-jitted functions remain the same
@njit
def conductance_fun_numba(eps, w0, R0_val):
    return 2 * np.cosh(2 * eps / w0)**(-2) / R0_val

@njit
def rlc_ode_system_numba(t, y, Lc, Cp_t, RL_eff, Rc, eps_val, V_s_val, w0, R0):
    v_Cp, i_L = y
    G_s_t = conductance_fun_numba(eps_val, w0, R0)

    if np.abs(Rc) < 1e-9:
        v_A = v_Cp 
        dv_Cp_dt = (i_L - v_Cp * G_s_t) / Cp_t
    else:
        v_A = (1 / (1 + Rc * G_s_t)) * (Rc * i_L + v_Cp)
        dv_Cp_dt = (v_A - v_Cp) / (Rc * Cp_t)
    di_L_dt = (V_s_val - RL_eff * i_L - v_A) / Lc
    
    return np.array([dv_Cp_dt, di_L_dt])

class RLC_sensor:
    """Represents a resonator sensor with an RLC circuit."""
    def __init__(self, params_resonator: dict, params_coulomb_peak: dict,
                 c_noise_model: OU_noise = None, eps_noise_model: OU_noise = None):
        """
        Initialize the RLC_sensor with parameter dictionaries.
        """
        # Extract resonator parameters
        self.Lc = params_resonator.get('Lc', 800e-9)
        self.Cp = params_resonator.get('Cp', 0.6e-12)
        self.RL = params_resonator.get('RL', 40)
        self.Rc = params_resonator.get('Rc', 100e6)
        self.Z0 = params_resonator.get('Z0', 50.0)
        self.self_capacitance = params_resonator.get('self_capacitance', 0)
        self.C_total = self.Cp + self.self_capacitance

        # Extract Coulomb peak parameters
        self.R0 = 1 / params_coulomb_peak.get('g0', 1/50*1e6)
        self.eps_w = params_coulomb_peak.get('eps_width', 500e-6)
        self.eps0 = params_coulomb_peak.get('eps0', 0.0) * self.eps_w

        # Resonant frequency calculations
        self.omega0 = 1 / np.sqrt(self.Lc * self.C_total)
        self.f0 = self.omega0 / (2 * np.pi)
        self.T0 = 1 / self.f0

        self.C_noise_model = c_noise_model
        self.eps_noise_model = eps_noise_model

        print(f"[RLC_sensor] Initialized with:")
        print(f"  Lc = {self.Lc} H")
        print(f"  Cp = {self.Cp} F")
        print(f"  Self-capacitance = {self.self_capacitance} F")
        print(f"  Total capacitance (C_total) = {self.C_total} F")
        print(f"  RL = {self.RL} Ω")
        print(f"  Rc = {self.Rc} Ω")
        print(f"  Z0 = {self.Z0} Ω")
        print(f"  R0 = {self.R0} Ω")
        print(f"  g0 = {1/self.R0} S")
        print(f"  eps_w = {self.eps_w} eV")
        print(f"  Resonant frequency (f0) = {self.f0:.3e} Hz")
        print(f"  Resonant period (T0) = {self.T0:.3e} s")
        if self.C_noise_model:
            print(f"  Capacitance noise model: {type(self.C_noise_model).__name__}")
        else:
            print("  Capacitance noise model: None") #NOTE: what is R0
        

    def get_signal(self, times: np.ndarray, dot_system: QuantumDotSystem,
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

        C_noise = np.zeros_like(times)
        if self.C_noise_model:
            for i in range(len(times)):
                C_noise[i] = self.C_noise_model.update(times[i] - (times[i-1] if i > 0 else 0))

        def ode_wrapper(t, y):
            eps_val = eps_interpolator(t)
            v_s_val = v_s_source_func(t)
            
            # --- CORRECTED ---
            # The noisy capacitance is now the total capacitance plus the noise term.
            C_noisy = self.C_total + C_noise[np.argmin(np.abs(times - t))]
            
            return rlc_ode_system_numba(t, y, self.Lc, C_noisy, RL_effective, self.Rc, eps_val, v_s_val, self.eps_w, self.R0)

        sol = solve_ivp(ode_wrapper, [0, t_end], y0, t_eval=times, method='Radau', rtol=1e-3, atol=1e-4)

        i_L_sim = sol.y[1, :]
        V_s_t = v_s_source_func(sol.t)
        amp = np.std(V_s_t) / np.sqrt(SNR_eff)
        V_refl_t = V_s_t - self.Z0 * i_L_sim - (V_s_t / 2.0) + np.random.normal(0, 1, len(V_s_t)) * amp
        V_refl_phasor = hilbert(V_refl_t) * np.exp(-1j * self.omega0 * sol.t)

        I = np.real(V_refl_phasor) 
        Q = np.imag(V_refl_phasor)

        return I, Q, V_refl_t, times