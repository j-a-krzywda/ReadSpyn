import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append('../../noise_sim')
import noise as noise
from numpy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import hilbert # Added for Hilbert transform




# Meaningful parameters



# --- Simulation Parameters ---
# Circuit Components
Lc = 800e-9   # H (Inductor)
Cp = 0.6e-12   # F (Capacitor)
RL = 40   # Ohm (Intrinsic resistance of Lc path)
Rc = 100e6  # Ohm (Resistance in series with Cp)
Z0= 50.0 # Ohm (Source impedance Z0)
R0 = 50 * 1e3 # Resistance at the peak



# --- Noise Parameters ---



eps_gamma = 1e7 # 1/tc of correlated noise where tc is the correlation time
SNR_white = 1 #amplitude of white noise 5-50


# Time parameters
omega0 = 1 / np.sqrt(Lc * Cp) 
f0 = 1.0*omega0 / (2 * np.pi)
T0 = 1/f0 
dt_ou = T0 / 5 # Time step for OU process and ODE  ns)



V_amplitude = 1.0 # V (Amplitude of sine wave source)
V_omega = omega0   
def V_s_source_func(t):
    return V_amplitude * np.sin(omega0 * t)

eps_w = 500      # Width of coulomb peak        #300-500 uV

def conductance_fun(eps, w0=eps_w, R0_val=R0):
    """
    Conductance function based on the given parameters.
    eps: the energy offset
    w0: the width of the Coulomb peak
    R0_val: the resistance at the peak
    """
    return 2 * np.cosh(2 * eps / w0)**(-2) / R0_val



def calculate_zeta_s(omega, R_s, Rc, Cp):
    Y = 0.
    if Rc != 0: Y += 1./Rc
    if R_s != 0: Y += 1./R_s
    Y += 1j * omega * Cp
    zeta = np.ones_like(Y, dtype=complex) * np.inf
    non_zero_Y = (Y != 0)
    zeta[non_zero_Y] = 1.0 / Y[non_zero_Y]
    dc_idx = np.where(omega == 0)[0]
    if len(dc_idx) > 0:
         Y_dc = 0.
         if Rc != 0: Y_dc += 1./Rc
         if R_s != 0: Y_dc += 1./R_s
         if Y_dc != 0:
             zeta[dc_idx] = 1.0 / Y_dc
         else:
             zeta[dc_idx] = np.inf
    return zeta

def calculate_Z_s(omega, R_s, RL, Lc, Rc, Cp):
    zeta = calculate_zeta_s(omega, R_s, Rc, Cp)
    return RL + 1j * omega * Lc + zeta

def calculate_Gamma(omega, Z_s, Z0):
    denom = Z_s + Z0
    gamma = np.ones_like(denom, dtype=complex)
    non_zero_denom = (denom != 0)
    gamma[non_zero_denom] = (Z_s[non_zero_denom] - Z0) / denom[non_zero_denom]
    return gamma

def calculate_chi(omega, R_s_avg, Z0, RL, Lc, Rc, Cp):
    zeta_s_avg = calculate_zeta_s(omega, R_s_avg, Rc, Cp)
    Z_s_avg = RL + 1j * omega * Lc + zeta_s_avg
    numerator = 2 * Z0 * zeta_s_avg**2
    denominator = (Z_s_avg + Z0)**2
    chi = np.zeros_like(denominator, dtype=complex)
    return chi


def rlc_ode_system(t, y, Lc, Cp, RL_eff, Rc, eps_interp_func, V_s_source_func):
    v_Cp, i_L = y
    R_s_t = 1/conductance_fun(eps_interp_func(t))

    if np.abs(Rc + R_s_t) < 1e-9 : # Avoid division by zero if denominator is tiny
         v_A = 0 
    else:
        v_A = (R_s_t / (Rc + R_s_t)) * (Rc * i_L + v_Cp)
    
    if np.abs(Rc) > 1e-9 : # If Rc is significantly non-zero
        dv_Cp_dt = (v_A - v_Cp) / (Rc * Cp)
    else: # Special case: Rc is zero (or effectively zero)
        v_A = v_Cp 
        if np.abs(R_s_t) < 1e-9: # Avoid division by zero if Rs_t is tiny
            dv_Cp_dt = i_L / Cp 
        else:
            dv_Cp_dt = (i_L - v_Cp / R_s_t) / Cp
        
    di_L_dt = (V_s_source_func(t) - RL_eff * i_L - v_A) / Lc
    
    return [dv_Cp_dt, di_L_dt]

from numba import njit

@njit
def conductance_fun_numba(eps, w0=eps_w, R0_val=R0):
    # Numba works best when globals are passed as arguments
    return 2*np.cosh(2*eps/w0)**(-2)/R0_val

@njit
def rlc_ode_system_numba(t, y, Lc, Cp, RL_eff, Rc, eps_interp_val, V_s_source_val):
    # The interpolator object can't be used in Numba,
    # so we pass its evaluated value. This requires a wrapper.
    v_Cp, i_L = y
    
    R_s_t = 1.0 / conductance_fun_numba(eps_interp_val)

    if np.abs(Rc + R_s_t) < 1e-9:
         v_A = 0.0
    else:
        v_A = (R_s_t / (Rc + R_s_t)) * (Rc * i_L + v_Cp)
    
    if np.abs(Rc) > 1e-9:
        dv_Cp_dt = (v_A - v_Cp) / (Rc * Cp)
    else:
        v_A = v_Cp 
        if np.abs(R_s_t) < 1e-9:
            dv_Cp_dt = i_L / Cp 
        else:
            dv_Cp_dt = (i_L - v_Cp / R_s_t) / Cp
        
    di_L_dt = (V_s_source_val - RL_eff * i_L - v_A) / Lc
    
    return np.array([dv_Cp_dt, di_L_dt])

# We need a wrapper function for the solver because it needs to call
# the interpolator and source function at arbitrary times 't'.
def make_ode_wrapper(eps_interpolator):
    def rlc_ode_system_wrapper(t, y, Lc, Cp, RL_eff, Rc, V_s_func):
        # These are called at each step by the solver
        eps_val = eps_interpolator(t)
        v_s_val = V_s_func(t)
        # Call the fast Numba function
        return rlc_ode_system_numba(t, y, Lc, Cp, RL_eff, Rc, eps_val, v_s_val)
    return rlc_ode_system_wrapper



def get_IQ_signal_ODE(times, qubit_state,  params, trajectory = None):
    
    # load params
    eps0 = params['eps0']*eps_w
    deps = params['deps']*eps_w
    t_end = times[-1]

    SNR_eff = params['SNR_white'] * np.abs(conductance_fun(eps0 + qubit_state * deps)) / conductance_fun(eps0) # Effective SNR for the noise process
    

    if trajectory is not None:
        eps_values = np.array(trajectory + eps0 + deps*qubit_state)

    eps_interpolator = interp1d(times, eps_values, kind='cubic', fill_value=(eps_values[0], eps_values[-1]),bounds_error=False)
    y0 = [0.0, 0.0] # Initial conditions: [v_Cp(0), i_L(0)]
    
    RL_effective_in_series = RL + Z0 # Total fixed resistance for the i_L ODE
    
    # Create the fast, wrapped ODE system
    fast_ode_system = make_ode_wrapper(eps_interpolator)
    
    sol = solve_ivp(fast_ode_system, [0, t_end], y0,
                    args=(Lc, Cp, RL_effective_in_series, Rc, V_s_source_func), # Note changed args
                    dense_output=False, t_eval=times, method='Radau', rtol=1e-4, atol=1e-7, max_step=T0/2)

    t_sim = sol.t
    i_L_sim = sol.y[1,:]
    g_sim_at_t_eval = conductance_fun(eps_interpolator(t_sim)) # R_s values at solution time points

    V_s_source_t_vals = V_s_source_func(t_sim) # Source voltage time series
    V_circuit_input_t_vals = V_s_source_t_vals - Z0 * i_L_sim # Voltage at RLC network input

    V_inc_wave_t_vals = V_s_source_t_vals / 2.0 # Incident voltage wave
    V_refl_wave_t_vals = V_circuit_input_t_vals - V_inc_wave_t_vals # Reflected voltage wave


    analytic_V_refl_t = hilbert(V_refl_wave_t_vals) 



    demodulation_signal = np.exp(-1j * V_omega * t_sim)
    V_refl_envelope_phasor_t = analytic_V_refl_t * demodulation_signal
    amp = np.std(V_refl_envelope_phasor_t)/SNR_eff

    I = np.real(V_refl_envelope_phasor_t) + np.random.normal(0, amp, len(analytic_V_refl_t)) 
    Q = np.imag(V_refl_envelope_phasor_t) + np.random.normal(0, amp, len(analytic_V_refl_t)) 
    V_inc_phasor_constant = -1j * V_amplitude / 2.0

    # Instantaneous complex reflection coefficient Gamma(t)
    Gamma_t_instantaneous = V_refl_envelope_phasor_t / V_inc_phasor_constant
    return g_sim_at_t_eval, t_sim, i_L_sim, Gamma_t_instantaneous, I, Q

