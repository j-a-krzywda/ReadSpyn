# src/readout_simulator/modules/noise_models.py
import numpy as np
# Assuming the original noise.py is available in the same directory
from noise import Over_f_noise


class OverFNoise:
    """A wrapper for the 1/f noise model to be used in the simulator."""
    def __init__(self, n_fluctuators: int, s1: float, sigma_couplings: float,
                 ommax: float, ommin: float, dt: float):
        self.noise_generator = Over_f_noise(
            n_fluctuators=n_fluctuators,
            S1=s1,
            sigma_couplings=sigma_couplings,
            ommax=ommax,
            ommin=ommin
        )
        self.dt = dt

    def generate_trajectory(self, num_points: int) -> np.ndarray:
        """Generates a noise trajectory of a given length."""
        return np.array([self.noise_generator.update(self.dt) for _ in range(num_points)])