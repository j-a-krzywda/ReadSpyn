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
    

class OU_noise():
    def __init__(self, sigma: float, gamma: float):
        self.sigma = sigma
        self.gamma = gamma
        self.dt = 1e-3  # Time step
        self.current_value = 0.0

    def update(self):
        """Updates the noise value using the Ornstein-Uhlenbeck process."""
        self.current_value += -self.gamma * self.current_value * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        return self.current_value