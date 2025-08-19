"""
Noise Models Wrapper Module

This module provides wrapper classes for the noise models defined in noise.py,
offering a cleaner interface for the simulator.
"""

import numpy as np
from typing import Optional

from .noise import Over_f_noise, OU_noise


class OverFNoise:
    """
    A wrapper for the 1/f noise model to be used in the simulator.
    
    This class provides a simplified interface for generating 1/f noise
    trajectories with configurable parameters.
    
    Attributes:
        noise_generator: The underlying Over_f_noise instance
        dt: Time step for noise generation
    """
    
    def __init__(self, n_fluctuators: int, s1: float, sigma_couplings: float,
                 ommax: float, ommin: float, dt: float, equally_dist: bool = False):
        """
        Initialize the 1/f noise wrapper.
        
        Args:
            n_fluctuators: Number of fluctuators to use
            s1: 1/f noise amplitude
            sigma_couplings: Coupling strength variation
            ommax: Maximum frequency
            ommin: Minimum frequency
            dt: Time step
            equally_dist: Whether to distribute frequencies equally
        """
        self.noise_generator = Over_f_noise(
            n_fluctuators=n_fluctuators,
            S1=s1,
            sigma_couplings=sigma_couplings,
            ommax=ommax,
            ommin=ommin,
            equally_dist=equally_dist
        )
        self.dt = dt

    def generate_trajectory(self, num_points: int) -> np.ndarray:
        """
        Generate a noise trajectory of a given length.
        
        Args:
            num_points: Number of points in the trajectory
            
        Returns:
            np.ndarray: Noise trajectory array
        """
        return np.array([self.noise_generator.update(self.dt) for _ in range(num_points)])
    
    def reset(self, x0: Optional[float] = None) -> float:
        """
        Reset the noise generator.
        
        Args:
            x0: Initial value (if None, random)
            
        Returns:
            float: Reset noise value
        """
        return self.noise_generator.reset(x0)
    
    def get_current_value(self) -> float:
        """
        Get the current noise value.
        
        Returns:
            float: Current noise value
        """
        return self.noise_generator.x