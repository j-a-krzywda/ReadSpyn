"""
Noise Models Module

This module provides various noise models for simulating realistic quantum dot
readout systems, including telegraph noise, Ornstein-Uhlenbeck processes, and 1/f noise.
"""

import numpy as np
from typing import Optional, List, Union


# Conversion constant
ueV_to_MHz = 1e3 / 4


def get_spectrum(signal: np.ndarray, time_step: float, total_time: float) -> tuple:
    """
    Calculate the power spectral density of a signal.
    
    Args:
        signal: Input signal array
        time_step: Time step between samples
        total_time: Total duration of the signal
        
    Returns:
        tuple: (frequencies, power_spectrum)
            - frequencies: Frequency array (Hz)
            - power_spectrum: Power spectral density
    """
    N = len(signal)
    f = np.fft.fftfreq(N, time_step)
    xf = np.fft.fft(signal)
    
    # Calculate power spectral density
    Sxx = 2 * time_step**2 / total_time * (xf * np.conj(xf))
    Sxx = Sxx.real
    
    # Return only positive frequencies
    return f[:int(N/2)], Sxx[:int(N/2)]


class Telegraph_Noise:
    """
    Telegraph noise model for simulating two-level fluctuators.
    
    This class implements a simple two-state noise model where the system
    randomly switches between two values with a given switching rate.
    
    Attributes:
        gamma (float): Switching rate (Hz)
        sigma (float): Noise amplitude
        x (float): Current noise value
        x0 (Optional[float]): Initial value (if specified)
    """
    
    def __init__(self, sigma: float, gamma: float, x0: Optional[float] = None):
        """
        Initialize telegraph noise.
        
        Args:
            sigma: Noise amplitude
            gamma: Switching rate (Hz)
            x0: Initial value (if None, random initial state)
        """
        self.gamma = gamma
        self.sigma = sigma
        self.x0 = x0
        
        if x0 is None:
            self.x = self.sigma * (2 * np.random.randint(0, 2) - 1)
        else:
            self.x = x0

    def update(self, dt: float) -> float:
        """
        Update the noise value.
        
        Args:
            dt: Time step
            
        Returns:
            float: Updated noise value
        """
        # Probability of switching
        switch_probability = 0.5 - 0.5 * np.exp(-2 * self.gamma * dt)
        r = np.random.rand()
        
        if r < switch_probability:
            self.x = -self.x
            
        return self.x
    
    def reset(self, x0: Optional[float] = None) -> float:
        """
        Reset the noise to initial state.
        
        Args:
            x0: New initial value (if None, use original x0)
            
        Returns:
            float: Reset noise value
        """
        if x0 is None:
            if self.x0 is None:
                self.x = self.sigma * (2 * np.random.randint(0, 2) - 1)
            else:
                self.x = self.x0
        else:
            self.x = x0
            
        return self.x


class OU_noise:
    """
    Ornstein-Uhlenbeck noise model.
    
    This class implements a continuous-time Markov process that generates
    correlated noise with exponential autocorrelation.
    
    Attributes:
        sigma (float): Noise amplitude
        gamma (float): Correlation rate (Hz)
        tc (float): Correlation time (1/gamma)
        x (float): Current noise value
        x0 (Optional[float]): Initial value
    """
    
    def __init__(self, sigma: float, gamma: float, x0: Optional[float] = None):
        """
        Initialize OU noise.
        
        Args:
            sigma: Noise amplitude
            gamma: Correlation rate (Hz)
            x0: Initial value (if None, random initial state)
        """
        self.tc = 1 / gamma
        self.sigma = sigma
        self.x0 = x0
        
        if x0 is None:
            self.x = np.random.normal(0, sigma)
        else:
            self.x = x0

        self.name = f'ou_tc_{self.tc:.2e}_sigma_{self.sigma:.2e}'
        self.constructor = np.array([sigma, gamma])

    def update(self, dt: float) -> float:
        """
        Update the noise value using the OU process.
        
        Args:
            dt: Time step
            
        Returns:
            float: Updated noise value
        """
        # OU update equation: dx = -γx dt + σ√(2γ) dW
        self.x = (self.x * np.exp(-dt / self.tc) + 
                  np.sqrt(1 - np.exp(-2 * dt / self.tc)) * np.random.normal(0, self.sigma))
        return self.x

    def reset(self, x0: Optional[float] = None) -> float:
        """
        Reset the noise to initial state.
        
        Args:
            x0: New initial value (if None, use original x0)
            
        Returns:
            float: Reset noise value
        """
        if x0 is None:
            self.x = np.random.normal(0, self.sigma)
        else:
            self.x = x0
        return self.x

    def set_x(self, x0: float) -> float:
        """
        Set the noise to a specific value.
        
        Args:
            x0: New noise value
            
        Returns:
            float: Set noise value
        """
        self.x = x0
        return self.x

    def update_mu(self, dt: float, mu: float, std: float) -> float:
        """
        Update mean value for analytical calculations.
        
        Args:
            dt: Time step
            mu: Current mean
            std: Current standard deviation
            
        Returns:
            float: Updated mean
        """
        return mu * np.exp(-dt / self.tc)
    
    def update_std(self, dt: float, mu: float, std: float) -> float:
        """
        Update standard deviation for analytical calculations.
        
        Args:
            dt: Time step
            mu: Current mean
            std: Current standard deviation
            
        Returns:
            float: Updated standard deviation
        """
        return np.sqrt(self.sigma**2 + (std**2 - self.sigma**2) * np.exp(-2 * dt / self.tc))
    
    def update_and_integrate(self, t: float) -> float:
        """
        Update and integrate the noise over time t.
        
        This method provides an efficient way to update the noise and
        calculate its time integral for long time intervals.
        
        Args:
            t: Integration time
            
        Returns:
            float: Time integral of the noise
        """
        if t / self.tc > 10:
            # Long time limit: use analytical expressions
            avg = self.x * self.tc * (1 - np.exp(-t / self.tc))
            sig = np.sqrt(2 * self.sigma**2 / self.tc)
            mu = 1 / self.tc
            std2 = sig**2 / 2 / mu**3 * (2 * mu * t - 3 + 4 * np.exp(-mu * t) - np.exp(-2 * mu * t))
            self.update(t)
            random = np.random.normal(loc=avg, scale=np.sqrt(std2))
            return random
            
        elif self.tc / t > 10:
            # Short time limit: use trapezoidal approximation
            x_old = self.x
            self.update(t)
            return (x_old + self.x) * t / 2
            
        else:
            # Intermediate time: use numerical integration
            x_old = self.x
            dt = self.tc / 100
            times = np.arange(dt, t + dt, dt)
            N = len(times)
            
            # Generate Wiener process
            wiener_process = np.sqrt(2 * self.sigma**2 * dt / self.tc) * np.random.normal(0, 1, size=N)
            exp_factors = np.exp((times - t) / self.tc)
            
            # Update state
            self.x = self.x * np.exp(-t / self.tc) + np.dot(wiener_process, exp_factors)
            
            # Calculate integral
            res = (x_old - self.x + np.sum(wiener_process)) * self.tc
            return res


class Over_f_noise:
    """
    1/f noise model using multiple fluctuators.
    
    This class implements 1/f noise by combining multiple two-level
    fluctuators with different switching rates.
    
    Attributes:
        n_telegraphs (int): Number of fluctuators
        S1 (float): 1/f noise amplitude
        sigma (float): Total noise amplitude
        sigma_couplings (float): Coupling strength variation
        ommax (float): Maximum frequency
        ommin (float): Minimum frequency
        fluctuators (List): List of individual fluctuators
        x (float): Current total noise value
    """
    
    def __init__(self, n_fluctuators: int, S1: float, sigma_couplings: float,
                 ommax: float, ommin: float, dt: float = 1e-3,
                 fluctuator_class: type = OU_noise, x0: Optional[float] = None,
                 equally_dist: bool = False):
        """
        Initialize 1/f noise.
        
        Args:
            n_fluctuators: Number of fluctuators
            S1: 1/f noise amplitude
            sigma_couplings: Coupling strength variation
            ommax: Maximum frequency
            ommin: Minimum frequency
            dt: Time step
            fluctuator_class: Class of individual fluctuators
            x0: Initial value
            equally_dist: Whether to distribute frequencies equally
        """
        self.n_telegraphs = int(n_fluctuators)
        self.S1 = S1 * ueV_to_MHz
        self.sigma = np.sqrt(2 * S1 * np.log(ommax / ommin))
        self.sigma_couplings = sigma_couplings
        self.ommax = ommax
        self.ommin = ommin
        self.equally_dist = equally_dist
        self.fluctuator_class = fluctuator_class
        self.dt = dt
        
        # Spawn individual fluctuators
        self.spawn_fluctuators(int(n_fluctuators), sigma_couplings)
        
        # Initialize total noise
        if x0 is None:
            self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        else:
            self.x = x0
            
        self.sigma = np.sqrt(np.sum([fluctuator.sigma**2 for fluctuator in self.fluctuators]))
        self.name = (f'over_f_n_{n_fluctuators}_S1_{S1:.2e}_'
                    f'sigma_couplings_{sigma_couplings:.2e}_'
                    f'ommax_{ommax:.2e}_ommin_{ommin:.2e}')
        self.constructor = np.array([int(n_fluctuators), S1, sigma_couplings, ommax, ommin])
        
    def spawn_fluctuators(self, n_fluctuator: int, sigma_couplings: float):
        """
        Create individual fluctuators with distributed parameters.
        
        Args:
            n_fluctuator: Number of fluctuators to create
            sigma_couplings: Coupling strength variation
        """
        uni = np.random.uniform(0, 1, size=n_fluctuator)
        
        if self.equally_dist:
            # Equally distributed frequencies
            gammas = self.ommin * np.exp(np.log(self.ommax / self.ommin) * np.linspace(0, 1, n_fluctuator))
        else:
            # Log-uniformly distributed frequencies
            gammas = self.ommax * np.exp(-np.log(self.ommax / self.ommin) * uni)
            
        # Distribute noise amplitudes
        sigmas = (self.sigma / np.sqrt(n_fluctuator) * 
                  np.random.normal(1, sigma_couplings, size=n_fluctuator))
        
        # Create fluctuators
        self.fluctuators = []
        for n, gamma in enumerate(gammas):
            self.fluctuators.append(self.fluctuator_class(sigmas[n], gamma))
        
    def update(self, dt: float) -> float:
        """
        Update all fluctuators and return total noise.
        
        Args:
            dt: Time step
            
        Returns:
            float: Total noise value
        """
        self.x = np.sum([fluctuator.update(dt) for fluctuator in self.fluctuators])
        return self.x
    
    def reset(self, x0: Optional[float] = None) -> float:
        """
        Reset all fluctuators.
        
        Args:
            x0: Initial value (if None, random)
            
        Returns:
            float: Reset noise value
        """
        for fluctuator in self.fluctuators:
            fluctuator.reset(x0)
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

    def update_mu(self, dt: float, mu: float, std: float) -> float:
        """Update mean (not implemented for 1/f noise)."""
        return mu
    
    def update_std(self, dt: float, mu: float, std: float) -> float:
        """Update standard deviation."""
        return self.sigma
        
    def set_x(self, x0: float) -> float:
        """
        Set noise to specific value by distributing across fluctuators.
        
        Args:
            x0: Target noise value
            
        Returns:
            float: Set noise value
        """
        x0s = x0 * np.random.normal(1, 0.5, size=len(self.fluctuators)) / len(self.fluctuators)
        for n, fluctuator in enumerate(self.fluctuators):
            fluctuator.set_x(x0s[n])
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

    def gen_trajectory(self, times: np.ndarray) -> np.ndarray:
        """
        Generate noise trajectory over time.
        
        Args:
            times: Time array
            
        Returns:
            np.ndarray: Noise trajectory
        """
        trajectory = []
        for time in times:
            trajectory.append(self.update(time))
        return trajectory
    
    def update_and_integrate(self, t: float) -> float:
        """
        Update and integrate noise over time.
        
        Args:
            t: Integration time
            
        Returns:
            float: Time integral
        """
        I = 0
        for fluctuator in self.fluctuators:
            I += fluctuator.update_and_integrate(t)
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return I
    
    def ideal_spectrum(self, oms: np.ndarray) -> np.ndarray:
        """
        Calculate ideal power spectrum.
        
        Args:
            oms: Angular frequencies
            
        Returns:
            np.ndarray: Power spectral density
        """
        Sxx = np.zeros(len(oms))
        for fluctuator in self.fluctuators:
            Sxx += (fluctuator.sigma**2 * fluctuator.tc / 
                   (fluctuator.tc**2 * oms**2 + 1))
        return Sxx

 