"""
JAX-based Noise Models for ReadSpyn

This module provides JAX-compatible noise models for simulating realistic quantum dot
readout systems, including Ornstein-Uhlenbeck processes and 1/f noise.
"""

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy jax and jnp for when JAX is not available
    class DummyJAX:
        def __getattr__(self, name):
            raise ImportError("JAX is not available. Install JAX to use JAX features.")
    jax = DummyJAX()
    jnp = DummyJAX()
from typing import Optional, Tuple, Dict, Any, Union
from functools import partial


class OU_noise:
    """
    JAX-compatible Ornstein-Uhlenbeck noise model.
    
    This class implements a continuous-time Markov process that generates
    correlated noise with exponential autocorrelation using JAX for efficient
    vectorized operations.
    
    Attributes:
        sigma (float): Noise amplitude
        gamma (float): Correlation rate (Hz)
        tc (float): Correlation time (1/gamma)
    """
    
    def __init__(self, sigma: float, gamma: float):
        """
        Initialize OU noise.
        
        Args:
            sigma: Noise amplitude
            gamma: Correlation rate (Hz)
        """
        self.sigma = sigma
        self.gamma = gamma
        self.tc = 1 / gamma
        
    def generate_trajectory(self, key: jax.random.PRNGKey, times: jax.Array) -> jax.Array:
        """
        Generate a complete noise trajectory using JAX.
        
        Args:
            key: JAX PRNG key for random number generation
            times: Time array
            
        Returns:
            jax.Array: Noise trajectory
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for OU_noise.generate_trajectory. Install JAX to use this feature.")
        dt = times[1] - times[0]
        n_steps = len(times)
        
        # Generate Wiener process increments
        key, subkey = jax.random.split(key)
        dw = jax.random.normal(subkey, shape=(n_steps,)) * jnp.sqrt(2 * self.gamma * dt)
        
        # Initialize state
        x0 = jax.random.normal(key, shape=()) * self.sigma
        
        # Define the OU update function
        def ou_step(carry, dw_step):
            x = carry
            dx = -self.gamma * x * dt + self.sigma * dw_step
            return x + dx, x + dx
        
        # Scan over the trajectory
        _, trajectory = jax.lax.scan(ou_step, x0, dw)
        
        return trajectory
    
    def get_spectrum(self, times: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Calculate the theoretical power spectrum.
        
        Args:
            times: Time array
            
        Returns:
            Tuple[jax.Array, jax.Array]: (frequencies, power_spectrum)
        """
        dt = times[1] - times[0]
        freqs = jnp.fft.fftfreq(len(times), dt)
        
        # Theoretical power spectrum for OU process
        omega = 2 * jnp.pi * freqs
        spectrum = (2 * self.sigma**2 * self.tc) / (1 + (omega * self.tc)**2)
        
        return freqs, spectrum


class OverFNoise:
    """
    JAX-compatible 1/f noise model using multiple fluctuators.
    
    This class implements 1/f noise by combining multiple Ornstein-Uhlenbeck
    fluctuators with different correlation times.
    
    Attributes:
        n_fluctuators (int): Number of fluctuators
        S1 (float): 1/f noise amplitude
        sigma_couplings (float): Coupling strength variation
        ommax (float): Maximum frequency
        ommin (float): Minimum frequency
        fluctuators (list): List of individual OU fluctuators
    """
    
    def __init__(self, n_fluctuators: int, S1: float, sigma_couplings: float,
                 ommax: float, ommin: float, equally_dist: bool = False):
        """
        Initialize 1/f noise.
        
        Args:
            n_fluctuators: Number of fluctuators
            S1: 1/f noise amplitude
            sigma_couplings: Coupling strength variation
            ommax: Maximum frequency
            ommin: Minimum frequency
            equally_dist: Whether to distribute frequencies equally
        """
        self.n_fluctuators = n_fluctuators
        self.S1 = S1
        self.sigma_couplings = sigma_couplings
        self.ommax = ommax
        self.ommin = ommin
        self.equally_dist = equally_dist
        
        # Create individual fluctuators
        self._create_fluctuators()
        
    def _create_fluctuators(self):
        """Create individual OU fluctuators with distributed parameters."""
        # Generate correlation times
        if self.equally_dist:
            # Equally distributed in log space
            log_gammas = jnp.linspace(jnp.log(self.ommin), jnp.log(self.ommax), self.n_fluctuators)
            gammas = jnp.exp(log_gammas)
        else:
            # Log-uniformly distributed
            key = jax.random.PRNGKey(0)  # Will be overridden in actual usage
            uni = jax.random.uniform(key, shape=(self.n_fluctuators,))
            gammas = self.ommax * jnp.exp(-jnp.log(self.ommax / self.ommin) * uni)
        
        # Calculate individual noise amplitudes
        total_variance = 2 * self.S1 * jnp.log(self.ommax / self.ommin)
        base_sigma = jnp.sqrt(total_variance / self.n_fluctuators)
        
        # Create fluctuators
        self.fluctuators = []
        for gamma in gammas:
            sigma = base_sigma * (1 + self.sigma_couplings * jax.random.normal(jax.random.PRNGKey(0)))
            self.fluctuators.append(OU_noise(sigma, gamma))
    
    def generate_trajectory(self, key: jax.random.PRNGKey, times: jax.Array) -> jax.Array:
        """
        Generate a complete 1/f noise trajectory.
        
        Args:
            key: JAX PRNG key for random number generation
            times: Time array
            
        Returns:
            jax.Array: Combined noise trajectory
        """
        # Generate individual trajectories
        trajectories = []
        for i, fluctuator in enumerate(self.fluctuators):
            subkey = jax.random.fold_in(key, i)
            traj = fluctuator.generate_trajectory(subkey, times)
            trajectories.append(traj)
        
        # Sum all trajectories
        return jnp.sum(jnp.array(trajectories), axis=0)
    
    def get_spectrum(self, times: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Calculate the theoretical power spectrum.
        
        Args:
            times: Time array
            
        Returns:
            Tuple[jax.Array, jax.Array]: (frequencies, power_spectrum)
        """
        dt = times[1] - times[0]
        freqs = jnp.fft.fftfreq(len(times), dt)
        
        # Sum individual spectra
        total_spectrum = jnp.zeros_like(freqs)
        for fluctuator in self.fluctuators:
            _, spectrum = fluctuator.get_spectrum(times)
            total_spectrum += spectrum
        
        return freqs, total_spectrum


def precompute_noise_trajectories(
    noise_model: Union[OU_noise, OverFNoise],
    key: jax.random.PRNGKey,
    times: jax.Array,
    n_realizations: int
) -> jax.Array:
    """
    Precompute multiple noise trajectory realizations.
    
    Args:
        noise_model: Noise model to use
        key: JAX PRNG key
        times: Time array
        n_realizations: Number of realizations to generate
        
    Returns:
        jax.Array: Array of shape (n_realizations, n_times) containing noise trajectories
    """
    def generate_single(key):
        return noise_model.generate_trajectory(key, times)
    
    # Generate multiple realizations
    keys = jax.random.split(key, n_realizations)
    trajectories = jax.vmap(generate_single)(keys)
    
    return trajectories