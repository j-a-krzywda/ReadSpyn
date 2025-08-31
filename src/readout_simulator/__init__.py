"""
ReadSpyn - JAX-based quantum dot readout simulator

A comprehensive simulator for quantum dot readout systems with realistic noise models,
RLC resonator sensors, and efficient JAX-based state scanning.
"""

from .quantum_dot_system import QuantumDotSystem
from .sensor_backend import RLC_sensor
from .noise_models import OU_noise, OverFNoise
from .jax_simulator import JAXReadoutSimulator

__version__ = "2.0.0"
__author__ = "Jan A. Krzywda"

__all__ = [
    "QuantumDotSystem",
    "RLC_sensor", 
    "OU_noise",
    "OverFNoise",
    "JAXReadoutSimulator"
] 