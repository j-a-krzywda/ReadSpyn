"""
ReadSpyn: A Quantum Dot Readout Simulator

A comprehensive simulator for quantum dot readout systems with realistic noise models
and RLC resonator sensors.

Author: Jan A. Krzywda
Email: j.a.krzywda@liacs.leidenuniv.nl
"""

from .simulator import ReadoutSimulator
from .sensor_backend import RLC_sensor
from .quantum_dot_system import QuantumDotSystem
from .noise_models import OverFNoise
from .noise import OU_noise, Telegraph_Noise, Over_f_noise

__version__ = "0.1.0"
__author__ = "Jan A. Krzywda"
__email__ = "j.a.krzywda@liacs.leidenuniv.nl"

__all__ = [
    "ReadoutSimulator",
    "RLC_sensor", 
    "QuantumDotSystem",
    "OverFNoise",
    "OU_noise",
    "Telegraph_Noise",
    "Over_f_noise"
] 