"""
Quantum Dot System Module

This module provides the QuantumDotSystem class for modeling systems of quantum
dots with capacitive coupling and sensor interactions.
"""

import numpy as np
from typing import Optional, Tuple


class QuantumDotSystem:
    """
    Manages a system of quantum dots with a constant interaction model.
    
    This class represents a system of quantum dots characterized by their
    mutual capacitances and coupling to sensors.
    
    Attributes:
        num_dots (int): Number of quantum dots in the system
        num_sensors (int): Number of sensors coupled to the system
        Cdd_inv (np.ndarray): Inverse of dot-dot capacitance matrix
        Cds (np.ndarray): Dot-sensor capacitance matrix
    """
    
    def __init__(self, Cdd: np.ndarray, Cds: np.ndarray):
        """
        Initialize the quantum dot system.
        
        Args:
            Cdd: Dot-dot capacitance matrix (Ndots × Ndots)
            Cds: Dot-sensor capacitance matrix (Ndots × Nsensors)
            
        Raises:
            ValueError: If matrix dimensions are inconsistent
        """
        if Cdd.shape[0] != Cdd.shape[1] or Cdd.shape[0] != Cds.shape[0]:
            raise ValueError(f"Inconsistent dimensions: Cdd is {Cdd.shape}, Cds is {Cds.shape}. "
                           f"Cdd must be square and Cds must have same number of rows.")
        
        self.num_dots = Cdd.shape[0]
        self.num_sensors = Cds.shape[1]
        self.Cdd_inv = np.linalg.inv(Cdd)
        self.Cds = Cds
        
        # Print coupling information
        coupling_strength = Cds.T @ self.Cdd_inv
        print(f"Dot-sensor coupling strength (Δε/ε_w): {coupling_strength}")

    @classmethod
    def from_random(cls, num_dots: int, num_sensors: int, 
                   Css: float = 1e-15) -> 'QuantumDotSystem':
        """
        Generate a QuantumDotSystem with random, experimentally relevant parameters.
        
        Args:
            num_dots: Number of quantum dots
            num_sensors: Number of sensors
            Css: Scale factor for capacitances (default: 1 fF)
            
        Returns:
            QuantumDotSystem: New instance with random parameters
        """
        # Generate diagonal elements (self-capacitances)
        Cdd_diag = np.random.uniform(8, 12, num_dots) * Css
        
        # Generate off-diagonal elements (mutual capacitances)
        Cdd_off_diag = np.random.uniform(-0.2, -0.1, (num_dots, num_dots)) * Css
        
        # Construct full capacitance matrix
        Cdd = np.diag(Cdd_diag) + Cdd_off_diag
        
        # Generate dot-sensor coupling capacitances
        Cds = np.random.uniform(-0.1, -0.05, (num_dots, num_sensors)) * Css
        
        return cls(Cdd, Cds)

    def get_energy_offset(self, charge_state: np.ndarray, 
                         sensor_voltages: np.ndarray, 
                         eps0: float) -> np.ndarray:
        """
        Calculate the energy offset for each sensor.
        
        The energy offset is determined by the charge state, sensor voltages,
        and a common gate voltage offset.
        
        Args:
            charge_state: Vector of charge states for each dot
            sensor_voltages: Vector of voltages applied to sensors
            eps0: Common gate voltage offset
            
        Returns:
            np.ndarray: Energy offset for each sensor
            
        Raises:
            ValueError: If charge_state has wrong dimension
        """
        if charge_state.shape[0] != self.num_dots:
            raise ValueError(f"Charge state vector must have length {self.num_dots}, "
                           f"but got {charge_state.shape[0]}.")
            
        # Calculate energy offset including DC gate voltage
        energy_offset = (self.Cds.T @ self.Cdd_inv @ 
                        (charge_state + self.Cds @ sensor_voltages) + eps0)
        
        return energy_offset

    def get_coupling_matrix(self) -> np.ndarray:
        """
        Get the effective coupling matrix between dots and sensors.
        
        Returns:
            np.ndarray: Coupling matrix Cds^T @ Cdd^(-1)
        """
        return self.Cds.T @ self.Cdd_inv
    
    def get_dot_energies(self, charge_state: np.ndarray, 
                        sensor_voltages: np.ndarray, 
                        eps0: float) -> np.ndarray:
        """
        Calculate the energy of each quantum dot.
        
        Args:
            charge_state: Vector of charge states for each dot
            sensor_voltages: Vector of voltages applied to sensors
            eps0: Common gate voltage offset
            
        Returns:
            np.ndarray: Energy of each dot
        """
        if charge_state.shape[0] != self.num_dots:
            raise ValueError(f"Charge state vector must have length {self.num_dots}.")
            
        # Energy of each dot
        dot_energies = self.Cdd_inv @ (charge_state + self.Cds @ sensor_voltages) + eps0
        return dot_energies

    def __repr__(self) -> str:
        """String representation of the quantum dot system."""
        return (f"QuantumDotSystem(num_dots={self.num_dots}, "
                f"num_sensors={self.num_sensors})")
    
    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"QuantumDotSystem:\n"
                f"  Number of dots: {self.num_dots}\n"
                f"  Number of sensors: {self.num_sensors}\n"
                f"  Dot-dot capacitance scale: {np.mean(np.abs(self.Cdd_inv)):.2e} F⁻¹\n"
                f"  Dot-sensor coupling scale: {np.mean(np.abs(self.Cds)):.2e} F")