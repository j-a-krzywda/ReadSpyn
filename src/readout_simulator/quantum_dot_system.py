
import numpy as np

class QuantumDotSystem:
    """
    Manages a system of quantum dots with a constant interaction model.
    """
    def __init__(self, Cdd: np.ndarray, Cds: np.ndarray):
        """
        Initializes the quantum dot system with dot-dot (Cdd) and
        dot-sensor (Cds) capacitance matrices.
        """
        if Cdd.shape[0] != Cdd.shape[1] or Cdd.shape[0] != Cds.shape[0]:
            raise ValueError("Inconsistent dimensions in capacitance matrices.")
        
        self.num_dots = Cdd.shape[0]
        self.num_sensors = Cds.shape[1]
        self.Cdd_inv = np.linalg.inv(Cdd)
        self.Cds = Cds
        print(f"$\Delta \epsilon/\epsilon_w$", Cds.T @ self.Cdd_inv)


    @classmethod
    def from_random(cls, num_dots: int, num_sensors: int, Css: float = 1e-15):
        """
        Generates a QuantumDotSystem with random, experimentally relevant parameters.
        """
        Cdd_diag = np.random.uniform(8, 12, num_dots) * Css
        Cdd_off_diag = np.random.uniform(-0.2, -0.1, (num_dots, num_dots)) * Css
        Cdd = np.diag(Cdd_diag) + Cdd_off_diag
        Cds = np.random.uniform(-0.1, -0.05, (num_dots, num_sensors)) * Css
        return cls(Cdd, Cds)

    def get_energy_offset(self, charge_state: np.ndarray, sensor_voltages: np.ndarray, eps0: float) -> np.ndarray:
        """
        Calculates the energy offset for each sensor based on the charge state,
        sensor voltages, and a common gate voltage (eps0).
        """
        if charge_state.shape[0] != self.num_dots:
            raise ValueError(f"The charge_state vector must have a length of {self.num_dots}, but it has a length of {charge_state.shape[0]}.")
            
        # The total energy now includes the DC offset from the gate voltage
        return self.Cds.T @ self.Cdd_inv @ (charge_state + self.Cds @ sensor_voltages) + eps0

    def __repr__(self):
        return f"QuantumDotSystem(num_dots={self.num_dots}, num_sensors={self.num_sensors})"