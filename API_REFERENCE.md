# ReadSpyn API Reference

This document provides a comprehensive reference for all classes, methods, and functions in the ReadSpyn package.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Noise Models](#noise-models)
3. [Utility Functions](#utility-functions)
4. [Configuration Examples](#configuration-examples)

## Core Classes

### QuantumDotSystem

The main class for modeling quantum dot systems with capacitive coupling.

#### Constructor

```python
QuantumDotSystem(Cdd: np.ndarray, Cds: np.ndarray)
```

**Parameters:**
- `Cdd`: Dot-dot capacitance matrix (Ndots × Ndots)
- `Cds`: Dot-sensor capacitance matrix (Ndots × Nsensors)

**Example:**
```python
Cdd = np.array([[1, 0], [0, 1]])  # 2x2 matrix
Cds = np.array([[1], [0.1]])      # 2x1 matrix
dot_system = QuantumDotSystem(Cdd, Cds)
```

#### Methods

##### `get_energy_offset(charge_state, sensor_voltages, eps0)`

Calculate energy offset for each sensor.

**Parameters:**
- `charge_state`: Vector of charge states for each dot
- `sensor_voltages`: Vector of voltages applied to sensors
- `eps0`: Common gate voltage offset

**Returns:**
- `np.ndarray`: Energy offset for each sensor

##### `get_coupling_matrix()`

Get effective coupling matrix between dots and sensors.

**Returns:**
- `np.ndarray`: Coupling matrix Cds^T @ Cdd^(-1)

##### `get_dot_energies(charge_state, sensor_voltages, eps0)`

Calculate energy of each quantum dot.

**Parameters:**
- `charge_state`: Vector of charge states for each dot
- `sensor_voltages`: Vector of voltages applied to sensors
- `eps0`: Common gate voltage offset

**Returns:**
- `np.ndarray`: Energy of each dot

##### `from_random(num_dots, num_sensors, Css=1e-15)`

Generate random quantum dot system with realistic parameters.

**Parameters:**
- `num_dots`: Number of quantum dots
- `num_sensors`: Number of sensors
- `Css`: Scale factor for capacitances (default: 1 fF)

**Returns:**
- `QuantumDotSystem`: New instance with random parameters

### RLC_sensor

Represents an RLC resonator sensor for quantum dot readout.

#### Constructor

```python
RLC_sensor(params_resonator: Dict[str, float], 
           params_coulomb_peak: Dict[str, float],
           c_noise_model: Optional[OU_noise] = None, 
           eps_noise_model: Optional[OU_noise] = None)
```

**Parameters:**
- `params_resonator`: Dictionary containing resonator parameters
- `params_coulomb_peak`: Dictionary containing Coulomb peak parameters
- `c_noise_model`: Capacitance noise model (optional)
- `eps_noise_model`: Energy offset noise model (optional)

**Resonator Parameters:**
```python
params_resonator = {
    'Lc': 800e-9,      # Inductance (H)
    'Cp': 0.5e-12,     # Parasitic capacitance (F)
    'RL': 40,          # Load resistance (Ω)
    'Rc': 100e6,       # Coupling resistance (Ω)
    'Z0': 50           # Characteristic impedance (Ω)
}
```

**Coulomb Peak Parameters:**
```python
params_coulomb_peak = {
    'g0': 1/50/1e6,    # Maximum conductance (S)
    'eps0': 0.5,       # Operating point (relative to eps_width)
    'eps_width': 1      # Energy width (eV)
}
```

#### Methods

##### `get_signal(times, dot_system, charge_state, sensor_index, params, noise_trajectory=None)`

Simulate IQ signal for given charge state and noise trajectory.

**Parameters:**
- `times`: Time array for simulation
- `dot_system`: Quantum dot system
- `charge_state`: Charge state vector
- `sensor_index`: Index of this sensor
- `params`: Simulation parameters
- `noise_trajectory`: Optional noise trajectory

**Returns:**
- `tuple`: (I, Q, V_refl_t, times)
  - `I`: In-phase component
  - `Q`: Quadrature component
  - `V_refl_t`: Raw reflected voltage
  - `times`: Time array

### ReadoutSimulator

Main class for orchestrating readout simulations.

#### Constructor

```python
ReadoutSimulator(dot_system: QuantumDotSystem, 
                sensors: Optional[List[RLC_sensor]] = None)
```

**Parameters:**
- `dot_system`: Quantum dot system to simulate
- `sensors`: List of sensor objects (optional)

#### Methods

##### `run_simulation(charge_states, t_end, params)`

Run simulation for list of charge states across all sensors.

**Parameters:**
- `charge_states`: List of charge state arrays to simulate
- `t_end`: End time in units of sensor oscillation periods
- `params`: Dictionary containing simulation parameters

**Simulation Parameters:**
```python
params = {
    'SNR_white': 1e12,        # White noise signal-to-noise ratio
    'eps0': 0.5,              # Nominal position on Coulomb peak
    'plot_conductance': True   # Enable conductance plotting
}
```

##### `get_int_IQ()`

Get integrated IQ data for all sensors and charge states.

**Returns:**
- `tuple`: (integrated_IQ_data, times)
  - `integrated_IQ_data`: Array of dictionaries with 'I' and 'Q' keys
  - `times`: Time array

##### `get_raw_signal()`

Get raw signal data for all simulations.

**Returns:**
- `np.ndarray`: Array of raw signal data

##### `calculate_fidelity(sensor_index=0)`

Calculate readout fidelity for specific sensor.

**Parameters:**
- `sensor_index`: Index of sensor to analyze

**Returns:**
- `float`: Readout fidelity (0-1)

##### `plot_results()`

Plot integrated IQ results for each sensor.

##### `get_sensor_results(sensor_index)`

Get results for specific sensor.

**Parameters:**
- `sensor_index`: Index of sensor

**Returns:**
- `List[Dict]`: Results for specified sensor

##### `get_charge_state_results(charge_state_id)`

Get results for specific charge state.

**Parameters:**
- `charge_state_id`: ID of charge state

**Returns:**
- `List[Dict]`: Results for specified charge state

## Noise Models

### OU_noise

Ornstein-Uhlenbeck noise model for correlated noise.

#### Constructor

```python
OU_noise(sigma: float, gamma: float, x0: Optional[float] = None)
```

**Parameters:**
- `sigma`: Noise amplitude
- `gamma`: Correlation rate (Hz)
- `x0`: Initial value (optional)

#### Methods

##### `update(dt)`

Update noise value using OU process.

**Parameters:**
- `dt`: Time step

**Returns:**
- `float`: Updated noise value

##### `reset(x0=None)`

Reset noise to initial state.

**Parameters:**
- `x0`: New initial value (optional)

**Returns:**
- `float`: Reset noise value

### OverFNoise

Wrapper for 1/f noise model using multiple fluctuators.

#### Constructor

```python
OverFNoise(n_fluctuators: int, s1: float, sigma_couplings: float,
           ommax: float, ommin: float, dt: float, equally_dist: bool = False)
```

**Parameters:**
- `n_fluctuators`: Number of fluctuators
- `s1`: 1/f noise amplitude
- `sigma_couplings`: Coupling strength variation
- `ommax`: Maximum frequency
- `ommin`: Minimum frequency
- `dt`: Time step
- `equally_dist`: Whether to distribute frequencies equally

#### Methods

##### `generate_trajectory(num_points)`

Generate noise trajectory of given length.

**Parameters:**
- `num_points`: Number of points in trajectory

**Returns:**
- `np.ndarray`: Noise trajectory array

### Telegraph_Noise

Two-level fluctuator noise model.

#### Constructor

```python
Telegraph_Noise(sigma: float, gamma: float, x0: Optional[float] = None)
```

**Parameters:**
- `sigma`: Noise amplitude
- `gamma`: Switching rate (Hz)
- `x0`: Initial value (optional)

## Utility Functions

### `get_spectrum(signal, time_step, total_time)`

Calculate power spectral density of signal.

**Parameters:**
- `signal`: Input signal array
- `time_step`: Time step between samples
- `total_time`: Total duration of signal

**Returns:**
- `tuple`: (frequencies, power_spectrum)
  - `frequencies`: Frequency array (Hz)
  - `power_spectrum`: Power spectral density

## Configuration Examples

### Basic Two-Dot System

```python
# Create quantum dot system
Cdd = np.array([[1, 0], [0, 1]])
Cds = np.array([[1], [0.1]])
dot_system = QuantumDotSystem(Cdd, Cds)

# Configure resonator
params_resonator = {
    'Lc': 800e-9,
    'Cp': 0.5e-12,
    'RL': 40,
    'Rc': 100e6,
    'Z0': 50
}

params_coulomb_peak = {
    'g0': 1/50/1e6,
    'eps0': 0.5,
    'eps_width': 1
}

# Create noise models
eps_noise = OverFNoise(n_fluctuators=5, s1=1e-3, sigma_couplings=1e-99,
                       ommax=1, ommin=0.2, dt=1, equally_dist=True)
c_noise = OU_noise(sigma=1e-13, gamma=1e5)

# Create sensor and simulator
sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
simulator = ReadoutSimulator(dot_system, [sensor])

# Run simulation
charge_states = [np.array([1, 0]), np.array([0, 1])] * 50
params = {'SNR_white': 1e12, 'eps0': 0.5}
simulator.run_simulation(charge_states, 2500, params)

# Get results
IQ_data, times = simulator.get_int_IQ()
fidelity = simulator.calculate_fidelity()
```

### Multi-Sensor System

```python
# Create quantum dot system with multiple sensors
Cdd = np.array([[1, 0], [0, 1]])
Cds = np.array([[1, 0.5], [0.1, 0.8]])  # 2 dots, 2 sensors
dot_system = QuantumDotSystem(Cdd, Cds)

# Create multiple sensors with different parameters
sensors = []
for i in range(2):
    params_resonator = {
        'Lc': 800e-9 * (1 + 0.1 * i),  # Slightly different inductances
        'Cp': 0.5e-12,
        'RL': 40,
        'Rc': 100e6,
        'Z0': 50
    }
    
    params_coulomb_peak = {
        'g0': 1/50/1e6,
        'eps0': 0.5,
        'eps_width': 1
    }
    
    sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
    sensors.append(sensor)

# Create simulator
simulator = ReadoutSimulator(dot_system, sensors)

# Run simulation
simulator.run_simulation(charge_states, 2500, params)

# Analyze each sensor
for i in range(len(sensors)):
    fidelity = simulator.calculate_fidelity(sensor_index=i)
    print(f"Sensor {i} fidelity: {fidelity:.3f}")
```

### Custom Noise Model

```python
class CustomNoise:
    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude
        self.frequency = frequency
        self.x = 0
        self.t = 0
    
    def update(self, dt):
        self.t += dt
        # Implement your custom noise model here
        self.x = self.amplitude * np.sin(2 * np.pi * self.frequency * self.t)
        return self.x
    
    def reset(self, x0=None):
        if x0 is None:
            self.x = 0
        else:
            self.x = x0
        self.t = 0
        return self.x

# Use custom noise model
custom_noise = CustomNoise(amplitude=1e-12, frequency=1e6)
sensor = RLC_sensor(params_resonator, params_coulomb_peak, custom_noise, eps_noise)
```

## Error Handling

The package includes comprehensive error checking:

- **Dimension Mismatch**: Ensures capacitance matrices have consistent dimensions
- **Parameter Validation**: Validates input parameters and provides helpful error messages
- **State Validation**: Checks charge state vectors match system dimensions
- **Sensor Configuration**: Ensures number of sensors matches system configuration

## Performance Tips

1. **Use Numba**: The package uses Numba for optimized numerical computations
2. **Vectorize Operations**: Use NumPy arrays for efficient bulk operations
3. **Memory Management**: For long simulations, consider processing data in chunks
4. **Parallel Processing**: Multiple sensors can be simulated in parallel (future feature)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is properly installed and in your Python path
2. **Memory Issues**: Reduce simulation duration or number of samples for large systems
3. **Convergence Problems**: Check ODE solver parameters in sensor implementation
4. **Noise Artifacts**: Verify noise model parameters are physically reasonable

### Debug Mode

Enable debug output by setting environment variable:
```bash
export READSPYN_DEBUG=1
```

This will provide additional logging information during simulation execution. 