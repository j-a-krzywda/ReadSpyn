# ReadSpyn Examples

This directory contains example scripts demonstrating various features of the ReadSpyn quantum dot readout simulator.

## Available Examples

### 1. Basic Simulation (`basic_simulation.py`)
**Purpose**: Introduction to ReadSpyn with a simple single-sensor system
**Features**:
- Single quantum dot system (2 dots, 1 sensor)
- Basic noise models
- Simple visualization
- Performance analysis

**Usage**:
```bash
python3 basic_simulation.py
```

**What you'll learn**:
- How to create quantum dot systems
- Basic sensor configuration
- Running simulations
- Extracting and analyzing results

### 2. Simple Two-Sensor System (`simple_two_sensor.py`)
**Purpose**: Demonstrate multi-sensor capabilities with minimal complexity
**Features**:
- 2 quantum dots, 2 sensors
- Different resonator parameters for each sensor
- Different noise models for each sensor
- Basic performance comparison

**Usage**:
```bash
python3 simple_two_sensor.py
```

**What you'll learn**:
- Multi-sensor system setup
- Parameter variation between sensors
- Performance comparison between sensors

### 3. Advanced Two-Sensor System (`two_sensor_system.py`)
**Purpose**: Comprehensive multi-sensor analysis with advanced features
**Features**:
- 2 quantum dots, 2 sensors
- Different resonator configurations
- Comprehensive noise modeling
- Advanced visualization and analysis
- Charge state separation analysis
- Sensor correlation analysis

**Usage**:
```bash
python3 two_sensor_system.py
```

**What you'll learn**:
- Advanced system configuration
- Complex noise modeling
- Detailed performance analysis
- Multi-dimensional visualization

## Running the Examples

### Prerequisites
Make sure you have ReadSpyn installed:
```bash
cd /path/to/ReadSpyn
pip install -e .
```

### Basic Usage
```bash
cd examples
python3 basic_simulation.py
```

### Customizing Examples
You can modify the examples to explore different configurations:

#### Changing System Parameters
```python
# Modify quantum dot system
Cdd = np.array([
    [1.0, 0.5],  # Change coupling strength
    [0.5, 1.0]
])

# Modify sensor parameters
params_resonator = {
    'Lc': 1000e-9,  # Change inductance
    'Cp': 0.8e-12,  # Change capacitance
    'RL': 50,        # Change load resistance
    # ... other parameters
}
```

#### Adjusting Noise Models
```python
# Modify 1/f noise
eps_noise = OverFNoise(
    n_fluctuators=10,     # Change number of fluctuators
    s1=2e-3,             # Change noise amplitude
    ommax=2,              # Change frequency range
    ommin=0.1
)

# Modify OU noise
c_noise = OU_noise(
    sigma=1e-12,          # Change noise amplitude
    gamma=1e6             # Change correlation rate
)
```

#### Changing Simulation Parameters
```python
# Modify simulation settings
nT_end = 2000            # Change simulation duration
samples = 100            # Change number of samples
params = {
    'SNR_white': 1e13,   # Change signal-to-noise ratio
    'eps0': 0.3          # Change operating point
}
```

## Example Output

### Basic Simulation
- Creates a 2-dot, 1-sensor system
- Runs simulation with 50 samples per charge state
- Generates IQ plots and performance metrics
- Shows time evolution of signals

### Two-Sensor Systems
- Creates 2-dot, 2-sensor systems
- Demonstrates different sensor configurations
- Compares performance between sensors
- Shows correlation analysis
- Provides comprehensive visualizations

## Understanding the Results

### IQ Plots
- **X-axis**: In-phase component (I)
- **Y-axis**: Quadrature component (Q)
- **Colors**: Different charge states
- **Clustering**: Well-separated clusters indicate good readout performance

### Performance Metrics
- **Fidelity**: Measure of readout accuracy (0-1, higher is better)
- **SNR**: Signal-to-noise ratio evolution over time
- **Separation**: Distance between different charge state clusters

### Time Evolution
- Shows how signals develop over time
- Demonstrates noise effects
- Illustrates integration benefits

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'readout_simulator'`
**Solution**: Ensure ReadSpyn is installed: `pip install -e .`

#### 2. Memory Issues
**Problem**: Out of memory errors
**Solution**: Reduce `nT_end` or `samples` parameters

#### 3. Slow Performance
**Problem**: Simulations take too long
**Solution**: 
- Reduce simulation duration
- Use fewer samples
- Ensure Numba is working properly

#### 4. Visualization Issues
**Problem**: Plots don't show or are empty
**Solution**: Check that simulation completed successfully and data was extracted

### Performance Tips

1. **Start Small**: Begin with basic examples and small parameter values
2. **Monitor Progress**: Watch the progress bars during simulation
3. **Check Output**: Verify that conductance values are reasonable
4. **Adjust Parameters**: Modify parameters to see their effects

## Next Steps

After running the examples:

1. **Experiment**: Modify parameters to see their effects
2. **Scale Up**: Increase system size or simulation duration
3. **Customize**: Adapt examples for your specific research needs
4. **Analyze**: Use the analysis functions to understand your results
5. **Extend**: Create new examples for different system configurations

## Contributing

Feel free to:
- Modify existing examples
- Create new examples
- Improve visualizations
- Add new analysis features
- Report issues or suggest improvements

## Support

If you encounter problems:
1. Check this README for troubleshooting tips
2. Review the main README.md file
3. Check the API_REFERENCE.md for detailed documentation
4. Open an issue on GitHub
5. Contact the author: j.a.krzywda@liacs.leidenuniv.nl

Happy simulating! 🚀 