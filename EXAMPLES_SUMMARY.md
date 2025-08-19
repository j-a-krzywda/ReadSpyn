# ReadSpyn Two-Sensor Examples Summary

## Overview

I've successfully created comprehensive examples demonstrating quantum dot systems with **two sensors**, showcasing the multi-sensor capabilities of ReadSpyn. These examples range from simple to advanced, allowing users to understand and experiment with multi-sensor readout systems.

## Available Examples

### 1. Simple Two-Sensor System (`simple_two_sensor.py`)
**Purpose**: Quick introduction to multi-sensor systems
**Complexity**: Low
**Runtime**: ~1-2 minutes
**Features**:
- 2 quantum dots, 2 sensors
- Different resonator parameters for each sensor
- Basic noise models
- Simple visualization (4 plots)
- Performance comparison

**Best for**: Learning the basics of multi-sensor systems

### 2. Advanced Two-Sensor System (`two_sensor_system.py`)
**Purpose**: Comprehensive multi-sensor analysis
**Complexity**: High
**Runtime**: ~3-4 minutes
**Features**:
- 2 quantum dots, 2 sensors
- Different resonator configurations (frequency, Q-factor)
- Comprehensive noise modeling
- Advanced visualization (9 plots)
- Charge state separation analysis
- Sensor correlation analysis
- Detailed performance metrics

**Best for**: Research and advanced analysis

## What These Examples Demonstrate

### 🎯 **Multi-Sensor Capabilities**
- **Independent Sensor Configuration**: Each sensor can have different parameters
- **Parallel Simulation**: Both sensors simulate simultaneously
- **Performance Comparison**: Direct comparison between sensor performance
- **Correlation Analysis**: Understanding how sensors relate to each other

### 🔬 **System Complexity**
- **2 Quantum Dots**: Demonstrates dot-dot coupling effects
- **2 Sensors**: Shows multi-sensor readout strategies
- **4 Charge States**: (0,0), (1,0), (0,1), (1,1) - realistic quantum dot configurations
- **Different Noise Models**: Each sensor can have different noise characteristics

### 📊 **Advanced Analysis**
- **IQ Space Analysis**: How different charge states appear in each sensor
- **Time Evolution**: Signal development over time for each sensor
- **Fidelity Calculation**: Performance metrics for each sensor
- **Separation Analysis**: How well different charge states can be distinguished
- **Correlation Studies**: Relationship between sensor readings

## Key Features Demonstrated

### **Resonator Diversity**
```python
# Sensor 1: Higher frequency, lower Q
params_resonator_1 = {
    'Lc': 600e-9,      # Lower inductance = higher frequency
    'Cp': 0.4e-12,     # Lower capacitance
    'RL': 30,           # Lower resistance = lower Q
}

# Sensor 2: Lower frequency, higher Q
params_resonator_2 = {
    'Lc': 1000e-9,     # Higher inductance = lower frequency
    'Cp': 0.6e-12,     # Higher capacitance
    'RL': 50,           # Higher resistance = higher Q
}
```

### **Noise Model Variation**
```python
# Sensor 1: Higher noise (more fluctuators)
eps_noise_1 = OverFNoise(n_fluctuators=8, s1=2e-3, ...)
c_noise_1 = OU_noise(sigma=2e-13, gamma=1e5)

# Sensor 2: Lower noise (fewer fluctuators)
eps_noise_2 = OverFNoise(n_fluctuators=3, s1=5e-4, ...)
c_noise_2 = OU_noise(sigma=5e-14, gamma=1e5)
```

### **Charge State Configuration**
```python
# Four possible charge configurations
charge_states = [
    np.array([0, 0]),  # Both dots empty
    np.array([1, 0]),  # Only dot 1 occupied
    np.array([0, 1]),  # Only dot 2 occupied
    np.array([1, 1])   # Both dots occupied
] * samples
```

## Running the Examples

### **Prerequisites**
```bash
cd /path/to/ReadSpyn
pip install -e .
```

### **Quick Start**
```bash
cd examples

# Simple example (recommended first)
python3 simple_two_sensor.py

# Advanced example (for detailed analysis)
python3 two_sensor_system.py
```

### **Expected Output**
- **Progress bars** showing simulation progress for each sensor
- **Conductance values** for each charge state
- **Performance metrics** (fidelity for each sensor)
- **Comprehensive visualizations** (4-9 plots depending on example)
- **Detailed analysis** of sensor performance and correlations

## Understanding the Results

### **IQ Plots**
- **X-axis**: In-phase component (I)
- **Y-axis**: Quadrature component (Q)
- **Colors**: Different charge states
- **Clustering**: Well-separated clusters indicate good readout performance

### **Performance Comparison**
- **Fidelity**: Measure of readout accuracy (0-1, higher is better)
- **Sensor Differences**: How different configurations affect performance
- **Noise Impact**: Effect of different noise models on readout quality

### **Time Evolution**
- Shows signal development over time
- Demonstrates noise effects
- Illustrates integration benefits

## Customization Options

### **System Parameters**
- Modify dot-dot coupling strength
- Change dot-sensor coupling
- Adjust number of charge states

### **Sensor Configuration**
- Vary resonator frequencies
- Change Q-factors
- Modify coupling strengths
- Adjust operating points

### **Noise Models**
- Change number of fluctuators
- Adjust noise amplitudes
- Modify correlation times
- Vary noise distributions

### **Simulation Settings**
- Adjust simulation duration
- Change number of samples
- Modify signal-to-noise ratio
- Adjust operating points

## Research Applications

### **Sensor Optimization**
- Compare different resonator designs
- Optimize for specific charge states
- Balance frequency vs. Q-factor

### **Noise Characterization**
- Understand noise impact on readout
- Compare different noise models
- Optimize noise parameters

### **System Design**
- Design multi-sensor readout systems
- Optimize dot-sensor coupling
- Balance performance vs. complexity

### **Performance Analysis**
- Quantify readout fidelity
- Analyze charge state discrimination
- Study sensor correlations

## Troubleshooting

### **Common Issues**
1. **Import Errors**: Ensure ReadSpyn is installed
2. **Memory Issues**: Reduce simulation duration or samples
3. **Slow Performance**: Check Numba installation and reduce parameters
4. **Visualization Issues**: Verify simulation completion

### **Performance Tips**
1. **Start Small**: Begin with simple examples
2. **Monitor Progress**: Watch progress bars during simulation
3. **Check Parameters**: Verify conductance values are reasonable
4. **Adjust Settings**: Modify parameters to see their effects

## Next Steps

After running the examples:

1. **Experiment**: Modify parameters to see their effects
2. **Scale Up**: Increase system size or simulation duration
3. **Customize**: Adapt examples for your specific research needs
4. **Analyze**: Use the analysis functions to understand your results
5. **Extend**: Create new examples for different system configurations

## Support and Contributions

- **Documentation**: Check README.md and API_REFERENCE.md
- **Examples**: Use these examples as templates
- **Issues**: Report problems on GitHub
- **Contributions**: Add new examples or improve existing ones
- **Contact**: j.a.krzywda@liacs.leidenuniv.nl

---

## Summary

These two-sensor examples demonstrate ReadSpyn's capability to simulate complex quantum dot readout systems with multiple sensors. They provide:

✅ **Working examples** that run successfully  
✅ **Comprehensive analysis** of multi-sensor systems  
✅ **Educational value** for understanding quantum dot readout  
✅ **Research tools** for system design and optimization  
✅ **Extensible framework** for custom applications  

The examples showcase the power of ReadSpyn for realistic quantum dot readout simulation and provide a solid foundation for further research and development. 🚀 