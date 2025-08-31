# ReadSpyn Examples

This directory contains example scripts demonstrating the JAX-based ReadSpyn implementation.

## Available Examples

### 1. `example_quick_test.py`
A minimal example for quick testing of the JAX-based implementation.

**Features:**
- Basic quantum dot system setup
- Simple noise models
- Quick simulation with minimal parameters
- Performance metrics

**Usage:**
```bash
python examples/example_quick_test.py
```

**Expected Output:**
- Simulation parameters and performance metrics
- Total operations and execution time
- Readout fidelity calculation

### 2. `example_jax_simulation.py`
A comprehensive example demonstrating all features of the JAX-based implementation.

**Features:**
- Full quantum dot system simulation
- Multiple charge states
- Advanced noise models
- Performance analysis
- Visualization plots
- Advanced features demonstration

**Usage:**
```bash
python examples/example_jax_simulation.py
```

**Expected Output:**
- Complete simulation workflow
- Performance summary
- Visualization plots saved as `jax_simulation_results.png`
- Advanced features demonstration

## Key Features Demonstrated

### JAX-based Implementation
- **Precomputed Noise Trajectories**: Noise is generated once and reused across all states
- **State Scanning**: Uses JAX scan for efficient processing of multiple charge states
- **Vectorized Operations**: All computations are vectorized for GPU acceleration
- **Post-processing Noise**: White noise is added after signal generation

### Performance Benefits
- **GPU Acceleration**: Compatible with JAX's GPU acceleration
- **Efficient Memory Usage**: Functional programming model reduces memory overhead
- **Scalable**: Performance scales well with number of states and realizations

### Noise Models
- **OU_noise**: Ornstein-Uhlenbeck noise with exponential autocorrelation
- **OverFNoise**: 1/f noise using multiple fluctuators
- **Precomputed Trajectories**: Efficient noise generation and reuse

## Requirements

- Python 3.9+ with JAX installed
- ReadSpyn package installed
- Matplotlib for visualization (in comprehensive example)

## Running Examples

1. **Quick Test** (recommended for first-time users):
   ```bash
   python examples/example_quick_test.py
   ```

2. **Comprehensive Example** (for full feature demonstration):
   ```bash
   python examples/example_jax_simulation.py
   ```

## Expected Performance

The examples demonstrate significant performance improvements:

- **Quick Test**: ~10,000 operations/second
- **Comprehensive Example**: ~750,000 operations/second
- **Scalable**: Performance improves with larger simulations

## Troubleshooting

If you encounter issues:

1. **JAX Installation**: Ensure JAX is properly installed for your Python version
2. **Import Errors**: Make sure the ReadSpyn package is in your Python path
3. **Memory Issues**: Reduce the number of realizations or time points for large simulations

## Customization

You can modify the examples to:

- Change quantum dot system parameters
- Adjust noise model parameters
- Modify simulation time and resolution
- Add custom analysis functions
- Integrate with your own workflows

The examples serve as templates for building your own quantum dot readout simulations using the JAX-based ReadSpyn implementation. 