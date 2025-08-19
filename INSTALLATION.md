# ReadSpyn Installation Guide

This guide provides step-by-step instructions for installing and setting up ReadSpyn on your system.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: Python 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended for large simulations)
- **Storage**: At least 1GB free disk space

### Python Dependencies

ReadSpyn requires the following Python packages:
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0
- numba >= 0.56.0

## Installation Methods

### Method 1: Install from Source (Recommended)

This method gives you the latest development version and allows you to modify the code.

```bash
# Clone the repository
git clone https://github.com/jan-a-krzywda/ReadSpyn.git
cd ReadSpyn

# Create a virtual environment (recommended)
python -m venv readspyn_env
source readspyn_env/bin/activate  # On Windows: readspyn_env\Scripts\activate

# Install in development mode
pip install -e .
```

### Method 2: Install Dependencies Only

If you want to use ReadSpyn without installing it as a package:

```bash
# Clone the repository
git clone https://github.com/jan-a-krzywda/ReadSpyn.git
cd ReadSpyn

# Create virtual environment
python -m venv readspyn_env
source readspyn_env/bin/activate  # On Windows: readspyn_env\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib tqdm numba
```

### Method 3: Using Conda (Alternative)

If you prefer using Conda:

```bash
# Create conda environment
conda create -n readspyn python=3.9
conda activate readspyn

# Install dependencies
conda install numpy scipy matplotlib tqdm
conda install -c conda-forge numba

# Clone and install ReadSpyn
git clone https://github.com/jan-a-krzywda/ReadSpyn.git
cd ReadSpyn
pip install -e .
```

## Verification

After installation, verify that ReadSpyn is working correctly:

```bash
# Run the basic test script
python test_basic.py
```

You should see output similar to:
```
=== ReadSpyn Basic Functionality Test ===

Testing imports...
✓ All modules imported successfully

Testing QuantumDotSystem...
✓ Basic system creation successful
✓ Random system generation successful
✓ Energy offset calculation successful

Testing noise models...
✓ OU noise creation and update successful
✓ 1/f noise creation and trajectory generation successful

Testing RLC sensor...
✓ RLC sensor creation successful

Testing ReadoutSimulator...
✓ Simulator creation successful

=== Test Results ===
Passed: 5/5
🎉 All tests passed! ReadSpyn is working correctly.
```

## Quick Start

Once installed, you can start using ReadSpyn:

```python
# Basic import
from readout_simulator import QuantumDotSystem, RLC_sensor, ReadoutSimulator

# Create a simple system
Cdd = np.array([[1, 0], [0, 1]])
Cds = np.array([[1], [0.1]])
dot_system = QuantumDotSystem(Cdd, Cds)

print(f"Created system: {dot_system}")
```

## Troubleshooting

### Common Installation Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'readout_simulator'`

**Solution**: Ensure you're in the correct directory and have installed the package:
```bash
cd ReadSpyn
pip install -e .
```

#### 2. Numba Installation Issues

**Problem**: Numba fails to install or compile

**Solution**: Try installing Numba separately first:
```bash
pip install numba
# or
conda install -c conda-forge numba
```

#### 3. Permission Errors

**Problem**: Permission denied when installing packages

**Solution**: Use a virtual environment or add `--user` flag:
```bash
pip install --user -e .
```

#### 4. Python Version Issues

**Problem**: Incompatible Python version

**Solution**: Ensure you're using Python 3.8 or higher:
```bash
python --version
```

### Performance Issues

#### 1. Slow Simulations

**Problem**: Simulations run very slowly

**Solutions**:
- Ensure Numba is properly installed and working
- Reduce simulation parameters (fewer samples, shorter duration)
- Use smaller quantum dot systems for testing

#### 2. Memory Issues

**Problem**: Out of memory errors

**Solutions**:
- Reduce the number of charge states or simulation duration
- Process data in smaller chunks
- Close other applications to free up memory

### Platform-Specific Issues

#### Windows

- Use Windows Subsystem for Linux (WSL) for better performance
- Ensure Microsoft Visual C++ Build Tools are installed
- Use Anaconda/Miniconda for easier dependency management

#### macOS

- Ensure Xcode Command Line Tools are installed
- Use Homebrew for system dependencies if needed

#### Linux

- Install development tools: `sudo apt-get install build-essential` (Ubuntu/Debian)
- Ensure Python development headers are available

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the Issues**: Search existing issues on GitHub
2. **Create New Issue**: Provide detailed error messages and system information
3. **Contact Author**: Email j.a.krzywda@liacs.leidenuniv.nl

## Development Setup

For developers who want to contribute to ReadSpyn:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## Uninstallation

To remove ReadSpyn:

```bash
# If installed with pip
pip uninstall ReadSpyn

# Remove the cloned repository
rm -rf ReadSpyn/
```

## Next Steps

After successful installation:

1. **Read the Documentation**: Check `README.md` and `API_REFERENCE.md`
2. **Run Examples**: Try the examples in the `examples/` directory
3. **Explore Features**: Experiment with different noise models and system configurations
4. **Join the Community**: Contribute to the project or report issues

Happy simulating! 🚀 