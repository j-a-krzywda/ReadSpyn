#!/usr/bin/env python3
"""
Basic Test Script for ReadSpyn

This script performs basic functionality tests to ensure the package
is working correctly.

Author: Jan A. Krzywda
Email: j.a.krzywda@liacs.leidenuniv.nl
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from readout_simulator import (
            QuantumDotSystem, 
            RLC_sensor, 
            ReadoutSimulator,
            OverFNoise, 
            OU_noise
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_quantum_dot_system():
    """Test QuantumDotSystem functionality."""
    print("\nTesting QuantumDotSystem...")
    
    try:
        from readout_simulator import QuantumDotSystem
        
        # Test basic creation
        Cdd = np.array([[1, 0], [0, 1]])
        Cds = np.array([[1], [0.1]])
        dot_system = QuantumDotSystem(Cdd, Cds)
        print("✓ Basic system creation successful")
        
        # Test random system generation
        random_system = QuantumDotSystem.from_random(2, 1)
        print("✓ Random system generation successful")
        
        # Test energy offset calculation
        charge_state = np.array([1, 0])
        sensor_voltages = np.array([0])
        eps0 = 0.5
        energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, eps0)
        print("✓ Energy offset calculation successful")
        
        return True
    except Exception as e:
        print(f"✗ QuantumDotSystem test failed: {e}")
        return False

def test_noise_models():
    """Test noise model functionality."""
    print("\nTesting noise models...")
    
    try:
        from readout_simulator import OU_noise, OverFNoise
        
        # Test OU noise
        ou_noise = OU_noise(sigma=1e-12, gamma=1e6)
        value = ou_noise.update(1e-6)
        print("✓ OU noise creation and update successful")
        
        # Test 1/f noise
        f_noise = OverFNoise(n_fluctuators=3, s1=1e-3, sigma_couplings=1e-99,
                            ommax=1, ommin=0.2, dt=1, equally_dist=True)
        trajectory = f_noise.generate_trajectory(100)
        print("✓ 1/f noise creation and trajectory generation successful")
        
        return True
    except Exception as e:
        print(f"✗ Noise model test failed: {e}")
        return False

def test_sensor():
    """Test RLC sensor functionality."""
    print("\nTesting RLC sensor...")
    
    try:
        from readout_simulator import RLC_sensor, QuantumDotSystem
        
        # Create minimal system
        Cdd = np.array([[1]])
        Cds = np.array([[1]])
        dot_system = QuantumDotSystem(Cdd, Cds)
        
        # Create sensor with minimal parameters
        params_resonator = {'Lc': 800e-9, 'Cp': 0.5e-12, 'RL': 40, 'Rc': 100e6, 'Z0': 50}
        params_coulomb_peak = {'g0': 1/50/1e6, 'eps0': 0.5, 'eps_width': 1}
        
        sensor = RLC_sensor(params_resonator, params_coulomb_peak)
        print("✓ RLC sensor creation successful")
        
        return True
    except Exception as e:
        print(f"✗ RLC sensor test failed: {e}")
        return False

def test_simulator():
    """Test ReadoutSimulator functionality."""
    print("\nTesting ReadoutSimulator...")
    
    try:
        from readout_simulator import (
            QuantumDotSystem, RLC_sensor, ReadoutSimulator, 
            OverFNoise, OU_noise
        )
        
        # Create minimal system
        Cdd = np.array([[1]])
        Cds = np.array([[1]])
        dot_system = QuantumDotSystem(Cdd, Cds)
        
        # Create sensor
        params_resonator = {'Lc': 800e-9, 'Cp': 0.5e-12, 'RL': 40, 'Rc': 100e6, 'Z0': 50}
        params_coulomb_peak = {'g0': 1/50/1e6, 'eps0': 0.5, 'eps_width': 1}
        
        eps_noise = OverFNoise(n_fluctuators=1, s1=1e-99, sigma_couplings=1e-99,
                              ommax=1, ommin=0.2, dt=1, equally_dist=True)
        c_noise = OU_noise(sigma=1e-99, gamma=1e7)
        
        sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
        
        # Create simulator
        simulator = ReadoutSimulator(dot_system, [sensor])
        print("✓ Simulator creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== ReadSpyn Basic Functionality Test ===\n")
    
    tests = [
        test_imports,
        test_quantum_dot_system,
        test_noise_models,
        test_sensor,
        test_simulator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! ReadSpyn is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 