# Readout Simulator

A brief description of your project.

## Installation

You can install the package using pip:

```bash
pip install readout_simulator
```

## Usage

Provide a simple example of how to use your simulator.

```python
from readout_simulator import main

# Example of how to run your simulator
main.run_simulation()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## Todo

### High priority
- [ ] Make the code work
- [ ] Fluctuations of Capacitance, w = 1/sqrt{L0 C0} (chat w Miguel -> parameters of C noise) (rotations in IQ)
- [x] Noise in R (drift, displacements in IQ spaces)


### Mid priority
- [ ] Simple analytics for understanding whats going on.
- [ ] Find expressions for "sensitivity" of the readout (e.g. how much does the readout change with a change in capacitance)

### Low priority
- [ ] Speed up
- [ ] Capacitance readout
- [ ] Geometric initialization
  - [ ] Pick positions of dots
  - [ ] Pick positions of sensors (resonator/random walker)
  - [ ] Generate Cdd Cds, tc (for capacitance readout)
- [ ] Make a game from inverse design

## Plan
1. (Thu 10) Finish HP-todo. 
2. (Thu 17) Try MP-todo



### What we have:
