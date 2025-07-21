# Readout Simulator

<img src="icon.png" alt="ReadSpyn Icon" width="250" />

A brief description of your project.

Link to the Overleaf document: https://www.overleaf.com/4595747995vjmghjrhsrhh#09deae

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
- [x] Make the code work
- [x] Fluctuations of Capacitance, w = 1/sqrt{L0 C0} (chat w Miguel -> parameters of C noise) (rotations in IQ? No!)
- [x] Noise in R (drift, displacements in IQ spaces)


### Mid priority
- [ ] Simple analytics for understanding whats going on.
- [ ] Find expressions for "sensitivity" of the readout (e.g. how much does the readout change with a change in capacitance)
- [ ] Simple numerics to understand how noise is affecting the raw signal and IQ

### Low priority
- [ ] Speed up
- [ ] Capacitance readout
- [ ] Geometric initialization
  - [ ] Pick positions of dots
  - [ ] Pick positions of sensors (resonator/random walker)
  - [ ] Generate Cdd Cds, tc (for capacitance readout)
- [ ] Make a game from inverse design
- [ ] Implement relative readout

## Plan
1. (Thu 25) Try MP-todo



### What we have:
- Noise in R can introduce different noise amplitudes for different states
- We can deform the blobs into elipse, by including charge noise 
- Two cases to compare eigenstate vs superposeation
- Shot-noise vs Fidelity, achivable fidelity in fixed time. 

What to test:
1. Adaptive 
   1. for white-noise simulator
   2. charge noise
   3. + capacitance noise
2. Characterise 1.1, 1.2, 1.3 in supplementary mat form

9§  



# TODO over weekend.
- Raw signal
- 2 dots
- 2 sensors
- Change parameters of sensors/QD 
- Clean the code a bit, 
- Adjust parameters to get good signal in approx 5us. 