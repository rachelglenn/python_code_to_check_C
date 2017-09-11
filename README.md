# python_code_to_check_C

This is a time-dependent Coupled Cluster (CCSD or CC2) calculator. 
The converged CCSD and CC2 values have been checked against psi4. 
All individual T1, T2, L1, and L2 intermediate terms for CCSD and CC2 have been checked against code pulled from psi4. 
The Fourier transform of the time-dependent simulation has peaks that correspond to an 
EOM-CC2 oscillator_strength calculation. 

## The syntax to run the simulation is:

```
python H2O.py <time_to_run>
```

where <time_to_run> is the time-length in seconds that the simulation will run for. I recommend using

```
python H2O.py <time_to_run> > <output.dat>
```

where <output.dat> is a file, that contains the psi4-output and also a copy of the time-dependent dipole moment.

## Files output from the simulation are:

H2O.csv          -The time-evolution of the dipole moment.

Parameters.dat   -The parameters needed for the next simulation to start from the end of last one.

timing.csv       -The recorded time to evaluate the right-hand-side of T1, T2, L1, and L2 (including cchbar).

Output files used to restart the simultion from the last point are:

F_real.dat

F_imag.dat

t1_real.dat

t1_imag.dat

t2_real.dat

t2_imag.dat

lam1_real.dat

lam1_imag.dat

lam2_real.dat

lam2_imag.dat

## Parameters:

The Electric Field Function is defined inside the Runge_Kutta defined function. 

w0 - Frequency of the Electric Field.

A  - Amplitude of the Electric Field.

dt - Time-step of the numeric solver.

t0 - Start-time of the simulation.

precs - The accuracy of the numerical solver.

## Switching between CCSD and CC2 for a simulation at t=0:

The instance for either a CCSD or CC2 calculation is:

```
mol= CC_Calculator(psi4, w0, A, t0, dt, precs)
```

A CCSD time-dependent calculation is initiated by using the following command:

```
mol.TDCCSD(<time_to_run>)
```

and a CC2 timed-dependent calculation is initiated by using the following command:

```
mol.TDCC2(<time_to_run>)
```

## Restarting the simulation from time, t != 0:

See Restart_H2O.py for an example restart of the simulation. Provide the same molecular input to psi4 as with the previous simulation. This will allow the restart program to get the two-electron integrals from psi4. A restart CCSD time-dependent calculation is initiated by using the following command:

```
mol = CC_Calculator(psi4)
mol.TDCC_restart(<time_to_run>);
```
There is no need to specify whether it is a CCSD or CC2 calculation. It will automatically determine that from the saved output files of the previous simulation.

## Links:
* [C++ Time-dependent CCSD/CC2 code](https://github.com/rachelglenn/tdccsd)

