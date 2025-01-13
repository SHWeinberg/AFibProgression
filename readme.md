CUDA code for the paper "Calcium homeostatic feedback control predicts atrial fibrillation initiation, remodeling, and progression", by Nicolae Moise and Seth H Weinberg.

Code is setup to run a heterogeneous tissue simulation (i.e. Figure 3-6).
Due to length of simulations, we run a quarter of the simulation at a time, then load data for next quarter.

There are further options and flags to replicate the homogeneous cases.

Simulation walltimes:

NVIDIA A100: 100s run time ~ 600s wall time
NVIDIA H100: 100s run time ~ 360s wall time
