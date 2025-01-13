CUDA code for the paper "Calcium homeostatic feedback control predicts atrial fibrillation initiation, remodeling, and progression", by Nicolae Moise and Seth H Weinberg.

Code is setup to run a heterogeneous tissue simulation (i.e. Figure 3-6).
Due to length of simulations, we run a quarter of the simulation at a time, then load data for next quarter.

There are further options and flags to replicate the homogeneous cases.

Simulation walltimes:

#10s A100 times: 1024x1024: 43s 512x512: 13s   5000s: 1024 = 06:30:00, 512 = 02:00:00
#38.5hrs for 25000s sim
#38.5hrs for 20000s sim 9pt --array=1-7,9-25  --array=8

NVIDIA A100: 100s run time ~ 600s wall time
NVIDIA H100: 100s run time ~ 36s wall time
