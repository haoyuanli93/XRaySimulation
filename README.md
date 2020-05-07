# XRaySimulation
This is my new simulation repo.

The reason that I am abandoning the previous repo is that
the previous one is too complicated and contains too much 
information and too many functions.

I need to simplify the repo and make it more maintainable.

In this new repo, there will be no effort made to include 
the simulation of Laue and forward Bragg diffraction at
present. Later in the future, I may include those case,
however.

# Notice
In this repo, not all the functions are optimized.
Functions in the GPU folder are optimized to some extend.
Functions in the MultiDevice.py file are not optimized.

# To-Do List

The following feature has not been implemented

1. Add automatic calculation of the electric susceptibility.
2. Add the bragg angle automatically.
3. Add Add Gaussian smoothing to the FWHM function.  
4. Add method to automatically calculate the chi values for different crystals.

The following feature need to be improved

1. The way to calculate the bragg angle in the MultiDevice.align_devices is not 
ideal. I would like to find a better way and a more analytical way to find this 
value.
2. 