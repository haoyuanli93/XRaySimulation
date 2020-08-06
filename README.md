# XRaySimulation
This is my new simulation repo.

The reason that I am abandoning the previous repo is that
the previous one is too complicated and contains too much 
information and too many functions.

I need to simplify the repo and make it more maintainable.

In this new repo, no effort has been made to include Laue 
and forward Bragg diffraction.
Later in the future, I may include those case,
however.

# Essential To-Do list
1. Add unit test module
2. Release a stable version.

# Potential To-Do List
The following feature has not been implemented
1. Add automatic calculation of the electric susceptibility.
2. Add the bragg angle automatically.
3. Add Add Gaussian smoothing to the FWHM function.  

# Notice
### Optimization
In this repo, not all the functions are optimized.
Functions in the GPU folder are optimized to some extend.
Functions in the MultiDevice.py file are not optimized.

### Fresnal diffraction
Previously, I have explicitly considered the Fresnel diffraction. 
However, I just realized that I do not need such things because 
in my simulation, the propagation of each monochromatic components
follow the more universal Maxwell equations. 

Therefore, in this version, I remove the Fresnel diffraction functions.

Later, I might add a function to handle a pure propagation directly. 

