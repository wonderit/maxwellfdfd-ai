MaxwellFDFD
===========
MaxwellFDFD is a MATLAB-based solver package of Maxwell's equations.  It solves the equations by the finite-difference frequency-domain (FDFD) method, and hence the name MaxwellFDFD.

See `INSTALL.md` for installation instruction.

See `doc/index.html` in MATLAB or any web browser for more detailed introduction and usage.

MaxwellFDFD-ai
===========
MaxwellFDFD-ai is a modified version to automatically generate random data from MaxwellFDFD.

When installation is done, open maxwellfdfd folder with Matlab. 

- Generate random design and calculate transmitted power
    - example/ai/generate_simple_design.m
    - example/ai/generate_random_length_bottom_fixed.m
    
- Simulator input image
    - example/ai/image_input_simulator.m
