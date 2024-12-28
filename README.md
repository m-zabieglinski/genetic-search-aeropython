### Usage of genetic algorithm search in aerodynamics as a means of decreasing aircraft development costs in the aviation industry
This repository contains the files that comprise my master's thesis, which was submitted at the Warsaw School of Economics.

### Abstract
The purpose of this work was to explore the possibility of implementing genetic algorithms in aircraft design, as a means of reducing development costs and times. A brief summary of the aviation industry was given, which concluded with the competition in the aircraft manufacturer sector and the increasing aircraft development prices. Potential steady 2D flow was introduced as an aerodynamic model and the panel method for an aerofoil as an example of computational method in aerodynamics. The genetic algorithm (GA) was introduced and described. An application of GA tools towards aerofoils was proposed. In Python, a computing environment was created to obtain solutions for aerofoils in potential steady 2D flow, based off [AeroPython](https://github.com/barbagroup/AeroPython). NACA0012 aerofoil was mutated to seed the first generation of solutions for GA. Multiple iterations of preliminary searches, including 2 parameter sweeps, were conducted with the proposed GA implementation over the aerodynamic search space. The obtained results showed the strength of GA principles and the inherent difficulties of computational fluid dynamics (CFD).

### Contents
1. dsd
2. [requirements.txt](https://github.com/m-zabieglinski/genetic-search-aeropython/blob/main/requirements.txt) - packages for the kernel I ran all my code in
3. [profiles.py](https://github.com/m-zabieglinski/genetic-search-aeropython/blob/main/profiles.py) - the source code used for generating panels, based off [AeroPython](https://github.com/barbagroup/AeroPython) with task-specific improvements
4. [gen1_generation.ipynb](https://github.com/m-zabieglinski/genetic-search-aeropython/blob/main/gen1_generation.ipynb) - the code used to create the first generation of specimen, by mutating 100 NACA0012 aeorofoils
5. `c*a*b*.ipynb` files (like [c025a1v1.ipynb](https://github.com/m-zabieglinski/genetic-search-aeropython/blob/main/c025a1b1.ipynb)) - notebooks used to conduct particular runs for a parameter sweep.
6. `shadow_*prob*.ipynb` files (like [shadow_power3_prob001.ipynb]https://github.com/m-zabieglinski/genetic-search-aeropython/blob/main/(shadow_power3_prob001.ipynb)) - notebooks used to conduct particular runs for a parameter sweep.
