Implementation for the Paper: Precomputing approach for a two-scale phase transition model
==========================================================================================

This repository contains the source codes for all simulations presented in the [paper](https://arxiv.org/abs/2407.21595).

To run the code, the following packages are mandatory:

- [FEniCS](https://fenicsproject.org/)
- [Gmsh](https://pypi.org/project/gmsh/)
- [SciPy](https://scipy.org/install/)
- [meshio](https://pypi.org/project/meshio/)

An example environment is provided in the ``environment.yml``.


Usage:
======

The main code is shown in ``Main.py``, to run the file you will need to use MPI, since
the cell problems are expected to be distributed across multiple processes.
To run the code with MPI using 8 processes, use the following command:
```
mpirun -n 8 path_to_python path_to_Main.py
```

The rest of the structure of this repository is as follows:
- `MeshCreation`: Contains all the micro meshes used in the paper and the code to create them.
- `InterpolationData`: Contains the precomputed effective conductivity values for different resolutions.
- `Utils` and `MicroProblems`: Contains code that implements the complete problem, micro-problems and 
 coupling of multiple processes.
- All Python files with `_test.py` implement the convergence studies of the paper.

Example Solution
================

![](https://github.com/TomF98/Precomputing-approach-two-scale-phase-transition-model/blob/master/Utils/output.gif)