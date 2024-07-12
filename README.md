==========================================================================================
Implementation for the Paper: Precomputing approach for a two-scale phase transition model
==========================================================================================

This repository contains the source codes for all simulations presented in the `paper`_.

To run the code, the following packages are mandatory:

- `FEniCS`_
- `Gmsh`_
- `SciPy`_
- `meshio`_

An example environment is provided in the ``environment.yml``.

.. _`paper`: ...
.. _`FEniCS`: https://fenicsproject.org/
.. _`Gmsh`: https://pypi.org/project/gmsh/
.. _`SciPy`: https://scipy.org/install/
.. _`meshio`: https://pypi.org/project/meshio/


Usage:
======

The main code is shown in ``Main.py``, to run the file you have to use MPI since
the cell problems are expected to be distributed over multiple processes.
To run the code with MPI using 8 processes is possible with the following command:
```
mpirun -n 8 path_to_python path_to_main.py
```

The remaining strucutre of this repository is as follows:
- `MeshCreation`: Contains all used micro meshes in the paper and code for creating them.
- `Interpolation`: Contains the precomputed effective conductivity values for different resolutions.
- `Utils` and `MicroProblems`: Contain code that implements the complete problem, micro problems and 
 coupling of multiple processes.
- All Python files with `_test.py` implement the convergence studies of the paper.

Example Solution
================

![](https://github.com/TomF98/Precomputing-approach-for-a-two-scale-phase-transition-mode/blob/master/Utils/output.gif)