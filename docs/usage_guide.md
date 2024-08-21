# Usage Guide

## Installation and Setup

To install DG_RoomAcoustics from GitHub repository, do:

```console
git clone git@github.com:Building-acoustics-TU-Eindhoven/edg-acoustics.git
cd edg-acoustics
python3 -m pip install .
```

It's highly recommended to create a conda environment first and to do the above pip install therein, to avoid potential conflicts with other Python packages.

To use [gmsh](https://gmsh.info/) in Python, as suggested in this [issue](https://github.com/conda-forge/gmsh-feedstock/issues/30), you need to install via pip by running the following command:

```console
pip install gmsh
```

## Overview of the Simulation Workflow

To run a simulation from scratch, you will need to complete the following steps essentially, which are further detailed in the following sections:

- Build a geometry model of the room as needed for the mesh generation of the simulation, which is acheived with [gmsh](https://gmsh.info/). Depending on the complexity of the room and your familiarity with mesh generation tools, you can either create the geometry with `gmsh` directly or build a [Sketchup](https://www.sketchup.com/en/plans-and-pricing/sketchup-free) model and then export it to gmsh. More details on mesh generation can be found in [Mesh Generation](#mesh-generation) section below.
  
- Fit the impedance data of the boundary materials to the time-domain impedance boundary condition (TDIBC) model [Wang2020]. A Matlab script named `CoeffByVF.m` is provided in the [examples/material_fit](../examples/material_fit) directory to fit the impedance data to the TDIBC model, which uses the Vector Fitting method [Gustaven1999](https://www.sintef.no/en/software/vector-fitting/).

- Fill out a setup script to configure the model parameters, including the geometry/mesh file, the medium properties, the boundary conditions parameters, the source and receiver locations, the upper frequency limit of the source signal and the simulation duration. Examples of the setup script of different scenarios can be found in the [examples](../examples) directory.

## Mesh Generation

The mesh generation is a crucial step in the simulation process, as it determines the spatial discretization of the room geometry. The mesh should be fine enough to capture the details of the room geometry and the wave phenomena accurately, while avoiding excessive computational cost. The mesh generation is typically done with the mesh generation tool `gmsh`, which is a powerful open-source mesh generation tool with a built-in CAD engine and post-processor. So far, there are two ways available to generate the mesh for the room geometry, and two scenarios are provided as examples in the [examples](../examples) directory respectively.

### Directly within gmsh

It's highly recommended to create the geometry of the room directly in `gmsh`. This approach gives you full control over the geometry and mesh generation process (e.g., the mesh size, meshing algorithm, locally refinement). You can find more information on how to use `gmsh` in the [gmsh documentation](https://gmsh.info/#Documentation) and an active user [community](https://gitlab.onelab.info/gmsh/gmsh/-/issues).
  
Please note that the mesh file should be saved in the legacy **2.2** format (straightforward, human-readable structure and easy to parse), whicåh is compatible with the `meshio` library used in the code and has been tested to work well with the current version.
  
Regarding the assignment of boundary conditions, the boundary faces should be tagged in `gmsh` with the corresponding boundary condition labels, which will be used in the setup script to match the parameters of boundary conditions. This can be done by creating physical groups in `gmsh` and assigning an integer number labels to these groups. For example, in the provided sceneario1 example, the `.geo` file [scenario1_coarse.geo](../examples/scenario1/scenario1_coarse.geo) contains the following lines (line 102-104) to assign the physical surface labels:

```markdown
Physical Surface(11) = { 1, 3, 4, 6, 8, 9, 10, 11 }; //hard surface
Physical Surface(13) = { 2 }; //carpet on the floor
Physical Surface(14) = { 5, 7, 12 }; //panel on the wall
```

where the integer numbers 11, 13, 14 are the boundary condition labels for the corresponding physical surfaces, and the comments are the auxiliary descriptions of the boundary conditions. These integer numbers should be consistent with the boundary condition labels defined in the setup script. The integer numbers in the curly braces are the elementary entities (faces) that belong to the physical groups.

In the setup script [examples/scenario1/main.py](../examples/scenario1/main.py), the generated `.msh` file ([scenario1_coarse.msh](../examples/scenario1/scenario1_coarse.msh) ) is loaded as input. Therein, the varibles of boundary condition labels *BC_labels* are implemented as dictionary, whose keys are the boundary condition names and values are supposed to match the physical surface numbers in the `.geo` file shown above. For example, the boundary condition labels for the scenario1 are defined as follows:
  
```python
BC_labels = {
    "hard wall": 11,
    "carpet": 13,
    "panel": 14,
}  
```

### Exporting from Sketchup and calling gmsh Python API

Another way to generate the mesh is to create the geometry in Sketchup and then export it to `gmsh` for mesh generation. This approach is more user-friendly for those who are not familiar with `gmsh` or prefer to use Sketchup for geometry modeling. The Sketchup model can be exported to `gmsh` using the `MeshKit` plugin, which is available in the [Sketchup Extension Warehouse](https://extensions.sketchup.com/extension/4bea8106-6de2-4fa9-9037-ec5b6c0a44e7/mesh-kit). An example of how to export a Sketchup model to `gmsh` can be found in the [Inputs/Geometry](https://building-acoustics-tu-eindhoven.github.io/Diffusion/Finite%20Volume%20Method%20Use.html) section of the tutorial.

The exported `.geo` file can be read by DGRoomAcoustics and the Python API of `gmsh` is called to generate the mesh. Different from the above mesh generation method, the boundary physical surfaces from the exported `.geo` file from Sketchup have string values and looks like the following, as shown in the provided sceneario2 example `.geo` file [Corridor.geo](../examples/scenario2/Corridor.geo):

```markdown
Physical Surface("Doors") = { 1, 11, 12, 13, 17, 19, 23 };
Physical Surface("UpperWall") = { 2, 6, 8, 10, 14, 16, 20, 22, 24, 26, 27, 29, 32 };
Physical Surface("Floor") = { 3 };
Physical Surface("LowerWall") = { 4, 7, 9, 15, 18, 21, 25, 28, 30, 31, 33 };
Physical Surface("Ceiling") = { 5 };
```

In the setup script [examples/scenario2/main.py](../examples/scenario2/main.py), the exported `.geo` file ([Corridor.geo](../examples/scenario2/Corridor.geo)) is loaded as input. Therein, the varibles of boundary condition labels *BC_labels* are implemented as dictionary, whose keys are the boundary material names and values are supposed to be increasing integer numbers starting from 1. Here, the order of the keys should match the order of the the boundary material names in the exported `.geo` file,  For example, the boundary condition labels for the scenario2 are defined as follows:
  
```python
BC_labels = {
    "Doors": 1,
    "UpperWall": 2,
    "Floor": 3, 
    "LowerWall": 4,
    "Ceiling": 5,
}
```

## Material Fitting

The time-domain impedance boundary condition (TDIBC) model [Wang2020] is used to model the boundary conditions in the room acoustics simulation. The TDIBC model requires the impedance data of the boundary materials to be fitted to the model. The impedance data can be obtained from measurements or simulations, and the fitting can be done with the Vector Fitting method [Gustaven1999](https://www.sintef.no/en/software/vector-fitting/). A Matlab script named `CoeffByVF.m` is provided in the [examples/material_fit](../examples/material_fit) directory to fit the impedance data to the TDIBC model. This script reads the impedance data as input, fits the data to the TDIBC model, and saves the fitted coefficients to a `.mat` file in the name of the boundary material. An example of Miki model is provided.

The fitted coefficients can then be used in the setup script to define the boundary conditions. Save the fitted coefficients in the same directory as the setup script and save it with the same name for the first couple of letters as the *BC_labels* of boundary material in the setup script.

In the example scenario1 folder [examples/scenario1](../examples/scenario1), the fitted coefficients for the carpet boundary material are saved in the file `carpet_N5_Fmax3000.mat`, where the first couple of letters *carpet* match the *BC_labels* of the carpet boundary material in the setup script. Similarly, the fitted coefficients for the panel boundary material in the scenario1 are saved in the file `panel_N5_Fmax3000.mat`, where the first couple of letters `panel` match the *BC_labels* of the panel boundary material in the setup script.

The setup script [examples/scenario1/main.py](../examples/scenario1/main.py) reads the fitted coefficients from the `.mat` files and assigns them to the corresponding boundary materials. If there is boundary material that has frequency-independent real-valued impedance, the real part of the impedance can be directly assigned to the boundary material in the setup script. For example, the setup script for the *scenario1* assigns the fitted coefficients to the carpet and panel boundary materials as follows:

```python
real_valued_impedance_boundary = [
    {"label": 11, "RI": 1}
]  # extra labels for real-valued impedance boundary condition, if needed. The label should be the similar to the label in BC_labels. Since it's frequency-independent, only "RI", the real-valued reflection coefficient, is required. If not needed, just clear the elements of this list and keep the empty list.
```

If there is no frequency-independent real-valued impedance boundary condition, the list should be empty.

The loaded fitting parameters are saved in the variable *BC_para* in the setup script, which is a of list of dictionaries. Instead of loading the fitting parameters from the `.mat` files, you can also directly assign the fitting parameters to the boundary materials in the setup script. Each dictionary contains the fitting parameters of a boundary material. More explanatory details of the data structure of *BC_para* can be found by searching "`BCpara`" in API Reference on the left side. As an example, in the setup script [examples/scenario2/main.py](../examples/scenario2/main.py),  the fitted coefficients are assigned to the boundary materials as follows:

```python
BC_para = [
    {"label": 1, "RI": 0.9999},
    {
        "label": 2,
        "RI": 0,
        "RP": numpy.array(
            [
                [2.849308439512733e03, 1.849308439512733e03],
                [2.843988875912554e03, 3.843988875912554e03],
            ]
        ),
        "CP": numpy.array([[-5.0, 10.0], [100.0, 3.0], [1.8e3, 1.3e3], [2.0e3, 4.0e3]]),
    },
    {
        "label": 3,
        "RI": 0,
        "RP": numpy.array(
            [
                [1.505778842079319e04, 1.805778842079319e04],
                [1.509502512409186e04, 2.509502512409186e04],
            ]
        ),
        "CP": numpy.array([[-5.0], [100.0], [1.8e3], [2.0e3]]),
    },
    {"label": 4, "RI": 0.95},
    {"label": 5, "RI": 0.9},
]
```

## Running the Simulation

After the mesh generation and material fitting are completed, you can run the simulation by executing the setup script. The setup script should be filled out with the necessary parameters, including the geometry/mesh file, the medium properties, the boundary conditions parameters, the source and receiver locations, the upper frequency limit of the source signal, and the simulation duration. Besides, there is an option to choose the polynomial order of the DG method, the time integration scheme.

Due to the extremely high sampling frequency needed for the time-domain simulation, the simulation results can be saved every N steps to reduce the file size. The simulation results can be saved in the `.mat` format, which is compatible with Matlab, or in the `.npz` format, which is compatible with Python. The temporary results can be saved every N steps during the simulation, which can be used to monitor the simulation process and to check the intermediate results. These can be set in the setup script as follows:

```python
save_every_Nstep = 10  # save the results every N steps
temporary_save_Nstep = 500  # save the results every N steps temporarily during the simulation. The temporary results will be saved in the root directory of this repo.
```

## Post-Processing

The current version of DG_RoomAcoustics does not provide extensive post-processing tools for acoustic parameter calculations. However, the simulated impulse response (IR) can be post-processed with other tools, such as Matlab or Python, to calculate the acoustic parameters of interest, such as the sound pressure level (SPL), the reverberation time (RT), the early decay time (EDT), the clarity (C80), and the definition (D50). The simulation results can be loaded from the saved `.mat` or `.npz` files, and the acoustic parameters can be calculated based on the sound pressure data at the receiver locations. The calculation of these acoustic parameters is beyond the scope of this documentation, but there are many resources available online that provide guidance on how to calculate these parameters.

In the current output results, there are two impulse responses saved in the output file, one is the *IR_Uncorrected*, and the other is the *IR*. The *IR_Uncorrected* is the impulse response without the correction of the Gaussian pulse, and it's the original simulation results from the time-domain DG simulation. The *IR* is obtained by correcting the source spectrum of the initial Gaussian pulse. The *IR* is the impulse response that should be used for further post-processing and acoustic parameter calculations. Details of the correction of the Gaussian pulse can be found in the paper [Wittebol2024].

## References

[Wang2020] Wang, H., & Hornikx, M. (2020). Time-domain impedance boundary condition modeling with the discontinuous Galerkin method for room acoustics simulations. The Journal of the Acoustical Society of America, 147(4), 2534-2546.

[Gustaven1999] Gustavsen, B., & Semlyen, A. (1999). Rational approximation of frequency domain responses by vector fitting. IEEE Transactions on power delivery, 14(3), 1052-1061.

[Wittebol2024] Wittebol, W., Wang, H., Hornikx, M., & Calamia, P. (2024). A hybrid room acoustic modeling approach combining image source, acoustic diffusion equation, and time-domain discontinuous Galerkin methods. Applied Acoustics, 223, 110068.
