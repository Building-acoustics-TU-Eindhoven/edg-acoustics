# DG_RoomAcoustics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13619551.svg)](https://doi.org/10.5281/zenodo.13619551)

## Description

DG_RoomAcoustics is an open-source software package designed for the simulation of room acoustics using the time-domain wave-based method. This software implements the nodal discontinuous Galerkin (DG) method for spatial discretization of the linear acoustic equations, integrated over time with high-order schemes such as explicit ADER (Arbitrary high-order DERivative) integration. DG_RoomAcoustics excels in scenarios where traditional geometrical acoustics tools fall short, providing superior accuracy in capturing complex wave phenomena such as diffraction, scattering, and modal effects. Following the principles of object-oriented programming paradigm, the software is designed with a focus on ease of use and flexibility, allowing researchers and engineers to simulate a wide range of room acoustics problems with high accuracy and computational efficiency.

## Installation

To install DG_RoomAcoustics from GitHub repository, do:

```console
git clone git@github.com:Building-acoustics-TU-Eindhoven/edg-acoustics.git
cd edg-acoustics
python3 -m pip install .
```

## Usage & Documentation

This [documentation](https://dg-roomacoustics.readthedocs.io/) is created to help you to use and develop DG_RoomAcoustics effectively. To use DG_RoomAcoustics, please refer to the examples provided in the [examples](examples) directory of the repository. These examples cover a range of scenarios to help you understand how to apply the software to your specific needs.

## License Information

This project is licensed under the GNU General Public License v3.0. For more details, see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to DG_RoomAcoustics! If you would like to help improve the project, you can have a look at the [contribution guidelines](CONTRIBUTING.md).

## Author and Contact Information

**Huiqing Wang**  
Email: <h.wang6@tue.nl>  

**Artur Palha**  
Email: <A.Palha@tudelft.nl>  

## Acknowledgments

DG_RoomAcoustics uses the following open-source packages/codes/toolkits:

- **[GMSH](https://gmsh.info/)**: A powerful mesh generation tool with a built-in CAD engine and post-processor.

- **[meshio](https://github.com/nschloe/meshio)**: A versatile mesh file input/output library.

- **[modepy](https://documen.tician.de/modepy/index.html)**: A library for evaluating and integrating with modal (polynomial) bases on simplices.

- **[numpy](https://numpy.org/)**: A powerful numerical computing library for Python.

- **[Vector Fitting](https://www.sintef.no/en/software/vector-fitting/)**: A software package for fitting rational functions to frequency-domain data.

We would like to thank the authors of above packages for their contributions to the open-source community.

## Citation

If you use DG_RoomAcoustics in your research,  please help our scientific visibility by citing our work:

```bibtex
@software{wang_2024_13619551,
  author       = {Wang, Huiqing and
                  Palha, Artur},
  title        = {DG\_RoomAcoustics},
  month        = aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0-alpha.1},
  doi          = {10.5281/zenodo.13619551},
  url          = {https://doi.org/10.5281/zenodo.13619551}
}
```

The software design is described in the following paper:

```bibtex
@inproceedings{wang2024open,
  title={An open-source time-domain wave-based room acoustic software in Python based on the nodal discontinuous Galerkin method},
  author={Wang, Huiqing and Palha, Artur and Hornikx, Maarten},
  booktitle={INTER-NOISE and NOISE-CON Congress and Conference Proceedings},
  volume={270},
  number={5},
  pages={6770--6777},
  year={2024},
  doi={https://doi.org/10.3397/IN_2024_3868},
  organization={Institute of Noise Control Engineering}
}
```

## Funding Information

This work was supported by the Dutch Research Council (NWO) under the project "[A new era of room acoustics simulation software: from academic advances to a sustainable open source project and community](https://www.cursor.tue.nl/en/news/2022/februari/week-4/nwo-subsidy-for-open-source-project-on-room-acoustics/)" (project number 19430). The authors would also like to acknowledge the support of eScience center of Netherlands under the grant of "Small-Scale Initiatives in Software Performance Optimization" ([OpenSSI 2021b](https://www.esciencecenter.nl/news/researchers-to-benefit-from-cutting-edge-research-software-in-25-newly-awarded-projects/))

## Development Plan

### Performance Optimization

- Implement high-performance computing techniques to accelerate simulations, particularly focusing on GPU acceleration.

### User Interface Development

- Develop a more user-friendly interface to simplify the setup and execution of simulation scenarios.

### Feature Extensions

- Integrate additional acoustic modeling features based on community feedback, including advanced boundary condition models and post-processing tools for acoustic parameter calculations.

### Documentation Expansion

- Enhance documentation and examples to provide more comprehensive guidance and support to new users.

## Credits

This package was firstly created following [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template). Later modifications were made by the authors of the project.
