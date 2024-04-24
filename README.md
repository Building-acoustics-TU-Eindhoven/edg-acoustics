# DG_RoomAcoustics

## Description

DG_RoomAcoustics is an open-source software package designed for the simulation of room acoustics using the time-domain wave-based method. This software implements the nodal discontinuous Galerkin (DG) method for spatial discretization of the linear acoustic equations, integrated over time with high-order schemes such as explicit Runge-Kutta and ADER (Arbitrary high-order DERivative) integration. DG_RoomAcoustics excels in scenarios where traditional geometrical acoustics tools fall short, providing superior accuracy in capturing complex wave phenomena such as diffraction, scattering, and modal effects. The software is designed to be user-friendly, with a focus on ease of use and flexibility, allowing researchers and engineers to simulate a wide range of room acoustics problems with high accuracy and computational efficiency.

## Installation

To install DG_RoomAcoustics from GitHub repository, do:

```console
git clone https://github.com/hqwang815/DG_RoomAcoustics.git
cd edg-acoustics
python3 -m pip install .
```

## Usage

To use DG_RoomAcoustics, please refer to the examples provided in the [examples](examples) directory of the repository. These examples cover a range of scenarios to help you understand how to apply the software to your specific needs.

## License Information

This project is licensed under the Apache License Version 2.0. For more details, see the [LICENSE](LICENSE) file for details.

## Documentation

This [documentation](https://dg-roomacoustics.readthedocs.io/) is designed to help you to understand and develop DG_RoomAcoustics effectively.

## Contributing

We welcome contributions to DG_RoomAcoustics! If you would like to help improve the project, you can have a look at the [contribution guidelines](CONTRIBUTING.md).

## Author and Contact Information

**Huiqing Wang**  
Email: <h.wang6@tue.nl>  

**Artur Palha**  
Email: <A.Palha@tudelft.nl>  

## Acknowledgments

DG_RoomAcoustics uses two open-source packages:

- **[meshio](https://github.com/nschloe/meshio)**: A versatile mesh file input/output library.

- **[modepy](https://documen.tician.de/modepy/index.html)**: A library for evaluating and integrating with modal (polynomial) bases on simplices.

We would like to thank the authors of these packages for their contributions to the open-source community.

## Citation

If you use DG_RoomAcoustics in your research,  please help our scientific visibility by citing our work! Please cite the following paper: to be added.

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
