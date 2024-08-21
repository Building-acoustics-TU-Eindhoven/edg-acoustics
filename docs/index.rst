.. DG_RoomAcoustics documentation master file, created by sphinx-quickstart 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DG_RoomAcoustics's documentation!
============================================

DG_RoomAcoustics is an open-source implementation of a time-domain wave-based room acoustic modeling software package, which includes a set of basic and essential tools for simulation setup and processing. In this software, the linear acoustic equations are spatially discretized by the nodal discontinuous Galerkin method, and are integrated in time by either the explicit the arbitrary high-order derivatives (ADER) integration schemes or Runge-Kutta scheme (under development). Following the principles of object-oriented programming paradigm, the software is structured to ensure generic applicability and to facilitate future extensions with additional functionalities (e.g., different time integration schemes, boundary conditions).

.. toctree::
  :maxdepth: 2
  :caption: Usage guide
  
  usage_guide.md

.. toctree::
  :maxdepth: 2
  :caption: Source code documentation from docstrings:


Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. .. figure:: /diagram.png
..   :width: 400px
..   :align: center
..   :alt: UML Diagram of the code

..   UML diagram of the code