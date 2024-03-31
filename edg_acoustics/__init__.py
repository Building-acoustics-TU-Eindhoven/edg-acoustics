"""Documentation about edg_acoustics"""

import logging
from .acoustics_simulation import AcousticsSimulation
from .mesh import Mesh
from .boundary_condition import BoundaryCondition, AbsorbBC
from .initial_condition import InitialCondition, Monopole_IC
from .preprocessing import UpwindFlux
from .time_integration import TimeIntegrator, TSI_TI


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Huiqing Wang, Artur Palha"
__email__ = "h.wang6@tue.nl, a.palha@esciencecenter.nl"
__version__ = "0.1.0"
