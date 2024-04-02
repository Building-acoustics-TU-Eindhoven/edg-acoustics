""" This module contains the abstract base class for time integrators and the implementation of the Taylor-series time integration scheme.

The edg_acoustics.time_integration provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup time integration.
"""

from __future__ import annotations

import typing
import abc
import math
import numpy
import edg_acoustics

__all__ = ["TimeIntegrator", "TSI_TI", "CFL_Default"]

CFL_Default = 0.5
"""float: Default value of the CFL number for time integration schemes."""


class TimeIntegrator(abc.ABC):
    """Base class for time integrators.

    Args:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number

    Attributes:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]
        ],
        dtscale: float,
        CFL: float = CFL_Default,
    ):
        self.L_operator = L_operator
        self.CFL = CFL
        self.dt = CFL * dtscale

    @abc.abstractmethod
    def step_dt(
        self,
        P: None,
        Vx: None,
        Vy: None,
        Vz: None,
        BC: None,
    ):
        """Takes the pressure, velocity at the time T and evolves it to time T + dt."""
        # P0 := P(T)
        # P := P(T + dt)
        # the same for all the other variables


class TSI_TI(TimeIntegrator):
    """Class for time integrator of Taylor-series time integration scheme.

    :class:`.TSI_TI` is used to evolve the pressure and velocity at the time T to time T + dt, based on the Taylor-series time integration scheme.

    Args:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number
        Nt (int): the order of the time integration scheme.

    Attributes:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
        Nt (int): the order of the time integration scheme.
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]
        ],
        dtscale: float,
        CFL: float = CFL_Default,
        **kwargs,
    ):
        super().__init__(L_operator, dtscale, CFL)
        self.Nt = kwargs["Nt"]
        print("TSI_TI initialized.")

    def step_dt(
        self,
        P: numpy.ndarray,
        Vx: numpy.ndarray,
        Vy: numpy.ndarray,
        Vz: numpy.ndarray,
        BC: edg_acoustics.AbsorbBC,
    ):
        """Takes the pressure, velocity at the time T and evolves/updates them to time T + dt.

        Args:
            P (numpy.ndarray): the pressure at the time T, will be updated to the pressure at the time T + dt
            Vx (numpy.ndarray): the x-component of the velocity at the time T, will be updated to the x-component of the velocity at the time T + dt
            Vy (numpy.ndarray): the y-component of the velocity at the time T, will be updated to the y-component of the velocity at the time T + dt
            Vz (numpy.ndarray): the z-component of the velocity at the time T, will be updated to the z-component of the velocity at the time T + dt
            BC (edg_acoustics.AbsorbBC): the boundary condition object
        """
        # Takes the pressure, velocity, and BCvar at the time T and evolves it to time T + dt
        # BC(edg_acoustics.BoundaryCondition) object neesds to be passed as a reference since we need to access the BCpara attribute
        #   P0 contains the (high-order) derivative values
        #   P := P(T) and P(T + dt)
        # the same for all the other variables
        ##########################
        P0 = P.copy()
        Vx0 = Vx.copy()
        Vy0 = Vy.copy()
        Vz0 = Vz.copy()

        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    BC.BCvar[index]["phi"] = BC.BCvar[index]["PHI"].copy()
                elif polekey == "CP":
                    BC.BCvar[index]["kexi1"] = BC.BCvar[index]["KEXI1"].copy()
                    BC.BCvar[index]["kexi2"] = BC.BCvar[index]["KEXI2"].copy()

        ##########################
        for Tind in range(1, self.Nt + 1):
            # Compute L (L^{Tind-1} q)
            P0, Vx0, Vy0, Vz0, BC.BCvar = self.L_operator(P0, Vx0, Vy0, Vz0, BC.BCvar)

            # Add the Taylor term \frac{dt^{Tind}}{Tind!}L^{Tind}q
            Vx += self.dt**Tind / math.factorial(Tind) * Vx0
            Vy += self.dt**Tind / math.factorial(Tind) * Vy0
            Vz += self.dt**Tind / math.factorial(Tind) * Vz0
            P += self.dt**Tind / math.factorial(Tind) * P0

            for index, paras in enumerate(BC.BCpara):
                for polekey in paras:
                    if polekey == "RP":
                        BC.BCvar[index]["PHI"] += (
                            self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["phi"]
                        )
                    elif polekey == "CP":
                        BC.BCvar[index]["KEXI1"] += (
                            self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["kexi1"]
                        )
                        BC.BCvar[index]["KEXI2"] += (
                            self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["kexi2"]
                        )
