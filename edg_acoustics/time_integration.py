"""
``edg_acoustics.time_integration``
======================

The edg_acoustics time_integration ***** UPDATE ****  provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) 
to setup initial condition for a specific scenario.

Functions and classes present in :mod:`edg_acoustics.initial_condition` are listed below.

Setup Initial Condition
---------------
    InitialCondition
"""

from __future__ import annotations

import typing
import abc
import math
import numpy
import edg_acoustics

__all__ = ["TimeIntegrator", "CFL_Default"]

CFL_Default = 0.5


class TimeIntegrator(abc.ABC):
    """
    Abstract base class for time integrators.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def step_dt_new(
        self,
        P: None,
        Vx: None,
        Vy: None,
        Vz: None,
        BC: None,
    ):
        """
        Takes the pressure, velocity at the time T and evolves it to time T + dt.
        """
        # P0 := P(T)
        # P := P(T + dt)
        # the same for all the other variables


class TSI_TI(TimeIntegrator):
    """
    class for time integrator of Taylor-series.
    """

    def __init__(
        self,
        L_operator: typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]],
        dtscale: float,
        Nt: int,
        CFL: float = CFL_Default,
    ):
        # Nt (int): the order of the time integration scheme.

        self.L_operator = L_operator  # the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        self.Nt = Nt  # degree of time integration
        self.CFL = CFL
        self.dt = CFL * dtscale

    def step_dt_new(
        self,
        P: numpy.ndarray,
        Vx: numpy.ndarray,
        Vy: numpy.ndarray,
        Vz: numpy.ndarray,
        BC: edg_acoustics.AbsorbBC,
    ):
        """
        Takes the pressure, velocity at the time T and evolves it to time T + dt.
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
        print(f"inside, P0 ID {id(P0)}, P ID {id(P)}")  # used for debugging

        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    BC.BCvar[index]["phi"] = BC.BCvar[index]["PHI"].copy()
                elif polekey == "CP":
                    BC.BCvar[index]["kexi1"] = BC.BCvar[index]["KEXI1"].copy()
                    BC.BCvar[index]["kexi2"] = BC.BCvar[index]["KEXI2"].copy()

        ##########################
        # for Tind in range(1, self.Nt + 1):
        #     # Compute L (L^{Tind-1} q)
        #     P0, Vx0, Vy0, Vz0, BC.BCvar = self.L_operator(P0, Vx0, Vy0, Vz0, BC.BCvar)

        #     # Add the Taylor term \frac{dt^{Tind}}{Tind!}L^{Tind}q
        #     Vx += self.dt**Tind / math.factorial(Tind) * Vx0
        #     Vy += self.dt**Tind / math.factorial(Tind) * Vy0
        #     Vz += self.dt**Tind / math.factorial(Tind) * Vz0
        #     P += self.dt**Tind / math.factorial(Tind) * P0

        #     for index, paras in enumerate(BC.BCpara):
        #         for polekey in paras:
        #             if polekey == "RP":
        #                 BC.BCvar[index]["PHI"] += (
        #                     self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["phi"]
        #                 )
        #             elif polekey == "CP":
        #                 BC.BCvar[index]["KEXI1"] += (
        #                     self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["kexi1"]
        #                 )
        #                 BC.BCvar[index]["KEXI2"] += (
        #                     self.dt**Tind / math.factorial(Tind) * BC.BCvar[index]["kexi2"]
        #                 )

        for Tind in range(1, self.Nt + 1):
            # Compute L (L^{Tind-1} q)
            P0, Vx0, Vy0, Vz0, BC.BCvar = self.L_operator(P0, Vx0, Vy0, Vz0, BC.BCvar)

            print(f"check before, P0 ID {id(P0)}")  # used for debugging

            P0 *= self.dt / Tind
            Vx0 *= self.dt / Tind
            Vy0 *= self.dt / Tind
            Vz0 *= self.dt / Tind
            print(f"check after, P0 ID {id(P0)}")  # used for debugging

            P += P0
            Vx += Vx0
            Vy += Vy0
            Vz += Vz0

            for index, paras in enumerate(BC.BCpara):
                for polekey in paras:
                    if polekey == "RP":
                        BC.BCvar[index]["phi"] *= self.dt / Tind
                        BC.BCvar[index]["PHI"] += BC.BCvar[index]["phi"]
                    elif polekey == "CP":
                        BC.BCvar[index]["kexi1"] *= self.dt / Tind
                        BC.BCvar[index]["KEXI1"] += BC.BCvar[index]["kexi1"]
                        BC.BCvar[index]["kexi2"] *= self.dt / Tind
                        BC.BCvar[index]["KEXI2"] += BC.BCvar[index]["kexi2"]
