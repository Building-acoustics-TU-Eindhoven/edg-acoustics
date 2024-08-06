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

__all__ = ["TimeIntegrator", "TSI_TI", "CFL_Default", "RK45_TI"]

CFL_Default = 0.5
"""float: Default value of the CFL number for time integration schemes."""

rk4a = numpy.array(
    [
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ]
)
"""Coefficients for the low-storage five-stage fourth-order Runge-Kutta time integration scheme."""
rk4b = numpy.array(
    [
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ]
)
"""Coefficients for the low-storage five-stage fourth-order Runge-Kutta time integration scheme."""

rk4c = numpy.array(
    [
        0.0,
        1432997174477.0 / 9575080441755.0,
        2526269341429.0 / 6820363962896.0,
        2006345519317.0 / 3224310063776.0,
        2802321613138.0 / 2924317926251.0,
    ]
)
"""Coefficients for the low-storage five-stage fourth-order Runge-Kutta time integration scheme."""


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
            print(f"TSI_TI step_dt: Tind = {Tind}")
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


class RK45_TI(TimeIntegrator):
    """Class for time integrator of low-storage five-stage fourth-order Runge-Kutta time integration scheme.

    :class:`.RK45_TI` is used to evolve the pressure and velocity at the time T to time T + dt, based on the low-storage five-stage fourth-order Runge-Kutta time integration scheme.

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
        super().__init__(L_operator, dtscale, CFL)
        print("5-stage 4th-order Runge-Kutta time integration initialized.")

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

        # P0 is zero matrix, the same size as P
        P0 = numpy.zeros_like(P)
        Vx0 = numpy.zeros_like(Vx)
        Vy0 = numpy.zeros_like(Vy)
        Vz0 = numpy.zeros_like(Vz)

        # for INTRK in range(0, 5):
        #     # print(f"before, BCvar ID {id(BC.BCvar)}")
        #     # print(f"before, BCvartemp ID {id(BCvarTemp)}")
        #     # print(f"before, P ID {id(P)}, should be the same as P outside the loop")

        #     BCvarTemp = BC.BCvar.copy()

        #     # for index, paras in enumerate(BC.BCpara):
        #     #     for polekey in paras:
        #     #         if polekey == "RP":
        #     #             BC.BCvar[index]["PHI"] = BC.BCvar[index]["phi"].copy()
        #     #         elif polekey == "CP":
        #     #             BC.BCvar[index]["KEXI1"] = BC.BCvar[index]["kexi1"].copy()
        #     #             BC.BCvar[index]["KEXI2"] = BC.BCvar[index]["kexi2"].copy()

        #     RHS_P, RHS_Vx, RHS_Vy, RHS_Vz, RHS_BCvar = self.L_operator(P, Vx, Vy, Vz, BC.BCvar)
        #     # print(f"after, ‘P’ ID {id(P)}")
        #     # print(f"after, BCvartemp ID {id(BCvarTemp)}")
        #     # print(f"after, BC.BCvar ID {id(BC.BCvar)}")
        #     # print(f"after, RHS_BCvar ID {id(RHS_BCvar)}")

        #     P0 = rk4a[INTRK] * P0 + self.dt * RHS_P
        #     Vx0 = rk4a[INTRK] * Vx0 + self.dt * RHS_Vx
        #     Vy0 = rk4a[INTRK] * Vy0 + self.dt * RHS_Vy
        #     Vz0 = rk4a[INTRK] * Vz0 + self.dt * RHS_Vz

        #     Vx += rk4b[INTRK] * Vx0
        #     Vy += rk4b[INTRK] * Vy0
        #     Vz += rk4b[INTRK] * Vz0
        #     P += rk4b[INTRK] * P0

        #     for index, paras in enumerate(BC.BCpara):
        #         for polekey in paras:
        #             if polekey == "RP":
        #                 BC.BCvar[index]["PHI"] = (
        #                     rk4a[INTRK] * BC.BCvar[index]["PHI"] + self.dt * RHS_BCvar[index]["phi"]
        #                 )
        #                 # BC.BCvar[index]["phi"] += rk4b[INTRK] * BC.BCvar[index]["PHI"]
        #                 BC.BCvar[index]["phi"] = (
        #                     BCvarTemp[index]["phi"] + rk4b[INTRK] * BC.BCvar[index]["PHI"]
        #                 )

        #             elif polekey == "CP":
        #                 BC.BCvar[index]["KEXI1"] = (
        #                     rk4a[INTRK] * BC.BCvar[index]["KEXI1"]
        #                     + self.dt * RHS_BCvar[index]["kexi1"]
        #                 )
        #                 BC.BCvar[index]["KEXI2"] = (
        #                     rk4a[INTRK] * BC.BCvar[index]["KEXI2"]
        #                     + self.dt * RHS_BCvar[index]["kexi2"]
        #                 )

        #                 BC.BCvar[index]["kexi1"] = (
        #                     BCvarTemp[index]["kexi1"] + rk4b[INTRK] * BC.BCvar[index]["KEXI1"]
        #                 )
        #                 BC.BCvar[index]["kexi2"] = (
        #                     BCvarTemp[index]["kexi2"] + rk4b[INTRK] * BC.BCvar[index]["KEXI2"]
        #                 )
        #                 # BC.BCvar[index]["kexi1"] += rk4b[INTRK] * BC.BCvar[index]["KEXI1"]
        #                 # BC.BCvar[index]["kexi2"] += rk4b[INTRK] * BC.BCvar[index]["KEXI2"]

        ###################################################################################################################################
        for INTRK in range(0, 5):

            BCvarTemp = BC.BCvar.copy()

            # print(f"before, BCvar ID {id(BC.BCvar)}")
            # print(f"before, BCvartemp ID {id(BCvarTemp)}")
            # print(f"before, P ID {id(P)}, should be the same as P outside the loop")

            RHS_P, RHS_Vx, RHS_Vy, RHS_Vz, _ = self.L_operator(P, Vx, Vy, Vz, BCvarTemp)
            # print(f"after, BC.BCvar ID {id(BC.BCvar)}")
            # print(f"after, BCvartemp ID {id(BCvarTemp)}")

            # print(f"after, RHS_BCvar ID {id(RHS_BCvar)}")

            P0 = rk4a[INTRK] * P0 + self.dt * RHS_P
            Vx0 = rk4a[INTRK] * Vx0 + self.dt * RHS_Vx
            Vy0 = rk4a[INTRK] * Vy0 + self.dt * RHS_Vy
            Vz0 = rk4a[INTRK] * Vz0 + self.dt * RHS_Vz

            Vx += rk4b[INTRK] * Vx0
            Vy += rk4b[INTRK] * Vy0
            Vz += rk4b[INTRK] * Vz0
            P += rk4b[INTRK] * P0

            for index, paras in enumerate(BC.BCpara):
                for polekey in paras:
                    if polekey == "RP":
                        # print(f"before, BCvartemp PHI ID {id(BCvarTemp[index]['PHI'])}")
                        # print(f"before, BCvartemp phi ID {id(BCvarTemp[index]['phi'])}")

                        BCvarTemp[index]["PHI"] = (
                            rk4a[INTRK] * BCvarTemp[index]["PHI"]
                            + self.dt * BCvarTemp[index]["phi"]
                        )

                        BC.BCvar[index]["phi"] += rk4b[INTRK] * BCvarTemp[index]["PHI"]

                        # print(f"after, BCvartemp PHI ID {id(BCvarTemp[index]['PHI'])}")
                        # print(f"after, BCvartemp phi ID {id(BCvarTemp[index]['phi'])}")

                        # print(f"BCvarTemp[index]['phi'] {BC.BCvar[index]['phi'].max()}")
                        # BC.BCvar[index]["phi"] = (
                        #     BC.BCvar[index]["phi"] + rk4b[INTRK] * BC.BCvar[index]["PHI"]
                        # )

                    elif polekey == "CP":
                        BC.BCvar[index]["KEXI1"] = (
                            rk4a[INTRK] * BC.BCvar[index]["KEXI1"]
                            + self.dt * BCvarTemp[index]["kexi1"]
                        )
                        BC.BCvar[index]["KEXI2"] = (
                            rk4a[INTRK] * BC.BCvar[index]["KEXI2"]
                            + self.dt * BCvarTemp[index]["kexi2"]
                        )

                        BC.BCvar[index]["kexi1"] += rk4b[INTRK] * BC.BCvar[index]["KEXI1"]
                        BC.BCvar[index]["kexi2"] += rk4b[INTRK] * BC.BCvar[index]["KEXI2"]
