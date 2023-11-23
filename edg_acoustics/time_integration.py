"""
``edg_acoustics.time_integration``
======================

The edg_acoustics time_integration ***** UPDATE ****  provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup initial condition for a specific scenario.

Functions and classes present in :mod:`edg_acoustics.initial_condition` are listed below.

Setup Initial Condition
---------------
   InitialCondition
"""
from __future__ import annotations
import meshio
import numpy
import abc
import edg_acoustics
from edg_acoustics.acoustics_simulation import AcousticsSimulation

__all__ = ['TimeIntegrator','RungeKutta_TI']


class TimeIntegrator(abc.ABC):
    def __init__(self, BC_object, AcousticSimulation_object):
        self.BC = BC_object
        self.AcousticSimulation = AcousticSimulation_object
    
    @abc.abstractmethod
    def step_dt(self, P_0, U_0, V_0, W_0)
        # Takes the pressure, velocity at the time T and evolves it to time T + dt
        #   P_0 := P(T)
        #   P_1 := P(T + dt)
        # the same for all the other variables
        pass

class RungeKutta_TI(TimeIntegrator):

    def step_dt(self, P_0, U_0, V_0, W_0):
    # Takes the pressure, velocity at the time T and evolves it to time T + dt
    #   P_0 := P(T)
    #   P_1 := P(T + dt)
    # the same for all the other variables
    phi = self.BC.phi
    Lift = self.AcousticSimulation.Lift

