""" This module provides initial condition functionalities for the edg_acoustics package.

The current version of edg_acoustics.initial_condition provides monopole source initial condition.
"""

from __future__ import annotations
import abc
import numpy

__all__ = ["InitialCondition", "Monopole_IC"]


class InitialCondition(abc.ABC):
    """Setup initial condition of a DG acoustics simulation for a specific scenario.

    :class:`.InitialCondition` is used to setup initial condition.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def Pinit(self, xyz: numpy.ndarray):
        """Setup initial condition for pressure."""

    @abc.abstractmethod
    def VXinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in x-direction."""

    @abc.abstractmethod
    def VYinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in y-direction."""

    @abc.abstractmethod
    def VZinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in z-direction."""


class Monopole_IC(InitialCondition):
    """Setup a monopole source for a specific scenario.

    :class:`.Monopole_IC` is used to setup monopple source initial condition.

    Args:
        xyz (numpy.ndarray): see :attr:`edg_acoustics.AcousticsSimulation.xyz`.
        source_xyz (numpy.ndarray): an (3,) array containing the physical coordinates of the monopole source.
        halfwidth (float): half-bandwidth of the initial Gaussian pulse.

    Attributes:
        source_xyz (numpy.ndarray): an (3,) array containing the physical coordinates of the monopole source.
        halfwidth (float): half-bandwidth of the initial Gaussian pulse.
    """

    def __init__(self, source_xyz: numpy.ndarray, halfwidth: float):
        self.source_xyz = source_xyz
        self.halfwidth = halfwidth

    def Pinit(self, xyz: numpy.ndarray):
        """Setup initial condition for pressure."""
        pressure = numpy.exp(
            -numpy.log(2)
            * (
                (xyz[0] - self.source_xyz[0]) ** 2
                + (xyz[1] - self.source_xyz[1]) ** 2
                + (xyz[2] - self.source_xyz[2]) ** 2
            )
            / self.halfwidth**2
        )
        return pressure

    def VXinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in x-direction."""
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])

    def VYinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in y-direction."""
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])

    def VZinit(self, xyz: numpy.ndarray):
        """Setup initial condition for velocity in z-direction."""
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])
