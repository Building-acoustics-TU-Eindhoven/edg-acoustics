"""
``edg_acoustics.preprocessing``
======================

The edg_acoustics preprocessing  provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup/precalculate simulation constants for a specific scenario.

Functions and classes present in :mod:`edg_acoustics.preprocessing` are listed below.

Setup Constants
---------------
   Flux
"""

from __future__ import annotations
import abc
import numpy


__all__ = ["Flux", "UpwindFlux"]


class Flux(abc.ABC):
    """abstract base class for fluxes."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def FluxP(self):
        """abstract method for pressure flux."""

    @abc.abstractmethod
    def FluxVx(self):
        """abstract method for flux of velocity in x-direction."""

    @abc.abstractmethod
    def FluxVy(self):
        """abstract method for flux of velocity in y-direction."""

    @abc.abstractmethod
    def FluxVz(self):
        """abstract method for flux of velocity in z-direction."""


class UpwindFlux(Flux):
    """Setup constants for upwind fluxes."""

    def __init__(self, rho0: float, c0: float, n_xyz: numpy.ndarray):
        self.rho0 = rho0
        self.c0 = c0
        self.n_xyz = n_xyz
        self.cn1s = -c0 * n_xyz[0] ** 2 / 2
        self.cn2s = -c0 * n_xyz[1] ** 2 / 2
        self.cn3s = -c0 * n_xyz[2] ** 2 / 2
        self.cn1n2 = -c0 * n_xyz[0] * n_xyz[1] / 2
        self.cn1n3 = -c0 * n_xyz[0] * n_xyz[2] / 2
        self.cn2n3 = -c0 * n_xyz[1] * n_xyz[2] / 2
        self.n1rho = n_xyz[0] / 2 / rho0
        self.n2rho = n_xyz[1] / 2 / rho0
        self.n3rho = n_xyz[2] / 2 / rho0
        self.csn1rho = c0**2 * rho0 * n_xyz[0] / 2
        self.csn2rho = c0**2 * rho0 * n_xyz[1] / 2
        self.csn3rho = c0**2 * rho0 * n_xyz[2] / 2

    def FluxP(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        return self.csn1rho * dvx + self.csn2rho * dvy + self.csn3rho * dvz - self.c0 / 2 * dp

    def FluxVx(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        return self.cn1s * dvx + self.cn1n2 * dvy + self.cn1n3 * dvz + self.n1rho * dp

    def FluxVy(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        return self.cn1n2 * dvx + self.cn2s * dvy + self.cn2n3 * dvz + self.n2rho * dp

    def FluxVz(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        return self.cn1n3 * dvx + self.cn2n3 * dvy + self.cn3s * dvz + self.n3rho * dp
