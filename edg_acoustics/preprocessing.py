"""This module provides preprocessing functionalities for the edg_acoustics package.
"""

from __future__ import annotations
import numpy


__all__ = ["UpwindFlux"]


class UpwindFlux:
    """Calculation of upwind fluxes."""

    def __init__(self, rho0: float, c0: float, n_xyz: numpy.ndarray):
        """Setup constants for upwind fluxes.

        Args:
            rho0 (float): The reference density.
            c0 (float): The reference speed of sound.
            n_xyz (numpy.ndarray): The array representing the normal vector of the face.

        """
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
        """This method calculates the pressure flux using the given input arrays.

        Args:
            dvx (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            numpy.ndarray: The calculated pressure flux.

        """
        return self.csn1rho * dvx + self.csn2rho * dvy + self.csn3rho * dvz - self.c0 / 2 * dp

    def FluxVx(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        """This method calculates the flux of velocity in x-direction using the given input arrays.

        Args:
            dvx (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            numpy.ndarray: The calculated flux of velocity in x-direction.
        """
        return self.cn1s * dvx + self.cn1n2 * dvy + self.cn1n3 * dvz + self.n1rho * dp

    def FluxVy(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        """This method calculates the flux of velocity in y-direction using the given input arrays.

        Args:
            dvx (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            numpy.ndarray: The calculated flux of velocity in y-direction.
        """
        return self.cn1n2 * dvx + self.cn2s * dvy + self.cn2n3 * dvz + self.n2rho * dp

    def FluxVz(
        self,
        dvx: numpy.ndarray,
        dvy: numpy.ndarray,
        dvz: numpy.ndarray,
        dp: numpy.ndarray,
    ):
        """This method calculates the flux of velocity in z-direction using the given input arrays.

        Args:
            dvx (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            numpy.ndarray: The calculated flux of velocity in z-direction.
        """
        return self.cn1n3 * dvx + self.cn2n3 * dvy + self.cn3s * dvz + self.n3rho * dp
