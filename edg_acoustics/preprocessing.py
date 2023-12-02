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
import meshio
import numpy
import abc
import edg_acoustics
from edg_acoustics.acoustics_simulation import AcousticsSimulation

__all__ = ['Flux', 'UpwindFlux']

# Constants



class Flux(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def FluxP(self):
        pass

    @abc.abstractmethod
    def FluxVx(self):
        pass

    @abc.abstractmethod
    def FluxVy(self):
        pass

    @abc.abstractmethod
    def FluxVz(self):
        pass
        


class UpwindFlux(Flux):

    def __init__(self, rho0: float, c0: float, n_xyz: numpy.ndarray):
        self.rho0 = rho0
        self.c0 = c0
        self.n_xyz = n_xyz
        self.cn1s = -c0 * n_xyz[0]**2 / 2
        self.cn2s = -c0 * n_xyz[1]**2 / 2
        self.cn3s = -c0 * n_xyz[2]**2 / 2
        self.cn1n2 = -c0 * n_xyz[0] * n_xyz[1] / 2
        self.cn1n3 = -c0 * n_xyz[0] * n_xyz[2] / 2
        self.cn2n3 = -c0 * n_xyz[1] * n_xyz[2] / 2
        self.n1rho = n_xyz[0] / 2 / rho0 
        self.n2rho = n_xyz[1] / 2 / rho0 
        self.n3rho = n_xyz[2] / 2 / rho0 
        self.csn1rho = c0**2 * rho0 * n_xyz[0] / 2
        self.csn2rho = c0**2 * rho0 * n_xyz[1] / 2
        self.csn3rho = c0**2 * rho0 * n_xyz[2] / 2



    def FluxP(self, dvx: numpy.ndarray, dvy: numpy.ndarray, dvz: numpy.ndarray, dp: numpy.ndarray):
        return self.csn1rho * dvx + self.csn2rho * dvy + self.csn3rho * dvz - self.c0 / 2 * dp
    
    def FluxVx(self, dvx: numpy.ndarray, dvy: numpy.ndarray, dvz: numpy.ndarray, dp: numpy.ndarray):
        return self.cn1s * dvx + self.cn1n2 * dvy + self.cn1n3 * dvz + self. n1rho * dp
    
    def FluxVy(self, dvx: numpy.ndarray, dvy: numpy.ndarray, dvz: numpy.ndarray, dp: numpy.ndarray):
        return self.cn1n2 * dvx + self.cn2s * dvy + self.cn2n3 * dvz + self. n2rho * dp
    
    def FluxVz(self, dvx: numpy.ndarray, dvy: numpy.ndarray, dvz: numpy.ndarray, dp: numpy.ndarray):
        return self.cn1n3 * dvx + self.cn2n3 * dvy + self.cn3s * dvz + self. n3rho * dp





# class Flux(AcousticsSimulation):
#     """Setup initial condition of a DG acoustics simulation for a specific scenario.

#     :class:`.Flux` is used to load the boundary condition parameters, determine the time step size.

#     Args:
#         xyz (numpy.ndarray): ``[3, Np, N_tets]``the physical space coordinates :math:`(x, y, z)` of the collocation points of each
#                 element of the mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates,
#                  ``xyz[2]`` contains the z-coordinates.
#         source_xyz (numpy.ndarray): an (3, ) array containing the physical coordinates of the monopole source.
#         halfwidth (float): half-bandwidth of the initial Gaussian pulse.
#         Np (int): number of collocation nodes in an element.

#         freq_max (float): maximum resolvable frequency of the simulation. <default>: edg_acoustics.FREQ_MAX

#     # Raises:
#     #     ValueError: If BCpara.keys() is not present in the acoustics_simulation.BC_list.values(), an error is raised. 
#     #         If a label is present in the acoustics_simulation.BC_list.values() but not in BCpara, an error is raised.

#     Attributes:
#         P (numpy.ndarray): ``[Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each element of the\
#             mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates, ``xyz[2]``
#             contains the z-coordinates.



#     Example:
#         An element of this class can be initialized in the following way


#     """
#     #     self.IC = edg_acoustics.Flux(self.mesh, source_xyz, halfwidth, self.Np)
#     #     self.IC.set_source_location(source_xyz)
#     #     self.IC.set_frequency(halfwidth)
#     #     self.initial_condition_field=self.IC.compute_field() #values at nodes, 
#     def __init__(self, P: numpy.ndarray, U: numpy.ndarray, V: numpy.ndarray, W: numpy.ndarray):
#         # Store input parameters
#         self.P = P
#         self.U = U
#         self.V = V
#         self.W = W

#         # self.P = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         # self.U = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         # self.V = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         # self.W = numpy.zeros([xyz.shape[1], xyz.shape[2]])

#         # self.source_xyz = Flux.set_source_location(source_xyz)
#         # self.halfwidth = halfwidth
#         # self.P,self.U, self.V, self.W = Flux.monopole(xyz, source_xyz, halfwidth)

#     # Static methods ---------------------------------------------------------------------------------------------------
#     @staticmethod
#     def set_source_location(source_xyz: float):
#         return source_xyz
    
#     @staticmethod
#     def set_frequency(halfwidth: float):
#         return halfwidth
    
#     # @classmethod
#     # def monopole(cls, self, source_xyz: float, halfwidth: float):
#     #     P = numpy.zeros([self.xyz.shape[1], self.xyz.shape[2]])
#     #     U = numpy.zeros([self.xyz.shape[1], self.xyz.shape[2]])
#     #     V = numpy.zeros([self.xyz.shape[1], self.xyz.shape[2]])
#     #     W = numpy.zeros([self.xyz.shape[1], self.xyz.shape[2]])
        
#     #     P = numpy.exp(-numpy.log(2) * ((self.xyz[0] - source_xyz[0])**2 + (self.xyz[1] - source_xyz[1])**2 + (self.xyz[2] - source_xyz[2])**2) / halfwidth**2)
#     #     return cls(P, U, V, W)

#     @classmethod
#     def monopole(cls, xyz: numpy.ndarray, source_xyz: float, halfwidth: float):
#         P = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         U = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         V = numpy.zeros([xyz.shape[1], xyz.shape[2]])
#         W = numpy.zeros([xyz.shape[1], xyz.shape[2]])
        
#         P = numpy.exp(-numpy.log(2) * ((xyz[0] - source_xyz[0])**2 + (xyz[1] - source_xyz[1])**2 + (xyz[2] - source_xyz[2])**2) / halfwidth**2)
#         return cls(P, U, V, W)
