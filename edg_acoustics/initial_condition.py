"""
``edg_acoustics.initial_condition``
======================

The edg_acoustics initial_condition  provide more necessary functionalities 
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

__all__ = ['InitialCondition','FREQ_MAX']

# Constants
FREQ_MAX = 2e3  # maximum resolvable frequency



class InitialCondition(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def P(self, xyz: numpy.ndarray)
        pass

    @abc.abstractmethod
    def U(self, xyz: numpy.ndarray)
        pass

    @abc.abstractmethod
    def V(self, xyz: numpy.ndarray)
        pass

    @abc.abstractmethod
    def W(self, xyz: numpy.ndarray)
        pass
        


class Monopole_IC(InitialCondition):

    def __init__(self, source_xyz: numpy.ndarray, halfwidth: float):
        self.source_xyz = source_xyz
        self.halfwidth = halfwidth

    def P(self, xyz: numpy.ndarray):
        pressure = numpy.exp(-numpy.log(2) * ((xyz[0] - self.source_xyz[0])**2 + (xyz[1] - self.source_xyz[1])**2 + (xyz[2] - self.source_xyz[2])**2) / self.halfwidth**2)
        return pressure
    
    def U(self, xyz: numpy.ndarray):
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])
    
    def V(self, xyz: numpy.ndarray):
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])
    
    def W(self, xyz: numpy.ndarray):
        return numpy.zeros([xyz.shape[1], xyz.shape[2]])





# class InitialCondition(AcousticsSimulation):
#     """Setup initial condition of a DG acoustics simulation for a specific scenario.

#     :class:`.InitialCondition` is used to load the boundary condition parameters, determine the time step size.

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
#     #     self.IC = edg_acoustics.InitialCondition(self.mesh, source_xyz, halfwidth, self.Np)
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

#         # self.source_xyz = InitialCondition.set_source_location(source_xyz)
#         # self.halfwidth = halfwidth
#         # self.P,self.U, self.V, self.W = InitialCondition.monopole(xyz, source_xyz, halfwidth)

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
