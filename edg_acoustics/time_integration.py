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
from multiprocessing import set_forkserver_preload
import meshio
import numpy
import abc
import edg_acoustics
import math
from edg_acoustics.acoustics_simulation import AcousticsSimulation
from edg_acoustics.preprocessing import Flux, UpwindFlux

__all__ = ['TimeIntegrator','RungeKutta_TI', 'CFL_Default']

CFL_Default = 0.5
class TimeIntegrator(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    
    @abc.abstractmethod
    def step_dt(self):
        # Takes the pressure, velocity at the time T and evolves it to time T + dt
        #   P0 := P(T)
        #   P := P(T + dt)
        # the same for all the other variables
        pass

class TSI_TI(TimeIntegrator):
    def __init__(self, AcousticSimulation_object, CFL: float=CFL_Default):
        # self.BC = BC_object
        self.sim = AcousticSimulation_object
        self.CFL = CFL
        self.dt = CFL * self.sim.dtscale / self.sim.c0 / (2 * self.sim.Nx +1)
        self.dVx = numpy.zeros_like(AcousticSimulation_object.Fscale)
        self.dVy = numpy.zeros_like(self.dVx)
        self.dVz = numpy.zeros_like(self.dVx)
        self.dP = numpy.zeros_like(self.dVx)
        
    def step_dt(self):
    # Takes the pressure, velocity at the time T and evolves it to time T + dt
    #   P0 := P(T), P0 contains the (high-order) derivative values as well 
    #   P := P(T + dt)
    # the same for all the other variables
        
        P0 = self.sim.P0
        Vx0 = self.sim.Vx0
        Vy0 = self.sim.Vy0
        Vz0 = self.sim.Vz0
##########################
        rho = self.sim.rho0
        c0 = self.sim.c0
        dt = self.dt
        Nt = self.sim.Nt



        # P = P0.copy()
        # Vx = Vx0.copy()
        # Vy = Vy0.copy()
        # Vz = Vz0.copy()

        P = self.sim.P
        Vx = self.sim.Vx
        Vy = self.sim.Vy
        Vz = self.sim.Vz
        # print(f"inside, P ID {id(P)}, sim.P ID {id(self.sim.P)}")

        dP = self.dP
        dVx = self.dVx
        dVy = self.dVy
        dVz = self.dVz

        nx = self.sim.n_xyz[0]
        ny = self.sim.n_xyz[1]
        nz = self.sim.n_xyz[2]
        
        # fluxP = self.sim.fluxP
        # fluxVx = self.sim.fluxVx
        # fluxVy = self.sim.fluxVy
        # fluxVz = self.sim.fluxVz

        lift = self.sim.lift
        Fscale = self.sim.Fscale
        Dr = self.sim.Dr
        Ds = self.sim.Ds
        Dt = self.sim.Dt
        rst_xyz = self.sim.rst_xyz

        vmapM = self.sim.vmapM
        vmapP = self.sim.vmapP
        BCnode = self.sim.BCnode  # list, each element is dic ['label'(int),'map'(numpy.ndarray),'vmap'(numpy.ndarray)]
        BCvar = self.sim.BC.BCvar  # list, each element is dic ['label', 'vn', 'ou', 'in', 'phi', 'PHI', 'kexi1', 'kexi2', 'KEXI1', 'KEXI2']
        BCpara = self.sim.BC.BCpara
        Flux = self.sim.Flux
        # print(f"inside, self.sim.BC.BCvar ID {id(self.sim.BC.BCvar)}, BCvar ID {id(BCvar)}")



        for Tind in range (1,Nt+1):
            ## below 4 lines are References
            # print(f"dVx ID {id(dVx)}, sim.dVx ID {id(self.sim.dVx)}")
            dVx.reshape(-1)[:] = Vx0.reshape(-1)[vmapM] - Vx0.reshape(-1)[vmapP]
            # print(f"dVx ID {id(dVx)}, sim.dVx ID {id(self.sim.dVx)}")
            dVy.reshape(-1)[:] = Vy0.reshape(-1)[vmapM] - Vy0.reshape(-1)[vmapP]
            dVz.reshape(-1)[:] = Vz0.reshape(-1)[vmapM] - Vz0.reshape(-1)[vmapP]
            dP.reshape(-1)[:] = P0.reshape(-1)[vmapM] - P0.reshape(-1)[vmapP]

            fluxVx = Flux.FluxVx(dVx, dVy, dVz, dP)  # has return object, might make copy, 
            fluxVy = Flux.FluxVy(dVx, dVy, dVz, dP)
            fluxVz = Flux.FluxVz(dVx, dVy, dVz, dP)
            fluxP = Flux.FluxP(dVx, dVy, dVz, dP)

            for index, paras in enumerate(BCpara):
                # 'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`. 
                # 'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row). 
                #     'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
                #          :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
                BCvar[index]['vn'] = nx.reshape(-1)[BCnode[index]['map']] * Vx0.reshape(-1)[BCnode[index]['vmap']] + \
                                ny.reshape(-1)[BCnode[index]['map']] * Vy0.reshape(-1)[BCnode[index]['vmap']] + \
                                nz.reshape(-1)[BCnode[index]['map']] * Vz0.reshape(-1)[BCnode[index]['vmap']] 
                BCvar[index]['ou'] = BCvar[index]['vn'] + P0.reshape(-1)[BCnode[index]['vmap']] / rho /c0
                BCvar[index]['in'] = BCvar[index]['ou'] * paras['RI']

                for polekey in paras:
                    if polekey== 'RP':
                            for i in range(paras['RP'].shape[1]):
                                BCvar[index]['in'] = BCvar[index]['in'] + paras['RP'][0,i] * BCvar[index]['phi'][i]
                    elif polekey=='CP':
                        pass # to be added

                fluxVx.reshape(-1)[BCnode[index]['map']] = nx.reshape(-1)[BCnode[index]['map']] * P0.reshape(-1)[BCnode[index]['vmap']] /rho - \
                                                        nx.reshape(-1)[BCnode[index]['map']] * c0 * (BCvar[index]['ou'] + BCvar[index]['in']) / 2
                fluxVy.reshape(-1)[BCnode[index]['map']] = ny.reshape(-1)[BCnode[index]['map']] * P0.reshape(-1)[BCnode[index]['vmap']] /rho - \
                                                        ny.reshape(-1)[BCnode[index]['map']] * c0 * (BCvar[index]['ou'] + BCvar[index]['in']) / 2
                fluxVz.reshape(-1)[BCnode[index]['map']] = nz.reshape(-1)[BCnode[index]['map']] * P0.reshape(-1)[BCnode[index]['vmap']] /rho - \
                                                        nz.reshape(-1)[BCnode[index]['map']] * c0 * (BCvar[index]['ou'] + BCvar[index]['in']) / 2
                fluxP.reshape(-1)[BCnode[index]['map']] = c0**2 * rho * (BCvar[index]['vn'] - 0.5 * (BCvar[index]['ou'] - BCvar[index]['in']))

            TP0 = P0.copy()
            P0 = -c0**2 * rho * (self.sim.grad_3d(Vx0, 'x') + self.sim.grad_3d(Vy0, 'y') + 
                                self.sim.grad_3d(Vz0, 'z')) + lift @ (Fscale * fluxP)
            dPdx, dPdy, dPdz = self.sim.grad_3d(TP0, 'xyz')
            Vx0 = -dPdx / rho + lift @ (Fscale * fluxVx)
            Vy0 = -dPdy / rho + lift @ (Fscale * fluxVy)
            Vz0 = -dPdz / rho + lift @ (Fscale * fluxVz)


            Vx += dt**Tind / math.factorial(Tind) * Vx0
            Vy += dt**Tind / math.factorial(Tind) * Vy0
            Vz += dt**Tind / math.factorial(Tind) * Vz0
            P += dt**Tind / math.factorial(Tind) * P0


            for index, paras in enumerate(BCpara):
                for polekey in paras:
                    if polekey== 'RP':
                            for i in range(paras['RP'].shape[1]):
                                BCvar[index]['phi'][i] = BCvar[index]['ou'] - paras['RP'][1,i] * BCvar[index]['phi'][i]

                            BCvar[index]['PHI'] += dt**Tind / math.factorial(Tind) * BCvar[index]['phi']
                    elif polekey=='CP':
                        pass # to be added

        # self.sim.P0 = P.copy()
        # # print(f"inside, self.sim.P0 ID {id(self.sim.P0)}, sim.P ID {id(P)}")

        # self.sim.Vx0 = Vx.copy()
        # self.sim.Vy0 = Vy.copy()
        # self.sim.Vz0 = Vz.copy()

        for index, paras in enumerate(BCpara):
            for polekey in paras:
                if polekey== 'RP':
                        BCvar[index]['phi'] = BCvar[index]['PHI'].copy()
                elif polekey=='CP':
                    pass # to be added

        # print(f"inside after loop, self.sim.BC.BCvar ID {id(self.sim.BC.BCvar)}, BCvar ID {id(BCvar)}")

        # print(f"max P inside loop {P.max()}")






        # phi = self.sim.BC.phi
        # Lift = self.AcousticSimulation.Lift

