import os
import sys
import edg_acoustics
import numpy


from edg_acoustics.time_integration import TimeIntegrator

# from edg_acoustics.mesh import Mesh
# print(dir(edg_acoustics))

# Boundary conditions
BC_labels = {"slip": 11, "impedance1": 13, "impedance2": 14}
# # BC paras
# BCpara (list [dict]): a list of boundary conditon parameters from the multi-pole model. Each element is a dictionary
# with keys (values) ['label'(int),'RI'(float),'RP'(numpy.ndarray),'CP'(numpy.ndarray)].
# 'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`.
# 'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row).
# 'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
#         :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
# More details about the multi-pole model parameters and boundary condition can be found in reference https://doi.org/10.1121/10.0001128.
# BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
# i.e., all boundary conditions in the simulation must have an associated boundary condition parameters.

BC_para = [
    {"label": 11, "RI": 1},
    {
        "label": 13,
        "RI": 0,
        "RP": numpy.array(
            [
                [2.849308439512733e03, 1.849308439512733e03],
                [2.843988875912554e03, 3.843988875912554e03],
            ]
        ),
    },
    {
        "label": 14,
        "RI": 0,
        "RP": numpy.array(
            [
                [1.505778842079319e04, 1.805778842079319e04],
                [1.509502512409186e04, 2.509502512409186e04],
            ]
        ),
    },
]
BC_para = [
    {"label": 11, "RI": 1},
    {
        "label": 13,
        "RI": 0,
        "RP": numpy.array(
            [
                [2.849308439512733e03, 1.849308439512733e03],
                [2.843988875912554e03, 3.843988875912554e03],
            ]
        ),
        "CP": numpy.array([[-5.0, 10.0], [100.0, 3.0], [1.8e3, 1.3e3], [2.0e3, 4.0e3]]),
    },
    {
        "label": 14,
        "RI": 0,
        "RP": numpy.array(
            [
                [1.505778842079319e04, 1.805778842079319e04],
                [1.509502512409186e04, 2.509502512409186e04],
            ]
        ),
        "CP": numpy.array([[-5.0], [100.0], [1.8e3], [2.0e3]]),
    },
]
rho0 = 1.2
c0 = 343
# Mesh
mesh_name = "coarse_cube_room.msh"
mesh_data_folder = os.path.abspath(
    os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        os.path.pardir,
        "data",
        "tests",
        "mesh",
    )
)
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)

monopole_xyz = numpy.array([0.5, 0.5, 0.5])
halfwidth = 0.2
IC = edg_acoustics.Monopole_IC(monopole_xyz, halfwidth)


# Approximation degrees
Nx = 2  # in space
Nt = 3  # in time
CFL = 0.5
recx = numpy.array([0.2, 0.3])
# recx = numpy.array([0.2])
recy = recx.copy()
recz = recx.copy()
rec = numpy.vstack((recx, recy, recz))  # dim:[3,n_rec]

ToT = 0.05  # total simulation time in seconds

sim = edg_acoustics.AcousticsSimulation(rho0, c0, Nx, mesh, BC_labels)


Flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)
AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)

# bc = edg_acoustics.BoundaryCondition(sim.BCnode, BC_para)

sim.init_BC(AbBC)
sim.init_IC(IC)
sim.init_Flux(Flux)
sim.init_rec(rec, "brute_force")  # brute_force or scipy(default)

tsi_time_integrator = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, Nt, CFL)
sim.init_TimeIntegrator(tsi_time_integrator)
sim.time_integration(total_time=ToT)
# sim.init_TimeIntegration(TSI, rec, ToT)

###########another simulation
# sim.resetIC() #
# sim.init_Flux(Flux)

# TSI = edg_acoustics.TSI_TI(sim, CFL)
# sim.init_TimeIntegration(TSI, rec, ToT)


# IC=edg_acoustics.InitialCondition.monopole(sim.xyz, source_xyz, halfwidth)
# edg_acoustics.InitialCondition.monopole(sim.xyz, source_xyz, halfwidth)
# sim.IC.set_frequency

# setup=edg_acoustics.setup_(sim,BC_para)
# simulation=edg_acoustics.time_

print("Finished!")
