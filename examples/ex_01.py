import os
import edg_acoustics
import numpy
import scipy.io

# from edg_acoustics.time_integration import TimeIntegrator

# from edg_acoustics.mesh import Mesh
# print(dir(edg_acoustics))

# Boundary conditions
BC_labels = {"slip": 11, "impedance1": 13, "impedance2": 14}

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

ToT = 0.001  # total simulation time in seconds

sim = edg_acoustics.AcousticsSimulation(rho0, c0, Nx, mesh, BC_labels)


flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)
AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)

sim.init_BC(AbBC)
sim.init_IC(IC)
sim.init_Flux(flux)
sim.init_rec(rec, "scipy")  # brute_force or scipy(default)

tsi_time_integrator = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL, Nt=3)
sim.init_TimeIntegrator(tsi_time_integrator)
prec = sim.time_integration(total_time=ToT, delta_step=10)

results = edg_acoustics.Monopole_postprocessor(sim, 1)
IR, TR, freqs = post.apply_correction()
# results.apply_correction()

# Save prec to Matlab format file
result_filename = os.path.join(os.path.split(os.path.abspath(__file__))[0], "result.mat")
scipy.io.savemat(result_filename, {"prec": prec, "IR": IR, "TR": TR, "freqs": freqs, "sim": sim})
print("Finished!")
