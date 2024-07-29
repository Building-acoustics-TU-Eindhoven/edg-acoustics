""" This is the main script for the scenario 1."""

# region Import Libraries
import os
import glob
import numpy
import edg_acoustics

# endregion

# --------------------
# Block 1: User input
# --------------------
rho0 = 1.213  # density of air at 20 degrees Celsius in kg/m^3
c0 = 343  # speed of sound in air at 20 degrees Celsius in m/s

BC_labels = {
    "Doors": 1,
    "UpperWall": 2,
    "Ceiling": 3,
    "LowerWall": 4,
    "Floor": 5,
}  # predefined labels for boundary conditions. please assign an arbitrary string to each type of boundary condition, e.g. hard wall, carpet, panel and an integer number of increasing order (starting from 1). The number should be unique for each type of boundary condition. The string help to keep track of the boundary surface and does not need to match the physical surface name exactly in the .geo mesh file.


mesh_name = "Corridor.geo"  # name of the geometry file. The .geo file should be in the same folder as this script.
monopole_xyz = numpy.array([12.45, 0.63, 1.5])  # x,y,z coordinate of the source in the room
freq_upper_limit = 200  # upper limit of the frequency content of the source signal in Hz. The source signal is a Gaussian pulse with a frequency content up to this limit.

# Approximation degrees
Nx = 4  # in space
Nt = 4  # in time
CFL = 0.5  # CFL number, default is 0.5.
recx = numpy.array([4.45])
recy = numpy.array([0.63])
recz = numpy.array([1.5])
rec = numpy.vstack((recx, recy, recz))  # dim:[3,n_rec]

impulse_length = 1  # total simulation time in seconds
save_every_Nstep = 10  # save the results every N steps
temporary_save_Nstep = 500  # save the results every N steps temporarily during the simulation. The temporary results will be saved in the root directory of this repo.

result_filename = "result"  # name of the result file. The result file will be saved in the same folder as this script. The result file will be saved in .mat format.

# --------------------------------------------------------------------------------
# Block 2: Initialize the simulation，run the simulation and save the results
# --------------------------------------------------------------------------------

# input Boundary conditions and parameters
BC_para = [
    {"label": 1, "RI": 0.99},
    {
        "label": 2,
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
        "label": 3,
        "RI": 0,
        "RP": numpy.array(
            [
                [1.505778842079319e04, 1.805778842079319e04],
                [1.509502512409186e04, 2.509502512409186e04],
            ]
        ),
        "CP": numpy.array([[-5.0], [100.0], [1.8e3], [2.0e3]]),
    },
    {"label": 4, "RI": 0.95},
    {"label": 5, "RI": 0.9},
]


# mesh_data_folder is the current folder by default
mesh_data_folder = os.path.split(os.path.abspath(__file__))[0]
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels, freq_max=freq_upper_limit)


IC = edg_acoustics.Monopole_IC(monopole_xyz, freq_upper_limit)

sim = edg_acoustics.AcousticsSimulation(rho0, c0, Nx, mesh, BC_labels)

flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)
AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)

sim.init_BC(AbBC)
sim.init_IC(IC)
sim.init_Flux(flux)
sim.init_rec(
    rec, "scipy"
)  # brute_force or scipy(default) approach to locate the receiver points in the mesh

tsi_time_integrator = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL, Nt=3)
sim.init_TimeIntegrator(tsi_time_integrator)
sim.time_integration(
    total_time=impulse_length,
    delta_step=save_every_Nstep,
    save_step=temporary_save_Nstep,
    format="mat",
)

results = edg_acoustics.Monopole_postprocessor(sim, 1)
results.apply_correction()


result_filename = os.path.join(os.path.split(os.path.abspath(__file__))[0], result_filename)
results.write_results(result_filename, "mat")
# load newresult.npy
# data = numpy.load("./examples/newresult.npz", allow_pickle=True)
# tempdata = numpy.load("./results_on_the_run.npz", allow_pickle=True)
print("Finished!")
