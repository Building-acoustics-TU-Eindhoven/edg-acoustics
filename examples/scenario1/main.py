""" This is the main script for the scenario 1."""

# region Import Libraries
import os
import glob
import numpy
import scipy.io
import edg_acoustics

# endregion

# --------------------
# Block 1: User input
# --------------------
rho0 = 1.213  # density of air at 20 degrees Celsius in kg/m^3
c0 = 343  # speed of sound in air at 20 degrees Celsius in m/s
BC_labels = {
    "hard wall": 11,
    "carpet": 13,
    "panel": 14,
}  # predefined labels for boundary conditions. please assign an arbitrary int number to each type of boundary condition, e.g. hard wall, carpet, panel. The number should be unique for each type of boundary condition and should match the physical surface number in the .geo mesh file. The string should be the same as the material name in the .mat file (at least for the first few letters).

real_valued_impedance_boundary = [
    # {"label": 11, "RI": 0.9}
]  # extra labels for real-valued impedance boundary condition, if needed. The label should be the similar to the label in BC_labels. Since it's frequency-independent, only "RI", the real-valued reflection coefficient, is required. If not needed, just clear the elements of this list and keep the empty list.

mesh_name = "scenario_2_coarser.msh"  # name of the mesh file. The mesh file should be in the same folder as this script.
monopole_xyz = numpy.array([3.04, 2.59, 1.62])  # x,y,z coordinate of the source in the room
halfwidth = 0.23  # halfwidth of the initial Gaussian source in meters. It determines the width of the initial Gaussian source in the simulation, which is used to control the upper limit of frequency content of the source signal. For simulations below 500 Hz, a value of 0.2 is recommended. For simulatoins between 500 and 1000 Hz, a value of 0.15 is recommended. For simulations between 1K and 3K Hz, a value of 0.075 is recommended.

# Approximation degrees
Nx = 3  # in space
Nt = 3  # in time
CFL = 0.5  # CFL number, default is 0.5.
recx = numpy.array([4.26])
recy = numpy.array([1.76])
recz = numpy.array([1.62])
rec = numpy.vstack((recx, recy, recz))  # dim:[3,n_rec]

impulse_length = 2  # total simulation time in seconds
save_every_Nstep = 10  # save the results every N steps
temporary_save_Nstep = 500  # save the results every N steps temporarily during the simulation. The temporary results will be saved in the root directory of this repo.

result_filename = "result"  # name of the result file. The result file will be saved in the same folder as this script. The result file will be saved in .mat format.

# --------------------------------------------------------------------------------
# Block 2: Initialize the simulation，run the simulation and save the results
# --------------------------------------------------------------------------------

# load Boundary conditions and parameters
BC_para = []  # clear the BC_para list
for material, label in BC_labels.items():
    if material == "hard wall":
        BC_para.append({"label": label, "RI": 0.99})
    else:
        mat_files = glob.glob(f"{os.path.split(os.path.abspath(__file__))[0]}/{material}*.mat")

        # if mat_files is empty, raise an error
        if not mat_files:
            raise FileNotFoundError(f"No .mat file found for material '{material}'")

        mat_file = scipy.io.loadmat(mat_files[0])

        material_dict = {"label": label}

        # Check if each variable exists in the .mat file and add it to the dictionary if it does
        if "RI" in mat_file:
            material_dict["RI"] = mat_file["RI"][0]
        else:
            material_dict["RI"] = 0

        if "AS" in mat_file and "lambdaS" in mat_file:
            material_dict["RP"] = numpy.array([mat_file["AS"][0], mat_file["lambdaS"][0]])  # type: ignore
        if "BS" in mat_file and "CS" in mat_file and "alphaS" in mat_file and "betaS" in mat_file:
            material_dict["CP"] = numpy.array(  # type: ignore
                [mat_file["BS"][0], mat_file["CS"][0], mat_file["alphaS"][0], mat_file["betaS"][0]]
            )

        BC_para.append(material_dict)
BC_para += real_valued_impedance_boundary


# mesh_data_folder is the current folder by default
mesh_data_folder = os.path.split(os.path.abspath(__file__))[0]
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)


IC = edg_acoustics.Monopole_IC(monopole_xyz, halfwidth)

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
