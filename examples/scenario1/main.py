""" This is the main script for the scenario 1."""

import os
import glob
import numpy
import scipy.io
import edg_acoustics


# load Boundary conditions and parameters
BC_labels = {"hard wall": 11, "carpet": 13, "panel": 14}
BC_para = []  # clear the BC_para list
for material, label in BC_labels.items():
    if material == "hard wall":
        BC_para.append({"label": label, "RI": 1})
    else:
        # Find the corresponding .mat file in the current folder no matter what cwd is
        mat_files = glob.glob(f"{os.path.split(os.path.abspath(__file__))[0]}/{material}*.mat")

        # if mat_files is empty, raise an error
        if not mat_files:
            raise FileNotFoundError(f"No .mat file found for material '{material}'")

        mat_file = scipy.io.loadmat(mat_files[0])

        # Create the dictionary for this material
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

rho0 = 1.213  # density of air at 20 degrees Celsius in kg/m^3
c0 = 343  # speed of sound in air at 20 degrees Celsius in m/s

# Mesh
mesh_name = "scenario_2_coarse.msh"
# mesh_data_folder is the current folder by default
mesh_data_folder = os.path.split(os.path.abspath(__file__))[0]
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)

monopole_xyz = numpy.array([3.04, 2.59, 1.62])
halfwidth = 0.2
IC = edg_acoustics.Monopole_IC(monopole_xyz, halfwidth)


# Approximation degrees
Nx = 4  # in space
Nt = 3  # in time
CFL = 0.9  # CFL number, default is 0.5.
recx = numpy.array([4.26])
# recx = numpy.array([0.2])
recy = numpy.array([1.76])
recz = numpy.array([1.62])
rec = numpy.vstack((recx, recy, recz))  # dim:[3,n_rec]

ToT = 0.0001  # total simulation time in seconds

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
prec = sim.time_integration(total_time=ToT, delta_step=2)


# Save prec to Matlab format file to the same folder as this script, named "result.mat"
result_filename = os.path.join(os.path.split(os.path.abspath(__file__))[0], "result.mat")
scipy.io.savemat(result_filename, {"prec": prec})
print("Finished!")
