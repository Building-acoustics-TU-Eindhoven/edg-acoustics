import os
import sys
import edg_acoustics
import numpy
#from edg_acoustics.mesh import Mesh
# print(dir(edg_acoustics))

# Boundary conditions
BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14}
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
            {'label': 11, 'RI': 1},
            {'label': 13, 'RI': 0, 'RP': numpy.array([[2.849308439512733e+03],[2.843988875912554e+03]])},
            {'label': 14, 'RI': 0, 'RP': numpy.array([[1.505778842079319e+04],[1.509502512409186e+04]])}
            ]
source_xyz = numpy.array ( [0.5, 0.5, 0.5])
halfwidth = 0.2
# Mesh
mesh_name = 'coarse_cube_room.msh'
mesh_data_folder = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], os.path.pardir, 'data', 'tests', 'mesh'))
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)

# Approximation degrees
Nx = 2  # in space
Nt = 3  # in time




sim = edg_acoustics.AcousticsSimulation(Nx, Nt, mesh, BC_labels)
sim.init_local_system()
# bc = edg_acoustics.BoundaryCondition(sim.BCnode, BC_para)

sim.init_BC(BC_para)
sim.init_IC(source_xyz, halfwidth)
# sim.IC.set_frequency

# setup=edg_acoustics.setup_(sim,BC_para)
# simulation=edg_acoustics.time_

print('Finished!')
