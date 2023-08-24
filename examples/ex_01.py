import os

import edg_acoustics
#from edg_acoustics.mesh import Mesh
# print(dir(edg_acoustics))

# Boundary conditions
BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14}

# Mesh
mesh_name = 'coarse_cube_room.msh'
mesh_data_folder = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], os.path.pardir, 'data', 'tests', 'mesh'))
mesh_filename = os.path.join(mesh_data_folder, mesh_name)
mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)

# Approximation degrees
Nx = 3  # in space
Nt = 3  # in time

sim = edg_acoustics.AcousticsSimulation(Nx, Nt, mesh, BC_labels)
sim.init_local_system()

print('Finished!')
