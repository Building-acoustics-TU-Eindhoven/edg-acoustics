import edg_acoustics

BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14, 'impedance3': 15}
filename = "/Users/apalha/work/dev/edg_acoustics/edg-acoustics/data/tests/mesh/CoarseMesh.msh"
mesh = edg_acoustics.Mesh(filename, BC_labels)
Nx = 3
Nt = 3

sim = edg_acoustics.AcousticsSimulation(Nx, Nt, mesh, BC_labels)
sim.init_local_system()

print('Finished!')


