import edg_acoustics
#from edg_acoustics.mesh import Mesh
print(dir(edg_acoustics))

BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14, 'impedance3': 15}
filename = "/home/hwang/Desktop/edg-acoustics/data/tests/mesh/CoarseMesh.msh"
mesh = edg_acoustics.Mesh(filename, BC_labels)

sim = edg_acoustics.AcousticsSimulation(2, 2, mesh, BC_labels)
sim.init_local_system()


