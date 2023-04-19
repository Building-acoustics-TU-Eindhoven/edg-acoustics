import edg_acoustics

BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14, 'impedance3': 15}
filename = "../data/tests/mesh/CoarseMesh.msh"
mesh = edg_acoustics.Mesh(filename, BC_labels)
