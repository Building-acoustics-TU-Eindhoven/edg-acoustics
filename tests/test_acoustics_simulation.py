"""Tests for the edg_acoustics.acoustics_simulation module.
"""
import pytest
import numpy
import pickle
import copy

import edg_acoustics


def read_mesh_dataset(mesh_option: str = 'coarse'):
    # Boundary conditions
    BC_labels =  {'slip': 11, 'impedance1': 13, 'impedance2': 14}

    # Mesh
    if mesh_option == 'coarse':
        mesh_filename = 'data/tests/mesh/coarse_cube_room.msh'
    else:
        assert False

    mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)

    # Approximation degrees
    Nx = 3  # in space
    Nt = 3  # in time

    sim = edg_acoustics.AcousticsSimulation(Nx, Nt, mesh, BC_labels)
    sim.init_local_system()

    return sim


reusable_coarse_mesh_sim = read_mesh_dataset('coarse')


@pytest.fixture
def coarse_mesh_sim():
    """Return coarse mesh simulation object.
        A deepcopy of the object is returned, it is not generated everytime to reduce the test time.
    """
    return copy.deepcopy(reusable_coarse_mesh_sim)

def test_geometry_factors_3d(coarse_mesh_sim):
    # Load the reference data
    reference_data = numpy.load('data/tests/acoustics_simulation/CoarseMesh_geometric_factors_3d.npz')

    # Check if sim.rx was correctly generated
    assert numpy.allclose(reference_data['rx'], coarse_mesh_sim.rst_xyz[0, 0], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.ry was correctly generated
    assert numpy.allclose(reference_data['ry'], coarse_mesh_sim.rst_xyz[0, 1], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.rz was correctly generated
    assert numpy.allclose(reference_data['rz'], coarse_mesh_sim.rst_xyz[0, 2], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.sx was correctly generated
    assert numpy.allclose(reference_data['sx'], coarse_mesh_sim.rst_xyz[1, 0], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.sy was correctly generated
    assert numpy.allclose(reference_data['sy'], coarse_mesh_sim.rst_xyz[1, 1], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.sz was correctly generated
    assert numpy.allclose(reference_data['sz'], coarse_mesh_sim.rst_xyz[1, 2], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.tx was correctly generated
    assert numpy.allclose(reference_data['tx'], coarse_mesh_sim.rst_xyz[2, 0], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.ty was correctly generated
    assert numpy.allclose(reference_data['ty'], coarse_mesh_sim.rst_xyz[2, 1], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.tz was correctly generated
    assert numpy.allclose(reference_data['tz'], coarse_mesh_sim.rst_xyz[2, 2], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.J was correctly generated
    assert numpy.allclose(reference_data['J'], coarse_mesh_sim.J, rtol=1e-10, atol=1e-10, equal_nan=False) == True
    
def test_normals_3d(coarse_mesh_sim):
    # Load the reference data
    reference_data = numpy.load('data/tests/acoustics_simulation/CoarseMesh_normals_3d.npz')

    # Check if sim.n_xyz[0] (x-component of normal vector) was correctly generated
    assert numpy.allclose(reference_data['nx'], coarse_mesh_sim.n_xyz[0], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.n_xyz[1] (y-component of normal vector) was correctly generated
    assert numpy.allclose(reference_data['ny'], coarse_mesh_sim.n_xyz[1], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.n_xyz[2] (z-component of normal vector) was correctly generated
    assert numpy.allclose(reference_data['nz'], coarse_mesh_sim.n_xyz[2], rtol=1e-10, atol=1e-10, equal_nan=False) == True

    # Check if sim.sJ (face Jacobians) was correctly generated
    assert numpy.allclose(reference_data['sJ'], coarse_mesh_sim.sJ, rtol=1e-10, atol=1e-10, equal_nan=False) == True
    