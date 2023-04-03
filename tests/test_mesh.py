"""Tests for the edg_acoustics.mesh module.
"""
import pytest

import edg_acoustics


@pytest.fixture
def mesh_raw_data_valid():
    """Define valid raw input data for mesh generation.
    """
    # The vertices of the mesh
    vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0]
    ]

    # The triangular cells of the mesh
    cells = [("triangle", [[0, 1, 2], [1, 3, 2]])]

    return [vertices, cells]


@pytest.fixture
def mesh_raw_data_invalid():
    """Define invalid raw input data for mesh generation.
    """
    # The vertices of the mesh
    vertices = None

    # The triangular cells of the mesh
    cells = None

    return [vertices, cells]


@pytest.fixture
def mesh_file_data_valid():
    """Define valid file input data for mesh generation.
    """
    # The valid filename with mesh data
    filename = "../data/tests/mesh/CoarseMesh.msh"

    return filename


@pytest.fixture
def mesh_file_data_invalid():
    """Define invalid file input data for mesh generation.
    """
    # The invalid filename with mesh data
    filename = None

    return filename


def test_mesh_raw_data_valid_input(mesh_raw_data_valid):
    mesh_raw_data = mesh_raw_data_valid
    mesh = edg_acoustics.Mesh(vertices=mesh_raw_data[0], cells=mesh_raw_data[1])


def test_mesh_file_data_valid_input(mesh_file_data_valid):
    mesh = edg_acoustics.Mesh(filename=mesh_file_data_valid)


def test_mesh_invalid_input(mesh_raw_data_invalid, mesh_file_data_invalid):
    mesh_raw_data = mesh_raw_data_invalid
    mesh_file_data = mesh_file_data_invalid

    with pytest.raises(ValueError) as excinfo:
        mesh = edg_acoustics.Mesh(vertices=mesh_raw_data[0], cells=mesh_raw_data[1],
                                  filename=mesh_file_data)
    assert "[edg_acoustics.mesh,Mesh] You must provide raw input (vertices, cells, [point_data, cell_data]) or mesh " \
           "filename from where to read data." in str(excinfo.value)

