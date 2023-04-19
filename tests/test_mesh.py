"""Tests for the edg_acoustics.mesh module.
"""
import pytest
import numpy

import edg_acoustics


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


@pytest.fixture
def BC_labels_data_valid():
    """Define valid BC_labels input data for mesh generation.
    """
    # The valid filename with mesh data
    BC_labels = {'slip': 11, 'impedance1': 13, 'impedance2': 14, 'impedance3': 15}

    return BC_labels

@pytest.fixture
def BC_labels_data_extra_invalid():
    """Define invalid BC_labels input data for mesh generation with extra label not present in mesh.
    """
    # The valid filename with mesh data
    BC_labels = {'slip': 11, 'impedance1': 13, 'impedance2': 14, 'impedance3': 15, 'extra_label': 18}

    return BC_labels


@pytest.fixture
def BC_labels_data_missing_invalid():
    """Define invalid BC_labels input data for mesh generation with a mesh label missing.
    """
    # The valid filename with mesh data
    BC_labels = {'slip': 11, 'impedance1': 13, 'impedance2': 14}

    return BC_labels


def test_mesh_file_data_valid_input(mesh_file_data_valid, BC_labels_data_valid):
    mesh = edg_acoustics.Mesh(mesh_file_data_valid, BC_labels_data_valid)


def test_mesh_file_extra_label_invalid_input(mesh_file_data_valid, BC_labels_data_extra_invalid):
    with pytest.raises(ValueError) as excinfo:
        mesh = edg_acoustics.Mesh(mesh_file_data_valid, BC_labels_data_extra_invalid)

    assert "[edg_acoustics.Mesh] All BC labels must be present in the mesh and all labels in the mesh must be " \
                "present in BC_labels." in str(excinfo.value)


def test_mesh_file_missing_label_invalid_input(mesh_file_data_valid, BC_labels_data_missing_invalid):
    with pytest.raises(ValueError) as excinfo:
        mesh = edg_acoustics.Mesh(mesh_file_data_valid, BC_labels_data_missing_invalid)

    assert "[edg_acoustics.Mesh] All BC labels must be present in the mesh and all labels in the mesh must be " \
                "present in BC_labels." in str(excinfo.value)

def test_mesh_connectivity(mesh_file_data_valid, BC_labels_data_valid):
    # Initialize the mesh
    mesh = edg_acoustics.Mesh(mesh_file_data_valid, BC_labels_data_valid)

    # Load the reference data
    reference_data = numpy.load('../data/tests/mesh/CoarseMesh_connectivity.npz')

    # Check if mesh.EToV was correctly generated
    assert (mesh.EToV - reference_data['EToV']).sum() == 0

    # Check if mesh.EToE was correctly generated
    assert (mesh.EToE - reference_data['EToE']).sum() == 0

    # Check if mesh.EToF was correctly generated
    assert (mesh.EToF - reference_data['EToF']).sum() == 0
