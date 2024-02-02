"""
``edg_acoustics.acoustics_simulation``
======================

The edg_acoustics acoustcs_simulation functions and classes provide functionality setup and run a DG acoustics
simulation.

Please note that the most-used mesh functions and classes in edg_acoustics are present in
the main :mod:`edg_acoustics` namespace rather than in :mod:`edg_acoustics.acoustics_simulation`.  These are:
:class:`.AcousticsSimulation`, and :func:`.some_function`.

Functions and classes present in :mod:`edg_acoustics.acoustics_simulation` are listed below.

Discretisation
--------------
   AcousticsSimulation
"""
from __future__ import annotations
import modepy
import numpy
import math
import edg_acoustics
from scipy.spatial import Delaunay


__all__ = ['AcousticsSimulation', 'NODETOL']

# Constants
NODETOL = 1.0e-7  # tolerance used to define if a node lies in a facet of the mesh

class AcousticsSimulation:
    """Acoustics simulation data structure for running a DG acoustics simulation.

    :class:`.AcousticsSimulation` contains the domain discretization and sets up the DG finite element discretisation
    for the solution to the acoustic wave propagation.

    Args:
        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.
        Nt (int): the order of the time integration scheme.
        mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
        BC_list (dict[str, int]): a dictionary containing the definition of the boundary
            conditions that are present in the mesh. BC_list.keys() must contain the same elements as
            mesh.BC_triangles.keys(), i.e., all boundary conditions in the mesh must have an associated boundary condition
            definition.
        node_tolerance (float): the tolerance used to check if a node belongs to a facet or not.
            <default>: edg_acoustics.NODETOL

    Raises:
        ValueError: If BC_list['my_label'] is not present in the mesh, an error is raised. If a label
            is present in the mesh but not in BC_list, an error is raised.

    Attributes:
        BC_list (dict[str, int]): a dictionary containing the definition of the boundary
            conditions that are present in the mesh. BC_list.keys() must contain the same elements as
            mesh.BC_triangles.keys(), i.e., all boundary conditions in the mesh must have an associated boundary condition
            definition.
        Dr (numpy.ndarray): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial r}`, in the rst reference element coordinate system.
        Ds (numpy.ndarray): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial s}`, in the rst reference element coordinate system.
        Dt (numpy.ndarray): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial t}`, in the rst reference element coordinate system.
        dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.
        Fmask (numpy.ndarray): ``[4, Nfp]`` array containing indices of nodes per surface of the tetrahedron.
        J: (numpy.ndarray): ``[Np, N_tets]`` The determinant of the Jacobian matrix for the coordinate transformation, 
            at the collocation nodes. 
        sJ:(numpy.ndarray): ``[4*Nfp, N_tets]`` The determinant of the surface Jacobian matrix at each of the
            collocation nodes, for each of the 4 faces of the ``N_tets`` elements. 
        Fscale:(numpy.ndarray): ``[4*Nfp, N_tets]`` ratio of surface to volume Jacobian of facial node 
            = self.sJ / self.J [self.Fmask.reshape(-1), :], to be element-wise multiplied with fluxes
        lift (numpy.ndarray): ``[Np, 4*Nfp]`` an array containing the product of inverse of the mass matrix (3D) with the face-mass matrices (2D).
        M (numpy.ndarray): the reference element mass matrix :math:`M := V^{-t}V^{-1}`.
            mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
        mesh: Mesh data structure generated from common mesh file formats
        Nfp (int): number of collocation nodes in a face.
        node_tolerance (float): tolerance used to determine if a node lies on a facet.
        Np (int): number of collocation nodes in an element.
        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.
        Nt (int): the order of the time integration scheme.
        rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
            ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
            the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates, ``rst[2]``
            contains the t-coordinates.
        rst_xyz (numpy.ndarray): ``[3, 3, Np, N_tets]`` The derivative of the local coordinates :math:`R = (r, s, t)` with 
            respect to the physical coordinates :math:`X = (x, y, z)`, i.e., :math:`\\frac{\\partial R}{\\partial X}`, at the collocation nodes. Specifically:

            rst_xyz[0, 0]: rx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                coordinates :math:`x`, i.e., :math:`\\frac{\\partial r}{\\partial x}`, at the collocation nodes.
            rst_xyz[1, 0]: sx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                coordinates :math:`x`, i.e., :math:`\\frac{\\partial s}{\\partial x}`, at the collocation nodes.
            rst_xyz[2, 0]: tx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                coordinates :math:`x`, i.e., :math:`\\frac{\\partial t}{\\partial x}`, at the collocation nodes.
            rst_xyz[0, 1]: ry (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                coordinates :math:`y`, i.e., :math:`\\frac{\\partial r}{\\partial y}`, at the collocation nodes.
            rst_xyz[1, 1]: sy (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                coordinates :math:`y`, i.e., :math:`\\frac{\\partial s}{\\partial y}`, at the collocation nodes.
            rst_xyz[2, 1]: ty (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                coordinates :math:`y`, i.e., :math:`\\frac{\\partial t}{\\partial y}`, at the collocation nodes.
            rst_xyz[0, 2]: rz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                coordinates :math:`z`, i.e., :math:`\\frac{\\partial r}{\\partial z}`, at the collocation nodes.
            rst_xyz[1, 2]: sz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                coordinates :math:`z`, i.e., :math:`\\frac{\\partial s}{\\partial z}`, at the collocation nodes.
            rst_xyz[2, 2]: tz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                coordinates :math:`z`, i.e., :math:`\\frac{\\partial t}{\\partial z}`, at the collocation nodes.
        V (numpy.ndarray): ``[Np, Np]`` the reference element van der Monde matrix of the orthonormal basis functions, :math:`f_{j}`, on the
            3D simplices (elements of the mesh), i.e., :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.
        V3D (numpy.ndarray): ``[Np, Np]`` the gradient of van der Monde matrix
        xyz (numpy.ndarray): ``[3, Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each element of the\
            mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates, ``xyz[2]``
            contains the z-coordinates.
        n_xyz (numpy.ndarray): ``[3, 4*Np, N_tets]`` the outwards normals :math:`\\vec{n}` at each collocation
            point on the element faces. Specifically:
                n_xyz[0, :] (numpy.ndarray): nx ``[4*Nfp, N_tets]`` The :math:`x`-component of the outward normal :math:`\\vec{n}`
                    at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
                n_xyz[1, :] (numpy.ndarray): ny ``[4*Nfp, N_tets]`` The :math:`y`-component of the outward normal :math:`\\vec{n}`
                    at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
                n_xyz[2, :] (numpy.ndarray): nz ``[4*Nfp, N_tets]`` The :math:`z`-component of the outward normal :math:`\\vec{n}`
                    at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
        vmapM (numpy.ndarray): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for interior values
        vmapP (numpy.ndarray): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for exterior values

    Example:
        An element of this class can be initialized in the following way

    """

  
    def __init__(self, rho0: float, c0: float, Nx: int, Nt: int, mesh: edg_acoustics.Mesh, BC_list: dict[str, int], node_tolerance: float = NODETOL):
        # apalha: Nt should not be an input and an attribute of acoustic simulation, because you may have different time integrators, which do not take Nt
        # Everything related to the time integrator is set in the time integrator. I did that, but did not change here to keep the old time evolution
        # function working. Once you agree with this change, we can remove these dead code.
        # Check if BC_list and mesh are compatible
        if not AcousticsSimulation.check_BC_list(BC_list, mesh):
            raise ValueError(
                "[edg_acoustics.AcousticSimulation] All BC labels must be present in the mesh and all labels in the mesh must be "
                "present in BC_list.")

        # Store input parameters
        self.rho0 = rho0
        self.c0 = c0
        self.mesh = mesh
        self.Nx = Nx
        self.Nt = Nt
        self.N_tets = mesh.EToV.shape[1]
        self.BC_list = BC_list
        self.dim = 3  # we are always in 3D, just added for external reference     
        self.node_tolerance = node_tolerance  # define a tolerance value for determining if a node belongs to a facet or not 
  
        # Compute attributes
        self.Np = AcousticsSimulation.compute_Np(Nx)  # number of colocation nodes in an element
        self.Nfp = AcousticsSimulation.compute_Nfp(Nx)  # number of nodes in a face
      
        # Set other attributes as None, since they are not yet initialized
        self.xyz = None
        self.rst = None
        self.V = None
        self.M = None
        self.Dr = None
        self.Ds = None
        self.Dt = None
        self.Fmask = None
        self.lift = None
        # dtype_input: str ='float64'
        # self.dtype_dict={'float64': numpy.float64, 'float32': numpy.float32}

        self.init_local_system()

    def init_local_system(self):
        """Compute local system matrices and local variables.

        Compute quantities associated to reference element and local coordinates, namely:
            - Reference element collocation nodes
            - Element collocation nodes (transformation of reference element to each element)
            - Reference element van der Monde matrix
            - Reference element mass matrix
            - Reference element derivative matrices

        Updates:
            - self.xyz
            - self.rst (3* Np)
            - self.V self.inV
            - self.M
            - self.Dr, self.Ds, self.Dt
            - self.Fmask (4*Nfp)
            - self.lift
        Args:

        Returns:
        """

        self.rst, self.xyz = AcousticsSimulation.compute_collocation_nodes(
            self.mesh.EToV, self.mesh.vertices, self.Nx, dim=self.dim)

        # Compute the van der Monde matrix and its inverse
        self.V = AcousticsSimulation.compute_van_der_monde_matrix(self.Nx, self.rst)
        # self.inV = numpy.linalg.inv(self.V)

        # Compute the van der Monde matrix of the gradients
        self.V3D = AcousticsSimulation.compute_grad_van_der_monde_matrix(self.Nx, self.rst)

        # Compute mass matrix
        self.M = AcousticsSimulation.compute_mass_matrix(self.V)

        # Compute the derivative matrices
        self.Dr, self.Ds, self.Dt = AcousticsSimulation.compute_derivative_matrix(self.Nx, self.rst)

        # Find all the ``Nfp`` face nodes that lie on each surface.
        self.Fmask=AcousticsSimulation.compute_Fmask(self.rst, self.node_tolerance)

        # Compute the product of inverse of the mass matrix (3D) with the face-mass matrices (2D)
        self.lift=AcousticsSimulation.compute_lift(self.V, self.rst, self.Fmask)

        # Compute the metric terms for the mesh
        self.rst_xyz, self.J = AcousticsSimulation.geometric_factors_3d(self.xyz, self.Dr, self.Ds, self.Dt)

        # Compute the face normals at the collocation points and the surface Jacobians
        self.n_xyz, self.sJ = AcousticsSimulation.normals_3d(self.xyz, self.rst_xyz, self.J, self.Fmask)

        # Compute ratio of surface to volume Jacobian of facial node
        self.Fscale = self.sJ / self.J [self.Fmask.reshape(-1), :]

        # Find connectivity for nodes given per surface in all elements
        self.vmapM, self.vmapP = AcousticsSimulation.build_maps_3d(self.xyz, self.mesh.EToE, self.mesh.EToF, self.Fmask, self.node_tolerance)

        # Build specialized nodal maps for various types of boundary conditions,specified in BC_list
        self.BCnode = AcousticsSimulation.build_BCmaps_3d(self.BC_list, self.mesh.EToV, self.vmapM, self.mesh.BC_triangles, self.Nx)
            
        self.dtscale = AcousticsSimulation.dtscale_3d(self.Fscale)    

        
    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_Np(Nx: int):
        """Computes the number of collocation nodes for basis of polynomial degree ``Nx``.

        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.

        Returns:
            Np (int): number of collocation nodes in an element.
        """
        
        return int((Nx+1)*(Nx+2)*(Nx+3)/6)
    
    @staticmethod
    def compute_Nfp(Nx: int):
        """Computes the number of collocation nodes lying on a face of the elements for basis of polynomial degree ``Nx``.

        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.

        Returns:
            Nfp (int): number of collocation nodes in a face.
        """
        
        return int((Nx+1)*(Nx+2)/2)
    
    @staticmethod
    def compute_Nx_from_Np(Np: int):
        """Computes the  polynomial degree ``Nx`` of basis from the number of collocation points.

        Args:
            Np (int): number of collocation nodes in an element.
            

        Returns:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave propagation problem.
        """
        # Since Np is given by:
        #   Np = (Nx + 1)*(Nx + 2)*(Nx + 3)/6
        # to compute Nx from Np we need to solve a third order polynomial equation:
        # Nx^3 + 6Nx^2 + 11Nx + 6(1 - Np) = 0
        polynominal = numpy.polynomial.Polynomial([(6 * (1 - Np)), 11, 6, 1])  # we setup the polynomial
        Nx = int(round(polynominal.roots()[-1].real))  # then we just get the roots and extract the root with the largest real component

        return Nx
        
    @staticmethod
    def check_BC_list(BC_list: dict[str, int], mesh: edg_acoustics.Mesh):
        """Check if BC_list is compatible with mesh.

        Given a mesh with a set of boundary conditions specified in mesh.BC_triangles, check if the list of boundary
        conditions specification, BC_list, is compatible. By compatible we mean that all boundary conditions (keys) in
        BC_list exist in mesh.BC_labels, and vice-versa.

        Args:
            mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
            BC_list (dict[str, int]): a dictionary containing the definition of the boundary
                conditions that are present in the mesh. BC_list.keys() must contain the same elements as
                mesh.BC_triangles.keys(), i.e., all boundary conditions in the mesh must have an associated boundary condition
                definition.

        Returns:
            is_compatible (bool): a flag specifying if BC_list is compatible with the mesh or not.
        """
        return BC_list.keys() == mesh.BC_triangles.keys()
    
    @staticmethod
    def compute_collocation_nodes(EToV: numpy.ndarray, vertices: numpy.ndarray, Nx: int, dim: int = 3):
        """
        Compute reference element (rst) coordinates of collocation points and the physical domain (xyz) coordinates
        self.dim and maximum polynomial degree to interpolate over these nodes (determines the number of nodes).
        for each element.These are so called Fekete points (low, close to optimal, Lebesgue constant) on simplices of dimension
        self.rst = modepy.warp_and_blend_nodes(self.dim, self.Nx)

        Args:
            EToV (numpy.ndarray): An (4 x self.N_tets) array containing the 4 indices of the vertices of the N_tets
               tetrahedra that make up the mesh.
            vertices (numpy.ndarray): An (M x self.N_vertices) array containing the M coordinates of the self.N_vertices
                vertices that make up the mesh. M specifies the geometric dimension of the mesh, such that the mesh
                describes an M-dimensional domain.
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.
            dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.

        Returns:
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            xyz (numpy.ndarray): ``[3, Np, N_tets]``the physical space coordinates :math:`(x, y, z)` of the collocation points of each
                element of the mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates,
                 ``xyz[2]`` contains the z-coordinates.
        """

        # Compute reference element (rst) coordinates of collocation points and the physical domain (xyz) coordinates
        # for each element.
        # These are so called Fekete points (low, close to optimal, Lebesgue constant) on simplices of dimension
        # self.dim and maximum polynomial degree to interpolate over these nodes (determines the number of nodes).
        rst = modepy.warp_and_blend_nodes(dim, Nx)

        # Compute the xyz coordinates for each element from the reference domain coordinates and the element's vertices

        # Start by allocating space for the xyz-coordinates for collocation points on each element
        xyz = numpy.zeros([dim, rst.shape[1], EToV.shape[1]])

        # Then get shorter references for the indices of the references for each of the nodes of the elements
        # get the indices of the first node of each element
        vertex_0_idx = EToV[0, :]
        # get the indices of the second node of each element
        vertex_1_idx = EToV[1, :]
        # get the indices of the third node of each element
        vertex_2_idx = EToV[2, :]
        # get the indices of the fourth node of each element
        vertex_3_idx = EToV[3, :]

        # Get shorter references for the r, s, and t coordinates, and the sum r + s + t
        rst_sum = rst.sum(axis=0).reshape([-1, 1])
        r = rst[0, :].reshape([-1, 1])
        s = rst[1, :].reshape([-1, 1])
        t = rst[2, :].reshape([-1, 1])

        # Compute the xyz coordinates
        xyz[0] = 0.5 * (-(1.0 + rst_sum) * vertices[0, vertex_0_idx] + (1.0 + r) * vertices[0, vertex_1_idx] +
                        (1.0 + s) * vertices[0, vertex_2_idx] + (1.0 + t) * vertices[0, vertex_3_idx])
        xyz[1] = 0.5 * (-(1.0 + rst_sum) * vertices[1, vertex_0_idx] + (1.0 + r) * vertices[1, vertex_1_idx] +
                        (1.0 + s) * vertices[1, vertex_2_idx] + (1.0 + t) * vertices[1, vertex_3_idx])
        xyz[2] = 0.5 * (-(1.0 + rst_sum) * vertices[2, vertex_0_idx] + (1.0 + r) * vertices[2, vertex_1_idx] +
                        (1.0 + s) * vertices[2, vertex_2_idx] + (1.0 + t) * vertices[2, vertex_3_idx])

        # Return the computed coordinates
        return rst, xyz
    
    @staticmethod
    def compute_van_der_monde_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
        """Compute the van der Monde matrix.

        Computes the van der Monde matrix for an orthonormal basis on the reference simplex. This polynomial basis
        can exactly represent polynomials up to degree ``Nx``.

        Consider the set of :math:`n` 3D nodes, with the coordinates of each node :math:`i` equal to
        :math:`(r_{i}, s_{i}, t_{i})`, in ``rst``, and the set of :math:`m` orthonormal basis functions, the van der
        Monde matrix will be :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.


        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.

        Returns:
            V (numpy.ndarray): the reference element van der Monde matrix of the orthonormal basis functions, :math:`f_{j}`, on the 3D simplices (elements of the mesh), i.e., :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.
        """

        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.modes.simplex_onb(dim, Nx)

        # Compute van der Monde matrix of simplex_basis over the nodes in rst
        return modepy.vandermonde(simplex_basis, rst)

    @staticmethod
    def compute_grad_van_der_monde_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
        """Compute the gradient van der Monde matrix.

        Computes the van der Monde matrices for the gradient of an orthonormal basis on the reference simplex. This polynomial basis
        can exactly represent polynomials up to degree ``Nx``.

        Consider the set of :math:`n` 3D nodes, with the coordinates of each node :math:`i` equal to
        :math:`(r_{i}, s_{i}, t_{i})`, in ``rst``, and the set of :math:`m` orthonormal basis functions, this van der
        Monde matrix will be :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.


        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.

        Returns:
            V (tuple of numpy.ndarray): the reference element van der Monde matrix of the gradient of the orthonormal 
                basis functions, :math:`\\nabla f_{j}`, on the 3D simplices (elements of the mesh), i.e.,
                :math:`\\left(V_{i,j}\\right)_{l} = \\left(\\nabla f_{j}(r_{i}, s_{i}, t_{i})\\right)_{l}`, where :math:`l`
                is one of the three components of the gradient :math:`(r, s, t)`.
                The order is the same as for the van der Monde matrix.
                V[0]: contains the :math:`r`-component of the gradient.
                V[1]: contains the :math:`s`-component of the gradient.
                V[2]: contains the :math:`t`-component of the gradient.
        """

        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.modes.grad_simplex_onb(dim, Nx)

        # Compute van der Monde matrix of the gradient of simplex_basis over the nodes in rst
        return modepy.vandermonde(simplex_basis, rst)
    
    @staticmethod
    def compute_mass_matrix(V: numpy.ndarray):
        """Compute the mass matrix from the van der Monde Matrix.

        Given the van der Monde matrix :math:`V`, compute the mass matrix :math:`M = V^{-t}V^{-1}`.


        Args:
            V (numpy.ndarray): the reference element van der Monde matrix of the orthonormal basis functions,
                :math:`f_{j}`, on the 3D simplices (elements of the mesh), i.e.,
                :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.

        Returns:
            M (numpy.ndarray): the reference element mass matrix :math:`M := V^{-t}V^{-1}`.
        """

        # Compute the inverse of the van der Monde matrix
        V_inv = numpy.linalg.inv(V)

        # Computes and returns the mass matrix V^{-t} V^{-1}
        return V_inv.transpose() @ V_inv

    @staticmethod
    def compute_derivative_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
        """Compute the derivative matrix.

        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.

        Returns:
            Return matrices carrying out differentiation on nodal values in the (r,s,t) unit directions. 
        """

        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.simplex_onb(dim,Nx)

        # Compute the grad of orthonormal polynomial basis of degree Nx and geometric dimension dim
        grad_simplex_basis = modepy.grad_simplex_onb(dim,Nx)


        # Compute differentiation matrix of simplex_basis over the nodes in rst, return a tuple D
        D = modepy.differentiation_matrices(simplex_basis, grad_simplex_basis, rst)

        # Return d/dr, d/ds and d/dt matrices
        return D[0], D[1], D[2]
    
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_Fmask(rst: numpy.ndarray, node_tol: float):
        """Find all the ``Nfp`` face nodes that lie on each surface.

        Args:
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            node_tol (float): the tolerance used to determine if a node lies on a facet.

        Returns:
            Fmask (numpy.ndarray): ``[4, Nfp]`` an array containing the local indices of the ``Nfp`` nodes on each of the four faces of 
                the reference element.
        """
        Np = rst.shape[1]  # get the number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(Nx)  # get the number of collocation points per face

        Fmask=numpy.zeros([4, Nfp],  dtype=numpy.uint8)

        # Find all the nodes that lie on each surface
        Fmask[0] = numpy.flatnonzero(numpy.abs(1+rst[2]) < node_tol)
        Fmask[1] = numpy.flatnonzero(numpy.abs(1+rst[1]) < node_tol)
        Fmask[2] = numpy.flatnonzero(numpy.abs(1+rst.sum(axis = 0 )) < node_tol)
        Fmask[3] = numpy.flatnonzero(numpy.abs(1+rst[0]) < node_tol)

        return Fmask
    
    @staticmethod
    def compute_lift(V: numpy.ndarray, rst: numpy.ndarray, Fmask: numpy.uint8):
        """Compute the lift matrix.

        Args:
            V (numpy.ndarray): ``[Np, Np]`` Vandermonde matrix for basis of degree ``Nx``.
            rst (numpy.ndarray): ``[3, Np]`` the reference element coordinates :math:`(r, s, t)` of the ``Np`` collocation points
                associated to a polynomial basis of degree ``Nx``.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            Fmask (numpy.ndarray): ``[4, Nfp]`` an array containing the local indices of the ``Nfp`` nodes on each of the four faces of 
                the reference element.
        Returns:
            Return lift (numpy.ndarray): ``[Np, 4*Nfp]`` an array containing the product of inverse of the mass matrix (3D) with the face-mass matrices (2D) 
        """
        Np = V.shape[1]  # get the number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

        Emat = numpy.zeros([Np, Nfp*4], dtype=numpy.float64)
        faceR = numpy.zeros([1, Nfp])
        faceS = numpy.zeros([1, Nfp])

        for face in range(4):
            if face == 0:
                faceR = rst[0, Fmask[0]]
                faceS = rst[1, Fmask[0]]

            elif face == 1:
                faceR = rst[0, Fmask[1]]
                faceS = rst[2, Fmask[1]]

            elif face == 2:
                faceR = rst[1, Fmask[2]]
                faceS = rst[2, Fmask[2]]
                
            else:
                faceR = rst[1, Fmask[3]]
                faceS = rst[2, Fmask[3]]
            
            simplex_basis = modepy.simplex_onb(2, Nx)
            vandermondeFace = modepy.vandermonde(simplex_basis, numpy.vstack((faceR,faceS)))   
            massFace = numpy.linalg.inv(vandermondeFace @ (vandermondeFace.transpose()))  

            Emat[Fmask[face], face*Nfp:(face+1)*Nfp] += massFace

            
        return V @ (V.transpose() @ Emat)

    @staticmethod
    def geometric_factors_3d(xyz: numpy.ndarray, Dr: numpy.ndarray, Ds: numpy.ndarray, Dt: numpy.ndarray):
        """Compute the metric elements for the local mappings of the elements.

        Args:
            xyz (numpy.ndarray): ``[3, Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each
                of the N_tets elements of the mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates,
                 ``xyz[2]`` contains the z-coordinates.
            Dr (numpy.ndarray): ``[Np, Np]`` the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial r}`.
            Ds (numpy.ndarray): ``[Np, Np]`` the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial s}`.
            Dt (numpy.ndarray): ``[Np, Np]`` the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial t}`.

        Returns:
            (tuple): tuple containing:
                rst_xyz (numpy.ndarray): ``[3, 3, Np, N_tets]`` The derivative of the local coordinates :math:`R = (r, s, t)` with 
                    respect to the physical coordinates :math:`X = (x, y, z)`, i.e., :math:`\\frac{\\partial R}{\\partial X}`, 
                    at the collocation nodes. Specifically:
                        rst_xyz[0, 0]: rx ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                            coordinates :math:`x`, i.e., :math:`\\frac{\\partial r}{\\partial x}`, at the collocation nodes.                      
                        rst_xyz[1, 0]: sx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                            coordinates :math:`x`, i.e., :math:`\\frac{\\partial s}{\\partial x}`, at the collocation nodes.
                        rst_xyz[2, 0]: tx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                            coordinates :math:`x`, i.e., :math:`\\frac{\\partial t}{\\partial x}`, at the collocation nodes.
                        rst_xyz[0, 1]: ry (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                            coordinates :math:`y`, i.e., :math:`\\frac{\\partial r}{\\partial y}`, at the collocation nodes.
                        rst_xyz[1, 1]: sy (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                            coordinates :math:`y`, i.e., :math:`\\frac{\\partial s}{\\partial y}`, at the collocation nodes.
                        rst_xyz[2, 1]: ty (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                            coordinates :math:`y`, i.e., :math:`\\frac{\\partial t}{\\partial y}`, at the collocation nodes.
                        rst_xyz[0, 2]: rz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                            coordinates :math:`z`, i.e., :math:`\\frac{\\partial r}{\\partial z}`, at the collocation nodes.
                        rst_xyz[1, 2]: sz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                            coordinates :math:`z`, i.e., :math:`\\frac{\\partial s}{\\partial z}`, at the collocation nodes.
                        rst_xyz[2, 2]: tz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                            coordinates :math:`z`, i.e., :math:`\\frac{\\partial t}{\\partial z}`, at the collocation nodes.
                J (numpy.ndarray): ``[Np, N_tets]`` The determinant of the Jacobian matrix for the coordinate transformation, 
                    at the collocation nodes. 
        """
        Np = xyz.shape[1]  # the number of collocation points
        N_tets = xyz.shape[2]  # the number of elements 

        # Compute the derivatives of the physical coordinates at the nodal points
        # x
        xr = Dr @ xyz[0]
        xs = Ds @ xyz[0]
        xt = Dt @ xyz[0]
        # y
        yr = Dr @ xyz[1]
        ys = Ds @ xyz[1]
        yt = Dt @ xyz[1]
        # z
        zr = Dr @ xyz[2]
        zs = Ds @ xyz[2]
        zt = Dt @ xyz[2]

        # Compute the Jacobian determinant of the coordinate transformation
        J = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt -ys * xt)

        # Compute the derivates of the local coordinates at the nodal points
        rst_xyz = numpy.zeros([3, 3, Np, N_tets])  # pre-allocate memory space

        # r
        rst_xyz[0, 0] =  (ys * zt - zs * yt) / J
        rst_xyz[0, 1] = -(xs * zt - zs * xt) / J
        rst_xyz[0, 2] = (xs * yt - ys * xt) / J
        # s
        rst_xyz[1, 0] = -(yr * zt - zr * yt) / J
        rst_xyz[1, 1] =  (xr * zt - zr * xt) / J
        rst_xyz[1, 2] = -(xr * yt - yr * xt) / J
        # t
        rst_xyz[2, 0] =  (yr * zs - zr * ys) / J
        rst_xyz[2, 1] = -(xr * zs - zr * xs) / J
        rst_xyz[2, 2] = (xr * ys - yr * xs) / J

        return rst_xyz, J

    @staticmethod
    def normals_3d(xyz: numpy.ndarray, rst_xyz: numpy.ndarray, J: numpy.array, Fmask: numpy.ndarray):
        """Compute outward pointing normals at element's faces as well as surface Jacobians.

        Args:
            xyz (numpy.ndarray): ``[3, Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each
                of the N_tets elements of the mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates,
                 ``xyz[2]`` contains the z-coordinates.
            rst_xyz (numpy.ndarray): ``[3, 3, Np, N_tets]`` The derivative of the local coordinates :math:`R = (r, s, t)` with 
                respect to the physical coordinates :math:`X = (x, y, z)`, i.e., :math:`\\frac{\\partial R}{\\partial X}`, 
                at the collocation nodes. Specifically:
                    rst_xyz[0, 0]: rx ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                        coordinates :math:`x`, i.e., :math:`\\frac{\\partial r}{\\partial x}`, at the collocation nodes.
                    rst_xyz[1, 0]: sx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                        coordinates :math:`x`, i.e., :math:`\\frac{\\partial s}{\\partial x}`, at the collocation nodes.
                    rst_xyz[2, 0]: tx (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                        coordinates :math:`x`, i.e., :math:`\\frac{\\partial t}{\\partial x}`, at the collocation nodes.
                    rst_xyz[0, 1]: ry (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                        coordinates :math:`y`, i.e., :math:`\\frac{\\partial r}{\\partial y}`, at the collocation nodes.
                    rst_xyz[1, 1]: sy (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                        coordinates :math:`y`, i.e., :math:`\\frac{\\partial s}{\\partial y}`, at the collocation nodes.
                    rst_xyz[2, 1]: ty (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                        coordinates :math:`y`, i.e., :math:`\\frac{\\partial t}{\\partial y}`, at the collocation nodes.
                    rst_xyz[0, 2]: rz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`r` with respect to the physical
                        coordinates :math:`z`, i.e., :math:`\\frac{\\partial r}{\\partial z}`, at the collocation nodes.
                    rst_xyz[1, 2]: sz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`s` with respect to the physical
                        coordinates :math:`z`, i.e., :math:`\\frac{\\partial s}{\\partial z}`, at the collocation nodes.
                    rst_xyz[2, 2]: tz (numpy.ndarray): ``[Np, N_tets]`` The derivative of the local coordinates :math:`t` with respect to the physical
                        coordinates :math:`z`, i.e., :math:`\\frac{\\partial t}{\\partial z}`, at the collocation nodes.
            J (numpy.ndarray): ``[Np, N_tets]`` The determinant of the Jacobian matrix for the coordinate transformation, 
                at the collocation nodes. 
            Fmask (numpy.ndarray): ``[4, Nfp]`` an array containing local indices of nodes for each of the four faces of the
                reference element.

        Returns:
            (tuple): tuple containing:
                n_xyz (numpy.ndarray): ``[3, 4*Np, N_tets]`` the outwards normals :math:`\\vec{n}` at each collocation
                    point on the element faces. Specifically:
                    n_xyz[0, :] (numpy.ndarray): nx ``[4*Nfp, N_tets]`` The :math:`x`-component of the outward normal :math:`\\vec{n}`
                        at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
                    n_xyz[1, :] (numpy.ndarray): ny ``[4*Nfp, N_tets]`` The :math:`y`-component of the outward normal :math:`\\vec{n}`
                        at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
                    n_xyz[2, :] (numpy.ndarray): nz ``[4*Nfp, N_tets]`` The :math:`z`-component of the outward normal :math:`\\vec{n}`
                        at each of the ``Nfp`` nodes on each of the 4 facets of each of the ``N_tets`` elements.
                sJ (numpy.ndarray): ``[4*Nfp, N_tets]`` The determinant of the surface Jacobian matrix at each of the
                    collocation nodes, for each of the 4 faces of the ``N_tets`` elements. 
        """
        N_tets = xyz.shape[2]  # number of elements
        Np = xyz.shape[1]  # number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

        # Extract the transformation derivatives over the faces
        # the structure is the same as for rst_xyz
        frst_xyz = rst_xyz[:, :, Fmask.reshape(-1), :]   

        # Construct the normals 
        n_xyz = numpy.zeros([3, 4*Nfp, N_tets])  # allocate memory space 

        face_0_idx = numpy.arange(0, Nfp)  # indices of the nodes of face 0
        face_1_idx = numpy.arange(Nfp, 2*Nfp)  # indices of the nodes of face 1
        face_2_idx = numpy.arange(2*Nfp, 3*Nfp)  # indices of the nodes of face 2
        face_3_idx = numpy.arange(3*Nfp, 4*Nfp)  # indices of the nodes of face 3

        # Face 0
        n_xyz[0, face_0_idx, :] = -frst_xyz[2, 0, face_0_idx, :]  # nx
        n_xyz[1, face_0_idx, :] = -frst_xyz[2, 1, face_0_idx, :]  # ny
        n_xyz[2, face_0_idx, :] = -frst_xyz[2, 2, face_0_idx, :]  # nz

        # Face 1
        n_xyz[0, face_1_idx, :] = -frst_xyz[1, 0, face_1_idx, :]  # nx
        n_xyz[1, face_1_idx, :] = -frst_xyz[1, 1, face_1_idx, :]  # ny
        n_xyz[2, face_1_idx, :] = -frst_xyz[1, 2, face_1_idx, :]  # nz

        # Face 2
        n_xyz[0, face_2_idx, :] = frst_xyz[0, 0, face_3_idx, :] + frst_xyz[1, 0, face_3_idx, :] + frst_xyz[2, 0, face_3_idx, :]  # nx
        n_xyz[1, face_2_idx, :] = frst_xyz[0, 1, face_3_idx, :] + frst_xyz[1, 1, face_3_idx, :] + frst_xyz[2, 1, face_3_idx, :]  # ny
        n_xyz[2, face_2_idx, :] = frst_xyz[0, 2, face_3_idx, :] + frst_xyz[1, 2, face_3_idx, :] + frst_xyz[2, 2, face_3_idx, :]  # nz

        # Face 3
        n_xyz[0, face_3_idx, :] = -frst_xyz[0, 0, face_3_idx, :]  # nx
        n_xyz[1, face_3_idx, :] = -frst_xyz[0, 1, face_3_idx, :]  # ny
        n_xyz[2, face_3_idx, :] = -frst_xyz[0, 2, face_3_idx, :]  # nz

        # Normalized the normal vectors
        norm_n = numpy.sqrt((n_xyz[0, :, :]**2) +  (n_xyz[1, :, :]**2) + (n_xyz[2, :, :]**2))  # vector norm of normal vectors
        n_xyz = n_xyz / norm_n  # this does broadcasting over all the 3 direction x, y, z
        
        # Compute the face Jacobians
        sJ = norm_n * J[Fmask.reshape(-1)]

        return n_xyz, sJ
    

    @staticmethod
    def build_maps_3d(xyz: numpy.ndarray,EToE: numpy.ndarray, EToF: numpy.ndarray, Fmask: numpy.ndarray, node_tol: float):
        """Find connectivity for nodes given per surface in all elements

        Args:
            xyz (numpy.ndarray): ``[3, Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each element of the\
            mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates, ``xyz[2]`` contains the z-coordinates.
            EToE (numpy.ndarray): an ``[4, N_tets]`` array containing the information of which elements are neighbors of
                an element, i.e., EToE[j, i] returns the index of the jth neighbor of element i. The definition of jth
                neighbor follows the mesh generator's convention.
            EToF (numpy.ndarray): an ``[4, N_tets]`` array containing the information of which face is shared between the
                element and its neighbor,  i.e.,  EToF[j, i] returns the face index of the jth neighbor of element i. 
                Face indices follow the same convention as neighbor indices.
            Fmask (numpy.ndarray): ``[4, Nfp]`` an array containing the local indices of the ``Nfp`` nodes on each of the four faces of 
                the reference element.
            node_tol (float): the tolerance used to determine if a node lies on a facet.

        Returns:
            (tuple): tuple containing:
                vmapM (numpy.ndarray): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for interior values.
                vmapP (numpy.ndarray): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for exterior values.
        """

        N_tets = xyz.shape[2]  # number of elements
        Np = xyz.shape[1]  # number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

        nodeids=numpy.arange(N_tets*Np, dtype=numpy.uint64).reshape(Np, N_tets)
        vmapM=numpy.zeros([4, Nfp, N_tets], dtype=numpy.uint64)
        vmapP=numpy.zeros([4, Nfp, N_tets], dtype=numpy.uint64)
        # tmp=numpy.ones([1, Nfp], dtype=numpy.uint8)
        tmp=numpy.ones(Nfp, dtype=numpy.uint8)
        D=numpy.zeros([Nfp, Nfp])

        xV=xyz[0].reshape(-1)  # viewing, get x in a 1D row-wise form, consistent with 
        yV=xyz[1].reshape(-1)
        zV=xyz[2].reshape(-1)

        for ke in range(N_tets):
            for face in range(4):
                vmapM[face, :, ke]=nodeids[Fmask[face],ke] #find index of face nodes with respect to volume node ordering

        for ke in range(N_tets):
            for face in range(4):
                #find neighbor
                ke2=EToE[face, ke]
                face2=EToF[face, ke]

                # find find volume node numbers of left and right nodes 
                vidM=vmapM[face, :, ke]
                vidP=vmapM[face2, :, ke2]

                # xM=numpy.outer(xyz[0].ravel(order='F')[vidM],tmp)  # returns a copy
                xM=numpy.outer(xV[vidM],tmp)  # viewing
                yM=numpy.outer(yV[vidM],tmp)  # viewing
                zM=numpy.outer(zV[vidM],tmp)  # viewing

                xP=numpy.outer(xV[vidP],tmp).transpose()  # viewing
                yP=numpy.outer(yV[vidP],tmp).transpose()  # viewing
                zP=numpy.outer(zV[vidP],tmp).transpose()  # viewing

                D=(xM-xP)**2 + (yM-yP)**2 + (zM-zP)**2

                (idM,idP)=numpy.nonzero(numpy.abs(D) < node_tol)

                vmapP[face,idM, ke]=vmapM[face2,idP, ke2]

        return vmapM.reshape(-1), vmapP.reshape(-1)
    
    @staticmethod
    # ismember_col function, which cols of a are in b:
    def ismember_col(a: numpy.ndarray, b: numpy.ndarray):
        _, rev = numpy.unique(numpy.concatenate((a,b),axis=1),axis=1,return_inverse=True) # The indices to reconstruct the original array from the unique array
        # Split the index
        b_rev = rev[a.shape[1]:]
        a_rev = rev[:a.shape[1]]
        # Return the result:
        return numpy.isin(a_rev,b_rev)

    @staticmethod
    def build_BCmaps_3d(BC_list: dict[str, int], EToV: numpy.ndarray, vmapM: numpy.ndarray, BC_triangles: dict[str, numpy.ndarray], Nx: int):
        """Build specialized nodal maps for various types of boundary conditions,specified in BC_list


        Args:
            BC_list (dict[str, int]): a dictionary containing the definition of the boundary
                conditions that are present in the mesh. BC_list.keys() must contain the same elements as
                mesh.BC_triangles.keys(), i.e., all boundary conditions in the mesh must have an associated boundary condition
                definition.
            EToV (numpy.ndarray): An (4 x self.N_tets) array containing the 4 indices of the vertices of the N_tets 
                tetrahedra that make up the mesh.
            vmapM (numpy.ndarray): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for interior values.
            BC_triangles (dict[str, numpy.ndarray]): a dictionary containing the list of triangles that have a certain
                boundary condition. BC_triangles['BC_label'] is a numpy.array with the nodes of each triangle where
                boundary condition of type 'BC_label' is to be implemented. The nodes defining each triangle in the
                numpy.array are stored per row.

        Returns:
            BCnode (list[dict]): List of boundary map nodes, each element being a dictionary 
                with keys (values) ['label'(int),'map'(numpy.ndarray),'vmap'(numpy.ndarray)]. 
        """
        Nfp = AcousticsSimulation.compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx
        N_tets = EToV.shape[1]
        BCType=numpy.zeros([4, N_tets], dtype=numpy.uint8)
        VNUM = numpy.array([[1, 2, 3],[1, 2, 4], [2, 3, 4], [1, 3, 4]])-1
        BCnode=[]
        for BCname, BClabel in BC_list.items():
            BCnode.append({'label': BClabel})
            # tri=BC_triangles[BCname].sort(axis=1)
            tri=numpy.sort(BC_triangles[BCname], axis=1).T
            for indexl in range(4):
                Face = numpy.sort(EToV[VNUM[indexl]], axis=0)
                K_ =AcousticsSimulation.ismember_col(Face, tri)
                # K_ = numpy.all(numpy.isin(Face, tri), axis=0) #wont work for all cases
                BCType[indexl, K_] = BClabel
        BCType=BCType.repeat(Nfp,axis=0)

        for i in range(len(BC_list)):
            BCnode[i]['map']=numpy.nonzero(BCType.reshape(-1)==BCnode[i]['label'])[0]
            # BCType.reshape(-1) and resulting 'map' first sweep through K, which is consistent with Nx.reshape(-1) and vmapM
            BCnode[i]['vmap']=vmapM[BCnode[i]['map']]

        return BCnode

    @staticmethod
    def dtscale_3d(Fscale: numpy.ndarray):
        # It calculate the minimum diameter of the inscribed spheres in all element, which is used as an indicator for the mesh size to determine maximum time step size
        Nfp = int(Fscale.shape[0] / 4)
        AtoV = 3 / 2 * Fscale[[0, Nfp, 2*Nfp, 3*Nfp]].sum(axis=0)
        diameter = 6 / AtoV
        return diameter.min()

    def grad_3d(self, U: numpy.ndarray, axis: str):
        """Compute partial derivative dU/dx, dU/dy, dU/dz, or gradient dU/dx + dU/dy + dU/dz

        Args:
            U (numpy.ndarray): ``[Np, N_tets]`` the acoustic variables that needs to be differentiated.
            axis (str): the axis to be differentiated w.r.t, e.g. 'x', 'y','z', 'xyz'

        Returns:
            (possible tuple):  containing:
                dUdx/dUdy/dUdz (numpy.ndarray): ``[Np, N_tets]`` derivatives/gradient at every nodal point
        """
        dUdr = self.Dr @ U
        dUds = self.Ds @ U
        dUdt = self.Dt @ U
        if axis == 'x':
            return self.rst_xyz[0, 0] * dUdr + self.rst_xyz[1, 0] * dUds + self.rst_xyz[2, 0] * dUdt
        elif axis == 'y':
            return self.rst_xyz[0, 1] * dUdr + self.rst_xyz[1, 1] * dUds + self.rst_xyz[2, 1] * dUdt
        elif axis == 'z':
            return self.rst_xyz[0, 2] * dUdr + self.rst_xyz[1, 2] * dUds + self.rst_xyz[2, 2] * dUdt
        elif axis == 'xyz':
            return self.rst_xyz[0, 0] * dUdr + self.rst_xyz[1, 0] * dUds + self.rst_xyz[2, 0] * dUdt, \
                   self.rst_xyz[0, 1] * dUdr + self.rst_xyz[1, 1] * dUds + self.rst_xyz[2, 1] * dUdt, \
                   self.rst_xyz[0, 2] * dUdr + self.rst_xyz[1, 2] * dUds + self.rst_xyz[2, 2] * dUdt


        
    @staticmethod
    #https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339#60745339
    def locate_simplex(node_coordinates: numpy.ndarray, node_ids: numpy.ndarray, rec: numpy.ndarray, methodLocate = 'scipy'):  
        """
        brutal force approach, adopted from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339#60745339

        Args:
            EToV (numpy.ndarray): An (4 x self.N_tets) array containing the 4 indices of the vertices of the N_tets
               tetrahedra that make up the mesh.
            vertices (numpy.ndarray): An (3 x self.N_vertices) array containing the xyz coordinates of the self.N_vertices
                vertices that make up the mesh. M specifies the geometric dimension of the mesh, such that the mesh
                describes an M-dimensional domain.
            node_coordinates (numpy.ndarray): (self.N_vertices,3) array containing the coordinates of each node
            node_ids (numpy.ndarray): An (4 x self.N_tets) array containing the 4 indices of the vertices of the N_tets
               tetrahedra that make up the mesh.
            An (M x self.N_vertices) array containing the M coordinates of the self.N_vertices
                vertices that make up the mesh. M specifies the geometric dimension of the mesh, such that the mesh
                describes an M-dimensional domain.
            rec (numpy.ndarray): An (N_rec x 3) array containing the (x, y, z) coordinates of N_rec microphone locations.  

        Returns:
            res (numpy.ndarray): indices of simplices containing the N_rec microphone points
        """
        if methodLocate == 'scipy':
            tri = Delaunay(node_coordinates.T, qhull_options = 'QJ')
            tri.simplices = node_ids.T
            tri.nsimplex = node_ids.shape[1]

            nodeindex = tri.find_simplex(rec.T)
        elif methodLocate == 'brute_force':
            node_ids = node_ids.T
            ori=node_coordinates.T[node_ids[:,0],:]
            v1=node_coordinates.T[node_ids[:,1],:]-ori
            v2=node_coordinates.T[node_ids[:,2],:]-ori
            v3=node_coordinates.T[node_ids[:,3],:]-ori
            n_tet=len(node_ids)
            v1r=v1.T.reshape((3,1,n_tet))
            v2r=v2.T.reshape((3,1,n_tet))
            v3r=v3.T.reshape((3,1,n_tet))
            mat = numpy.concatenate((v1r,v2r,v3r), axis=1)
            inv_mat = numpy.linalg.inv(mat.T).T    # https://stackoverflow.com/a/41851137/12056867        
            # if rec.size==3:  # to make rec has a dimension of (N_rec,3)
            #     rec=rec.reshape((1,3))
            N_rec=rec.shape[1]
            orir=numpy.repeat(ori[:,:,numpy.newaxis], N_rec, axis=2)
            newp=numpy.einsum('imk,kmj->kij',inv_mat,rec-orir)
            val=numpy.all(newp>=0, axis=1) & numpy.all(newp <=1, axis=1) & (numpy.sum(newp, axis=1)<=1)
            id_tet, id_p = numpy.nonzero(val)
            nodeindex = -numpy.ones(N_rec, dtype=id_tet.dtype) # Sentinel value
            nodeindex[id_p]=id_tet
        else:
            raise ValueError("{method} is not an available search method, see documentation for available methods".format(method=methodLocate))
            
        return nodeindex
        

    def sample3D(self, methodLocate):
        """
        Compute interpolation weights required to interpolate the nodal data to the sample (i.e., microphone location)

        Args:
            methodLocate (str): search method to locate the simplices containing the sample points

        Returns:
            sampleWeight (numpy.ndarray): ``[N_rec, Np]`` interpolation weights required to interpolate the nodal data to the sample (i.e., microphone location).
            nodeindex (numpy.ndarray): ``[N_rec, ]`` index of simplices that contatin the sample (microphone) points
        """
        # self.mesh.EToV, self.mesh.vertices, self.xyz, rec, self.Nx
        nodeindex = AcousticsSimulation.locate_simplex(self.mesh.vertices, self.mesh.EToV, self.rec, methodLocate)

        old_nodes = self.xyz[:,:,nodeindex]
        simplex_basis = modepy.modes.simplex_onb(self.dim, self.Nx)
        v_new = modepy.vandermonde(simplex_basis, self.rec)
        sampleWeight = numpy.zeros([self.rec.shape[1], len(simplex_basis)])

        for i in range(old_nodes.shape[2]):
            v_old = modepy.vandermonde(simplex_basis, old_nodes[:,:,i])
            sampleWeight[i] = v_new[i] @ numpy.linalg.inv(v_old)

        return sampleWeight, nodeindex
    

    # instance method -------------------------------------------------------------------------------------------------------

    def init_IC(self, IC: edg_acoustics.InitialCondition):
        """setup the initial condition.
        Args:

        Returns:
        """
        # self.IC = IC
        self.P0 = IC.Pinit(self.xyz)
        self.Vx0 = IC.VXinit(self.xyz)
        self.Vy0 = IC.VYinit(self.xyz)
        self.Vz0 = IC.VZinit(self.xyz)

    def init_BC(self, BC: edg_acoustics.BoundaryCondition):
        """setup the boundary condition.

        
        Args:
        
        Returns:
        """
        # self.BC = edg_acoustics.BoundaryCondition(self.BCnode, BC_para)
        self.BC = BC

    def init_rec(self, rec: numpy.ndarray, methodLocate: str ='scipy'):
        """setup the receiver locations.
        Args:
        
        Returns:
        """
        self.rec = rec
        self.sampleWeight, self.nodeindex = self.sample3D(methodLocate)


    def init_Flux(self, Flux: edg_acoustics.Flux):
        """setup the interior flux  calculation.

        
        Args:

        Returns:
        """
        self.Flux = Flux
        # self.Flux = edg_acoustics.UpwindFlux(self.rho0, self.c0, self.n_xyz)

    def init_TimeIntegrator(self, time_integrator: edg_acoustics.TimeIntegrator):
        """
        perform the time integration

        
        Args:

        Returns:
        """
        self.time_integrator = time_integrator


    def time_integration(self, rec: numpy.ndarray, **kwargs):
        """
        perform the time integration

        
        Args:

        kwargs:
            n_time_steps (int): number of time steps to compute.
            total_time (float): total simulation time to compute, determines the number of time steps given the current time step.

        Returns:
        """

        # Process optional input arguments
        if ("n_time_steps" in kwargs and "total_time" in kwargs):
            # Not possible to input both number of steps and total simulation time
            raise ValueError("Set only n_time_steps or total_time, do not set both...")
        
        elif ("n_time_steps" in kwargs):
            # Directly use the number of timesteps
            self.Ntimesteps = kwargs['n_time_steps'] 

        elif ("total_time" in kwargs):
            # Compute the number of time steps from the total simulation time and the time step size of the time integrator
            total_time = kwargs['total_time']
            self.Ntimesteps = math.floor(total_time / self.time_integrator.dt)  # apalha: I would make this math.ceil, because you want to simulate a bit more and not less

        else:
            raise ValueError("You need to set n_time_steps or total_time...")
        

        

        print(f"Total simulation time is {total_time}")
        print(f"Total number of simulation steps is {self.Ntimesteps}")

        self.prec = numpy.zeros([rec.shape[1], self.Ntimesteps])
        self.sampleWeight, self.nodeindex = AcousticsSimulation.sample3D(self.mesh.EToV, self.mesh.vertices, self.xyz, rec, self.Nx)

        # Reset the time step variables
        self.P = self.P0.copy()
        self.Vx = self.Vx0.copy()
        self.Vy = self.Vy0.copy()
        self.Vz = self.Vz0.copy()

        # Step the solution
        for StepIndex in range(self.Ntimesteps):
            print(f"Current/Total step {StepIndex+1}/{self.Ntimesteps}")
            print(f"Current/Total time {self.time_integrator.dt * StepIndex}/{total_time}")
            # print(f"outside, self.P ID {id(self.P)}, self.P0 ID {id(self.P0)}")
            # print(f"outside, self.BC.BCvar ID {id(self.BC.BCvar)}")

            # Reset the initial condition to prepare for the new time step
            # For the first step this does nothing, but for the next steps it does the work
            # Resetting before is better than after because in this way the previous step and
            # current step solutions are available, which can always be useful.
            self.P0 = self.P.copy()
            self.Vx0 = self.Vx.copy()
            self.Vy0 = self.Vy.copy()
            self.Vz0 = self.Vz.copy()

            self.P, self.Vx, self.Vy, self.Vz = self.time_integrator.step_dt_new(self.P0, self.Vx0, self.Vy0, self.Vz0) 
            
            self.prec[:,StepIndex] = numpy.diag(self.sampleWeight @ self.P[:,self.nodeindex])

            # print(f"outside,after loop, self.P ID {id(self.P)}, self.P0 ID {id(self.P0)}")
            # print(f"outside after loop, self.BC.BCvar ID {id(self.BC.BCvar)}")

            print(f"P at mic locations {self.prec[:,StepIndex]}")

    def init_TimeIntegration(self, time_integrator: edg_acoustics.TimeIntegrator, rec: numpy.ndarray, TotalTime: float=0.05):
        """
        perform the time integration

        
        Args:

        Returns:
        """
        # create new attributes, storing the time-dependent acoustic variables
        self.P = self.P0.copy()
        self.Vx = self.Vx0.copy()
        self.Vy = self.Vy0.copy()
        self.Vz = self.Vz0.copy()

        self.Ntimesteps = int(TotalTime / time_integrator.dt)
        print(f"Total simulation time is {TotalTime}")
        print(f"Total number of simulation steps is {self.Ntimesteps}")

        self.prec = numpy.zeros([self.rec.shape[1], self.Ntimesteps])


        for StepIndex in range(self.Ntimesteps):
            print(f"Current/Total step {StepIndex+1}/{self.Ntimesteps}")
            print(f"Current/Total time {time_integrator.dt * StepIndex}/{TotalTime}")
            # print(f"outside, self.P ID {id(self.P)}, self.P0 ID {id(self.P0)}")
            # print(f"outside, self.BC.BCvar ID {id(self.BC.BCvar)}")


            time_integrator.step_dt() 
            
            self.P0 = self.P.copy()
            self.Vx0 = self.Vx.copy()
            self.Vy0 = self.Vy.copy()
            self.Vz0 = self.Vz.copy()
            self.prec[:,StepIndex] = numpy.diag(self.sampleWeight @ self.P[:,self.nodeindex])

            # print(f"outside,after loop, self.P ID {id(self.P)}, self.P0 ID {id(self.P0)}")
            # print(f"outside after loop, self.BC.BCvar ID {id(self.BC.BCvar)}")

            print(f"P at mic locations {self.prec[:,StepIndex]}")
  

    def L(self, P, Vx, Vy, Vz):
        # Initialize variables
        dVx = numpy.zeros_like(self.Fscale)
        dVy = numpy.zeros_like(dVx)
        dVz = numpy.zeros_like(dVx)
        dP = numpy.zeros_like(dVx)

        # apalha: Some description needed
        dVx.reshape(-1)[:] = Vx.reshape(-1)[self.vmapM] - Vx.reshape(-1)[self.vmapP]
        # print(f"dVx ID {id(dVx)}, sim.dVx ID {id(self.sim.dVx)}")
        dVy.reshape(-1)[:] = Vy.reshape(-1)[self.vmapM] - Vy.reshape(-1)[self.vmapP]
        dVz.reshape(-1)[:] = Vz.reshape(-1)[self.vmapM] - Vz.reshape(-1)[self.vmapP]
        dP.reshape(-1)[:] = P.reshape(-1)[self.vmapM] - P.reshape(-1)[self.vmapP]

        # Compute the inter-element fluxes
        fluxVx = self.Flux.FluxVx(dVx, dVy, dVz, dP)  # has return object, might make copy, 
        fluxVy = self.Flux.FluxVy(dVx, dVy, dVz, dP)
        fluxVz = self.Flux.FluxVz(dVx, dVy, dVz, dP)
        fluxP = self.Flux.FluxP(dVx, dVy, dVz, dP)

        for index, paras in enumerate(self.BC.BCpara):
            # 'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`. 
            # 'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row). 
            #     'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
            #          :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
            self.BC.BCvar[index]['vn'] = self.n_xyz[0].reshape(-1)[self.BCnode[index]['map']] * Vx.reshape(-1)[self.BCnode[index]['vmap']] + \
                            self.n_xyz[1].reshape(-1)[self.BCnode[index]['map']] * Vy.reshape(-1)[self.BCnode[index]['vmap']] + \
                            self.n_xyz[2].reshape(-1)[self.BCnode[index]['map']] * Vz.reshape(-1)[self.BCnode[index]['vmap']] 
            self.BC.BCvar[index]['ou'] = self.BC.BCvar[index]['vn'] + P.reshape(-1)[self.BCnode[index]['vmap']] / self.rho0 / self.c0
            self.BC.BCvar[index]['in'] = self.BC.BCvar[index]['ou'] * paras['RI']

            for polekey in paras:
                if polekey == 'RP':
                        for i in range(paras['RP'].shape[1]):
                            self.BC.BCvar[index]['in'] = self.BC.BCvar[index]['in'] + paras['RP'][0,i] * self.BC.BCvar[index]['phi'][i]
                elif polekey=='CP':
                    pass # to be added

            fluxVx.reshape(-1)[self.BCnode[index]['map']] = self.n_xyz[0].reshape(-1)[self.BCnode[index]['map']] * P.reshape(-1)[self.BCnode[index]['vmap']] /self.rho0 - \
                                                    self.n_xyz[0].reshape(-1)[self.BCnode[index]['map']] * self.c0 * (self.BC.BCvar[index]['ou'] + self.BC.BCvar[index]['in']) / 2
            fluxVy.reshape(-1)[self.BCnode[index]['map']] = self.n_xyz[1].reshape(-1)[self.BCnode[index]['map']] * P.reshape(-1)[self.BCnode[index]['vmap']] /self.rho0 - \
                                                    self.n_xyz[1].reshape(-1)[self.BCnode[index]['map']] * self.c0 * (self.BC.BCvar[index]['ou'] + self.BC.BCvar[index]['in']) / 2
            fluxVz.reshape(-1)[self.BCnode[index]['map']] = self.n_xyz[2].reshape(-1)[self.BCnode[index]['map']] * P.reshape(-1)[self.BCnode[index]['vmap']] /self.rho0 - \
                                                    self.n_xyz[2].reshape(-1)[self.BCnode[index]['map']] * self.c0 * (self.BC.BCvar[index]['ou'] + self.BC.BCvar[index]['in']) / 2
            fluxP.reshape(-1)[self.BCnode[index]['map']] = self.c0**2 * self.rho0 * (self.BC.BCvar[index]['vn'] - 0.5 * (self.BC.BCvar[index]['ou'] - self.BC.BCvar[index]['in']))

        dPdx, dPdy, dPdz = self.grad_3d(P, 'xyz')
        L_P = -self.c0**2 * self.rho0 * (self.grad_3d(Vx, 'x') + self.grad_3d(Vy, 'y') + 
                            self.grad_3d(Vz, 'z')) + self.lift @ (self.Fscale * fluxP)
        L_Vx = -dPdx / self.rho0 + self.lift @ (self.Fscale * fluxVx)
        L_Vy = -dPdy / self.rho0 + self.lift @ (self.Fscale * fluxVy)
        L_Vz = -dPdz / self.rho0 + self.lift @ (self.Fscale * fluxVz)

        return L_P, L_Vx, L_Vy, L_Vz

    # Properties -------------------------------------------------------------------------------------------------------
    # @property
    # def EToV(self):
    #     """numpy.ndarray: An (self.N_tets x 4) array containing the 4 indices of the vertices of the self.N_tets
    #            tetrahedra that make up the mesh. It returns the value in self.tets, since it is the same data."""
    #     return self.tets
    # ------------------------------------------------------------------------------------------------------------------
