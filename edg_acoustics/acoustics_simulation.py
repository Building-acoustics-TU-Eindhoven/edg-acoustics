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
import edg_acoustics


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

  
    def __init__(self, Nx: int, Nt: int, mesh: edg_acoustics.Mesh, BC_list: dict[str, int], node_tolerance: float = NODETOL):
        # Check if BC_list and mesh are compatible
        if not self.__check_BC_list(BC_list, mesh):
            raise ValueError(
                "[edg_acoustics.AcousticSimulation] All BC labels must be present in the mesh and all labels in the mesh must be "
                "present in BC_list.")

        # Store input parameters
        self.mesh = mesh
        self.Nx = Nx
        self.Nt = Nt
        self.BC_list = BC_list
        self.dim = 3  # we are always in 3D, just added for external reference     
        self.node_tolerance = node_tolerance  # define a tolerance value for determining if a node belongs to a facet or not 
  
        # Compute attributes
        self.Np = AcousticsSimulation.__compute_Np(Nx)  # number of colocation nodes in an element
        self.Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # number of nodes in a face
      
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

        self.rst, self.xyz = self.__compute_collocation_nodes(
            self.mesh.EToV, self.mesh.vertices, self.Nx, dim=self.dim)

        # Compute the van der Monde matrix and its inverse
        self.V = self.__compute_van_der_monde_matrix(self.Nx, self.rst)
        self.inV = numpy.linalg.inv(self.V)

        # Compute the van der Monde matrix of the gradients
        self.V3D = self.__compute_grad_van_der_monde_matrix(self.Nx, self.rst)

        # Compute mass matrix
        self.M = self.__compute_mass_matrix(self.V)

        # Compute the derivative matrices
        self.Dr, self.Ds, self.Dt = self.__compute_derivative_matrix(self.Nx, self.rst)

        # Find all the ``Nfp`` face nodes that lie on each surface.
        self.Fmask=self.__compute_Fmask(self.rst, self.node_tolerance)

        # Compute the product of inverse of the mass matrix (3D) with the face-mass matrices (2D)
        self.lift=self.__compute_lift(self.V, self.rst, self.Fmask)

        # Compute the metric terms for the mesh
        self.rst_xyz, self.J = self.__geometric_factors_3d(self.xyz, self.Dr, self.Ds, self.Dt)

        # Compute the face normals at the collocation points and the surface Jacobians
        self.n_xyz, self.sJ = self.__normals_3d(self.xyz, self.rst_xyz, self.J, self.Fmask)

        # Compute ratio of surface to volume Jacobian of facial node
        self.Fscale = self.sJ / self.J [self.Fmask.reshape(-1), :]

        # Find connectivity for nodes given per surface in all elements
        self.vmapM, self.vmapP = self.__build_maps_3d(self.xyz, self.mesh.EToE, self.mesh.EToF, self.Fmask, self.node_tolerance)

        # Build specialized nodal maps for various types of boundary conditions,specified in BC_list
        self.BCnode = self.__build_BCmaps_3d(self.BC_list, self.mesh.EToV, self.vmapM, self.mesh.BC_triangles, self.Nx)
            

        # print(self.vmapM)
    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def __compute_Np(Nx: int):
        """Computes the number of collocation nodes for basis of polynomial degree ``Nx``.

        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.

        Returns:
            Np (int): number of collocation nodes in an element.
        """
        
        return int((Nx+1)*(Nx+2)*(Nx+3)/6)

    def __compute_Nfp(Nx: int):
        """Computes the number of collocation nodes lying on a face of the elements for basis of polynomial degree ``Nx``.

        Args:
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.

        Returns:
            Nfp (int): number of collocation nodes in a face.
        """
        
        return int((Nx+1)*(Nx+2)/2)
    
    def __compute_Nx_from_Np(Np: int):
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
    def __check_BC_list(BC_list: dict[str, int], mesh: edg_acoustics.Mesh):
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
    def __compute_collocation_nodes(EToV: numpy.ndarray, vertices: numpy.ndarray, Nx: int, dim: int = 3):
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
    def __compute_van_der_monde_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
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
    def __compute_grad_van_der_monde_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
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
    def __compute_mass_matrix(V: numpy.ndarray):
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
    def __compute_derivative_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
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
    def __compute_Fmask(rst: numpy.ndarray, node_tol: float):
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
        Nx = AcousticsSimulation.__compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # get the number of collocation points per face

        Fmask=numpy.zeros([4, Nfp], dtype=numpy.uint8)

        # Find all the nodes that lie on each surface
        Fmask[0] = numpy.flatnonzero(numpy.abs(1+rst[2]) < node_tol)
        Fmask[1] = numpy.flatnonzero(numpy.abs(1+rst[1]) < node_tol)
        Fmask[2] = numpy.flatnonzero(numpy.abs(1+rst.sum(axis=0)) < node_tol)
        Fmask[3] = numpy.flatnonzero(numpy.abs(1+rst[0]) < node_tol)

        return Fmask
    
    @staticmethod
    def __compute_lift(V: numpy.ndarray, rst: numpy.ndarray, Fmask: numpy.uint8):
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
        Nx = AcousticsSimulation.__compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

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
    def __geometric_factors_3d(xyz: numpy.ndarray, Dr: numpy.ndarray, Ds: numpy.ndarray, Dt: numpy.ndarray):
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
    def __normals_3d(xyz: numpy.ndarray, rst_xyz: numpy.ndarray, J: numpy.array, Fmask: numpy.ndarray):
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
        Nx = AcousticsSimulation.__compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

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
    def __build_maps_3d(xyz: numpy.ndarray,EToE: numpy.ndarray, EToF: numpy.ndarray, Fmask: numpy.ndarray, node_tol: float):
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
        Nx = AcousticsSimulation.__compute_Nx_from_Np(Np)  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx

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
    def __ismember_col(a: numpy.ndarray, b: numpy.ndarray):
        _, rev = numpy.unique(numpy.concatenate((a,b),axis=1),axis=1,return_inverse=True) # The indices to reconstruct the original array from the unique array
        # Split the index
        b_rev = rev[a.shape[1]:]
        a_rev = rev[:a.shape[1]]
        # Return the result:
        return numpy.isin(a_rev,b_rev)

    @staticmethod
    def __build_BCmaps_3d(BC_list: dict[str, int], EToV: numpy.ndarray, vmapM: numpy.ndarray, BC_triangles: dict[str, numpy.ndarray], Nx: int):
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
        Nfp = AcousticsSimulation.__compute_Nfp(Nx)  # the number of nodes per surface for basis of polynomial degree Nx
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
                K_ =AcousticsSimulation.__ismember_col(Face, tri)
                # K_ = numpy.all(numpy.isin(Face, tri), axis=0) #wont work for all cases
                BCType[indexl, K_] = BClabel
        BCType=BCType.repeat(Nfp,axis=0)

        for i in range(len(BC_list)):
            BCnode[i]['map']=numpy.nonzero(BCType.reshape(-1)==BCnode[i]['label'])[0]
            # BCType.reshape(-1) and resulting 'map' first sweep through K, which is consistent with Nx.reshape(-1) and vmapM
            BCnode[i]['vmap']=vmapM[BCnode[i]['map']]

        return BCnode




    # Properties -------------------------------------------------------------------------------------------------------
    # @property
    # def EToV(self):
    #     """numpy.ndarray: An (self.N_tets x 4) array containing the 4 indices of the vertices of the self.N_tets
    #            tetrahedra that make up the mesh. It returns the value in self.tets, since it is the same data."""
    #     return self.tets
    # ------------------------------------------------------------------------------------------------------------------
