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

    :class:`.AcousticsSimulation`contains the domain discretization and sets up the DG finite element discretisation
    for the solution to the acoustic wave propagation.

    Args:
        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.
        Nt (int): the order of the time integration scheme.
        mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
        BC_list (dict[str, edg_acoustics.BCCondition]): a dictionary containing the definition of the boundary
            conditions that are present in the mesh. BC_list.keys() must contain the same elements as
            mesh.BC_triangles.keys(), i.e., all boundary conditions in the mesh must have an associated boundary condition
            definition.
        node_tolerance (float): the tolerance used to check if a node belongs to a facet or not.
            <default>: edg_acoustics.NODETOL

    Raises:
        ValueError: If BC_list['my_label'] is not present in the mesh, an error is raised. If a label
            is present in the mesh but not in BC_list, an error is raised.

    Attributes:
        BC_list (dict[str, edg_acoustics.BCCondition]): a dictionary containing the definition of the boundary
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
        Fmask: Needs info.
        lift: Needs info.
        M (numpy.ndarray): the reference element mass matrix :math:`M := V^{-t}V^{-1}`.
        mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
        Nfp (int): number of nodes in a facet.
        node_tolerance (float): tolerance used to determine if a node lies on a facet.
        Np (int): number of nodes in an element.
        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.
        Nt (int): the order of the time integration scheme.
        rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
            ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
            the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates, ``rst[2]``
            contains the t-coordinates.
        V (numpy.ndarray): the reference element van der Monde matrix of the orthonormal basis functions, :math:`f_{j}`, on the
            3D simplices (elements of the mesh), i.e., :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.
        xyz (numpy.ndarray): the physical space coordinates :math:`(x, y, z)` of the collocation points of each element of the\
            mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates, ``xyz[2]``
            contains the z-coordinates.

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
        self.Np = int((Nx+1)*(Nx+2)*(Nx+3)/6)  # number of nodes in an element
        self.Nfp =int((Nx+1)*(Nx+2)/2)  # number of nodes in a facet
      
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

        # Compute mass matrix
        self.M = self.__compute_mass_matrix(self.V)

        # Compute the derivative matrices
        self.Dr, self.Ds, self.Dt = self.__compute_derivative_matrix(self.Nx, self.rst)

        self.Fmask=self.__compute_Fmask(self.rst, self.Nfp, self.node_tolerance)

        self.lift=self.__compute_lift(self.V,self.Nx,self.rst,self.Fmask,self.Np, self.Nfp)

        # Compute the metric terms for the mesh
        self.rx, self.ry, self.rz, \
            self.sx, self.sy, self.sz, \
            self.tx, self.ty, self.tz, \
            self.J = self.__geometric_factors_3d(self.xyz, self.Dr, self.Ds, self.Dt)

    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def __check_BC_list(BC_list: dict[str, int], mesh: edg_acoustics.Mesh):
        """Check if BC_list is compatible with mesh.

        Given a mesh with a set of boundary conditions specified in mesh.BC_triangles, check if the list of boundary
        conditions specification, BC_list, is compatible. By compatible we mean that all boundary conditions (keys) in
        BC_list exist in mesh.BC_labels, and vice-versa.

        Args:
            mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
            BC_list (dict[str, edg_acoustics.BCCondition]): a dictionary containing the definition of the boundary
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
            xyz (numpy.ndarray): the physical space coordinates :math:`(x, y, z)` of the collocation points of each
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

        Computes the vander Monde matrix for an orthonormal basis on the reference simplex. This polynomial basis
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
            V (numpy.ndarray): the reference element van der Monde matrix of the orthonormal basis functions,
                :math:`f_{j}`, on the 3D simplices (elements of the mesh), i.e.,
                :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.
        """

        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.modes.simplex_onb(dim,Nx)

        # Compute van der Monde matrix of simplex_basis over the nodes in rst
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


        # Compute differentiation matrix of simplex_basis over the nodes in rst
        D = modepy.differentiation_matrices(simplex_basis, grad_simplex_basis, rst)

        # Return d/dr, d/ds and d/dt matrices
        return D[0], D[1], D[2]
    
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __compute_Fmask(rst: numpy.ndarray, Nfp: int, node_tol: float):
        """Compute the face nodes indices.

        Args:
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            Nfp (int): number of nodes per surface.
            node_tol (float): the tolerance used to determine if a node lies on a facet.

        Returns:
            Fmask (numpy.ndarray): a (4 x Nfp) array containing indices of nodes per surface
        """
        Fmask=numpy.zeros([4, Nfp], dtype=numpy.uint8)

        # Find all the nodes that lie on each surface
        Fmask[0] = numpy.flatnonzero(numpy.abs(1+rst[2]) < node_tol)
        Fmask[1] = numpy.flatnonzero(numpy.abs(1+rst[1]) < node_tol)
        Fmask[2] = numpy.flatnonzero(numpy.abs(1+rst.sum(axis=0)) < node_tol)
        Fmask[3] = numpy.flatnonzero(numpy.abs(1+rst[0]) < node_tol)

        return Fmask
    
    @staticmethod
    def __compute_lift(V: numpy.ndarray, Nx: int, rst: numpy.ndarray, Fmask: numpy.uint8, Np: int, Nfp: int):
        """Compute the lift matrix.

        Args:
            V (numpy.ndarray): Vandermonde matrix.
            Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
                propagation problem.
            rst (numpy.ndarray): the reference element coordinates :math:`(r, s, t)` of the collocation points.
                ``xyz`` are obtained by mapping for each element the ``rst`` coordinates of the reference element into
                the physical domain. ``rst[0]`` contains the r-coordinates, ``rst[1]`` contains the s-coordinates,
                ``rst[2]`` contains the t-coordinates.
            Fmask (numpy.ndarray): an (4*Nfp) array containing indices of nodes per surface.
            Np (int): sds.
            Nfp (int): number of nodes per surface.

        Returns:
            Return lift matrices carrying out surface integral. 
        """
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

        print(Emat)
            
        return V @ (V.transpose() @ Emat)

    @staticmethod
    def __geometric_factors_3d(xyz: numpy.ndarray, Dr: numpy.ndarray, Ds: numpy.ndarray, Dt: numpy.ndarray):
        """Compute the metric elements for the local mappings of the elements.

        Args:
            xyz (numpy.ndarray): the physical space coordinates :math:`(x, y, z)` of the collocation points of each
                element of the mesh. ``xyz[0]`` contains the x-coordinates, ``xyz[1]`` contains the y-coordinates,
                 ``xyz[2]`` contains the z-coordinates.
            Dr (numpy.ndarray): the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial r}`.
            Ds (numpy.ndarray): the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial s}`.
            Dt (numpy.ndarray): the differentiation matrix on the collation points implementing the discrete version of
                :math:`\\frac{\\partial}{\\partial t}`.

        Returns:
            (tuple): tuple containing:
                rx (numpy.ndarray): The derivative of the local coordinates :math:`r` with respect to the physical
                    coordinates :math:`x`, i.e., :math:`\\frac{\\partial r}{\\partial x}`, at the collocation nodes.
                sx (numpy.ndarray): The derivative of the local coordinates :math:`s` with respect to the physical
                    coordinates :math:`x`, i.e., :math:`\\frac{\\partial s}{\\partial x}`, at the collocation nodes.
                tx (numpy.ndarray): The derivative of the local coordinates :math:`t` with respect to the physical
                    coordinates :math:`x`, i.e., :math:`\\frac{\\partial t}{\\partial x}`, at the collocation nodes.
                ry (numpy.ndarray): The derivative of the local coordinates :math:`r` with respect to the physical
                    coordinates :math:`y`, i.e., :math:`\\frac{\\partial r}{\\partial y}`, at the collocation nodes.
                sy (numpy.ndarray): The derivative of the local coordinates :math:`s` with respect to the physical
                    coordinates :math:`y`, i.e., :math:`\\frac{\\partial s}{\\partial y}`, at the collocation nodes.
                ty (numpy.ndarray): The derivative of the local coordinates :math:`t` with respect to the physical
                    coordinates :math:`y`, i.e., :math:`\\frac{\\partial t}{\\partial y}`, at the collocation nodes.
                rz (numpy.ndarray): The derivative of the local coordinates :math:`r` with respect to the physical
                    coordinates :math:`z`, i.e., :math:`\\frac{\\partial r}{\\partial z}`, at the collocation nodes.
                sz (numpy.ndarray): The derivative of the local coordinates :math:`s` with respect to the physical
                    coordinates :math:`z`, i.e., :math:`\\frac{\\partial s}{\\partial z}`, at the collocation nodes.
                tz (numpy.ndarray): The derivative of the local coordinates :math:`t` with respect to the physical
                    coordinates :math:`z`, i.e., :math:`\\frac{\\partial t}{\\partial z}`, at the collocation nodes.
                J (numpy.ndarray): The determinant of the Jacobian matrix for the coordinate transformation, 
                    at the collocation nodes. 
        """
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
        # r
        rx =  (ys * zt - zs * yt) / J
        ry = -(xs * zt - zs * xt) / J
        rz = (xs * yt - ys * xt) / J
        # s
        sx = -(yr * zt - zr * yt) / J
        sy =  (xr * zt - zr * xt) / J
        sz = -(xr * yt - yr * xt) / J
        # t
        tx =  (yr * zs - zr * ys) / J
        ty = -(xr * zs - zr * xs) / J
        tz = (xr * ys - yr * xs) / J

        return rx, ry, rz, sx, sy, sz, tx, ty, tz, J

    # Properties -------------------------------------------------------------------------------------------------------
    # @property
    # def EToV(self):
    #     """numpy.ndarray: An (self.N_tets x 4) array containing the 4 indices of the vertices of the self.N_tets
    #            tetrahedra that make up the mesh. It returns the value in self.tets, since it is the same data."""
    #     return self.tets
    # ------------------------------------------------------------------------------------------------------------------
