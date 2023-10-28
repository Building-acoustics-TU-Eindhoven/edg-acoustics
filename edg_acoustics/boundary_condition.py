"""
``edg_acoustics.boundary_condition``
======================

The edg_acoustics boundary_condition  provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup boundary condition for a specific scenario.

Please note that the most-used mesh functions and classes in edg_acoustics are present in
the main :mod:`edg_acoustics` namespace rather than in :mod:`edg_acoustics.boundary_condition`.  These are:
:class:`.BoundaryCondition`, and :func:`.some_function`.

Functions and classes present in :mod:`edg_acoustics.boundary_condition` are listed below.

Setup Boundary Condition
---------------
   BoundaryCondition
"""
from __future__ import annotations
import meshio
import numpy
import edg_acoustics

__all__ = ['BoundaryCondition','FREQ_MAX']

# Constants
FREQ_MAX = 2e3  # maximum resolvable frequency

class BoundaryCondition:
    """Setup boundary condition of a DG acoustics simulation for a specific scenario.

    :class:`.BoundaryCondition` is used to 
    load the boundary condition parameters, determine the time step size.

    Args:
        acoustics_simulation (edg_acoustics.AcousticsSimulation): the simulation object containing the spatial discretisation.
        BCpara (list [dict]): a list of boundary conditon parameters from the multi-pole model. Each element is a dictionary 
            with keys (values) ['label'(int),'RI'(float),'RP'(numpy.ndarray),'CP'(numpy.ndarray)]. 
            'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`. 
            'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row). 
                'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
                     :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
            BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
            i.e., all boundary conditions in the simulation must have an associated boundary condition parameters. 

        freq_max (float): maximum resolvable frequency of the simulation. <default>: edg_acoustics.FREQ_MAX

    Raises:
        ValueError: If BCpara.keys() is not present in the acoustics_simulation.BC_list.values(), an error is raised. 
            If a label is present in the acoustics_simulation.BC_list.values() but not in BCpara, an error is raised.

    Attributes:
        BC_triangles (dict[str, numpy.ndarray]): a dictionary containing the list of triangles that have a certain
            boundary condition. self.BC_triangles['BC_label'] is a numpy.array with the nodes of each triangle where
            boundary condition of type 'BC_label' is to be implemented. The nodes defining each triangle in the
            numpy.array are stored per row.
        EToE (numpy.ndarray): an (4, N_tets) array containing the information of which elements are neighbors of
            an element, i.e., EToE[j, i] returns the index of the jth neighbor of element i. The definition of jth
            neighbor follows the mesh generator's convention.



    Example:
        An element of this class can be initialized in the following way


    """

    def __init__(self, acoustics_simulation: edg_acoustics.AcousticsSimulation, BCpara: list[dict], freq_max: float=FREQ_MAX):
        # Check if BCpara is compatible with AcousticsSimulation.BCnode and satisfies physical admissibility condition.
        self.__check_BCpara(self, acoustics_simulation.BCnode, BCpara, freq_max)

        # Store input parameters
        self.BCpara = BCpara
        self.BCvar=self.__init_ADEvariables(self, self.BCpara, acoustics_simulation.BCnode)



    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def __check_BCpara(self, BCnode: list[dict], BCpara: list[dict], freq_max: float):
        """Check if BCpara is compatible with AcousticsSimulation.BCnode and satisfies physical admissibility condition.

        Given an acoustics simulation data structure with a set of boundary conditions specified in acoustics_simulation.BC_list, 
        check if the list of boundary conditions specification and parameters are compatible. 
        By compatible we mean that all boundary conditions (keys) in BCpara exist in acoustics_simulation.BC_list, and vice-versa.
        Also, to satisfy the causality and reality conditions, multi-pole model parameters :math:`\\zeta_i` (stored in first row of 
        numpy.array BCpara[BC_label] need to be positive.
        To satisfy the passivity condition, the magnitude of the reflection coefficient from the multi-pole model need to be smaller than 1,
        that is, :math:`|R(\\omega)|\\leq 1', where :math:`R(\\omega)=R_\\infty+\\sum_{k=1}^{S}\\frac{A_k}{\\zeta_k+\\mathrm{i}\\omega}'
        

        Args:
            BCnode (list[dict]): List of boundary map nodes, each element being a dictionary 
                with keys (values) ['label'(int),'map'(numpy.ndarray),'vmap'(numpy.ndarray)]. 
            BCpara (list [dict]): a list of boundary conditon parameters from the multi-pole model. Each element is a dictionary 
                with keys (values) ['label'(int),'RI'(float),'RP'(numpy.ndarray),'CP'(numpy.ndarray)]. 
                'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`. 
                'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row). 
                'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
                     :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
                More details about the multi-pole model parameters and boundary condition can be found in reference https://doi.org/10.1121/10.0001128.
                BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
                i.e., all boundary conditions in the simulation must have an associated boundary condition parameters. 
            freq_max (float): maximum resolvable frequency of the simulation. <default>: edg_acoustics.FREQ_MAX

        Returns:
            is_compatible (bool): a flag specifying if BCpara is compatible with the simulation object and satisfies the conditions or not.
        """
        omega=numpy.arange(1., freq_max)
        for index, paras in enumerate(BCpara):
            assert paras['label'] == BCnode[index]['label'], "[edg_acoustics.BoundaryCondition]" 
            "All BC types must be present in the BCnode"
            "and all labels in the BCnode must have boundary parameters input."
            assert numpy.abs(self.__compute_Re(omega, paras)).all() <= 1.0, "[edg_acoustics.BoundaryCondition] All reflection coefficient must be smaller than 1"
            print('boundary parameter with label: ' + str(paras['label']) + ' has passed the physical admissbility test')
     
        
    @staticmethod
    def __compute_Re(omega: numpy.ndarray, paras: dict):
        """Computes the reflection coefficient given the passed parameter of the multi-pole model at the frequencies of omega.

        Args:
            omega (numpy.ndarray): angular frequency.
            paras (dict): a dictionary of the multi-pole model parameters with keys (values) ['label'(int),'RI'(float),'RP'(numpy.ndarray),'CP'(numpy.ndarray)] 
                The limit value of the reflection coefficient as the frequency approaches infinity is stored in the first row and first column.

        Returns:
            Re (numpy.ndarray): reflection coefficient at the frequencies of omega.
        """
        Re=numpy.ones(omega.shape)

        for polekey in paras:
            if polekey=='RI':
                    Re=Re*paras['RI']
            elif polekey== 'RP':
                    Re=Re*paras['RI']
                    A=paras['RP'][0,:]
                    zeta=paras['RP'][1,:]
                    for j in range(len(A)):
                        Re=Re+A[j] / (1j * omega + zeta[j])
            elif polekey=='CP':
                    Re=Re*paras['RI']
                    B=paras['CP'][0,:]
                    C=paras['CP'][1,:]
                    alpha=paras['CP'][2,:]
                    beta=paras['CP'][3,:]
                    for j in range(len(B)):
                        Re = Re + 0.5 * ((B[j] + 1j * C[j]) / (alpha[j] + 1j * beta[j] + 1j * omega) +
                                    (B[j] - 1j * C[j]) / (alpha[j] - 1j * beta[j] + 1j * omega))
        return Re
    
    @staticmethod
    def __init_ADEvariables(self, BCpara: list[dict], BCnode: list[dict]):
        """Initiate ADE variables, normal velocity, characteristic waves (outgoing and incoming).

        Args:
            BCpara (list [dict]): a list of boundary conditon parameters from the multi-pole model. Each element is a dictionary 
                with keys (values) ['label'(int),'RI'(float),'RP'(numpy.ndarray),'CP'(numpy.ndarray)]. 
                'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\inf`. 
                'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row). 
                'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row),
                     :math:`\\alpha` (stored in 3rd row), :math:`\\beta`(stored in 4th row).
                BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
                i.e., all boundary conditions in the simulation must have an associated boundary condition parameters. 
            BCnode (list[dict]): List of boundary map nodes, each element being a dictionary 
                with keys (values) ['label'(int),'map'(numpy.ndarray),'vmap'(numpy.ndarray)]. 

        Returns:
            BCvar (list [dict]): a list of ADE variables. Each element corresponds to one type of BC, and is a dictionary 
                with potential keys ['label', 'vn', 'ou', 'in', 'phi', 'PHI', 'kexi1', 'kexi2', 'KEXI1', 'KEXI2']. 
                More details about the multi-pole model parameters and boundary condition can be found in reference https://doi.org/10.1121/10.0001128.
                All values are stored rowwise in the numpy.array BCpara[BC_label], with :math:`\\zeta_i` occupying the first row.
                is stored in the first row and last column.
                BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
                i.e., all boundary conditions in the simulation must have an associated boundary condition parameters. 
        """
        BCvar=[]
        for index, paras in enumerate(BCpara):
            #  BCvar.append({'label': paras['label'], \
            #                'vn': numpy.empty(BCnode[index]['map'].shape), \
            #                'ou': numpy.empty(BCnode[index]['map'].shape), \
            #                'in': numpy.empty(BCnode[index]['map'].shape), \
            #                })
             BCvar.append({'label': paras['label']})
             BCvar[index].update({key: numpy.empty(BCnode[index]['map'].shape) for key in ['vn', 'ou', 'in']})
             for polekey in paras:
                if polekey== 'RP':
                        BCvar[index].update({key: numpy.empty([paras['RP'].shape[1], BCnode[index]['map'].shape[0]]) for key in ['phi', 'PHI']})
                elif polekey=='CP':
                        BCvar[index].update({key: numpy.empty([paras['CP'].shape[1], BCnode[index]['map'].shape[0]]) for key in ['kexi1', 'kexi2', 'KEXI1', 'KEXI2']})

             
        # kexi for one pole are stored in row, so 3 poles means 3 rows, the number of columns are determined by the number of BCnode

        return BCvar
    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
