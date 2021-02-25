"""Module for calculating Hamiltonians for a multi_qubit system.

Can calculate Ising spin coefficients of these systems and find the fluxes for
a given schedule. Has exact methods that use Schrieffer-Wolff and can be used for
small circuits (scales exponentially with system size), but it also has
approximate methods that use pair-wise SW and can be used for larger circuits
(scale linearly with system size).
"""
import warnings
import copy

import numpy as np
from scipy.linalg import sqrtm, expm
import scipy.sparse.linalg as sprsalg
from scipy import sparse, interpolate
import lmfit

from utils import multi_krond, multi_krons, basis_vec, _b
from elements import CSFQ, Coupler
from utils import e_j, e_c, e_l

warnings.filterwarnings("ignore", message="splu requires CSC matrix format")
warnings.filterwarnings(
    "ignore",
    message="spsolve is more efficient when sparse b is " "in the CSC matrix format",
)


class AnnealingCircuit:
    """Construct a circuit consisting of multiple flux qubits and couplers.
    This circuit object can then be used to extract system properties such
    as Ising coefficients or customized fluxes.
    """

    def __init__(self, elements, mutual_mat, trunc_vec):
        """
        Arguments
        ---------
        elements : list
            an ordered list of circuit objects
        mutual_mat : ndarray
            the mutual matrix that describes how the circuit elements in
            "elements" are coupled. The matrix indices correspond to the
            objects in the "elements".
            If elements i and j are coupled to each other via mutual inductance
            of M, then we should have mutual_mat[i,j] = mutual_mat[j,i] = -M
            dim=(len(elements), len(elements))
        trunc_vec : ndarray
            low-energy truncation for circuit elements: how many low energy
            eigenvalues to keep for each object in "elements"
            values must be >= 2
            dim=len(elements)

        Attributes
        ----------
        elements : list
            an ordered list of circuit objects
        total_elements : int
            total number of elements in the circuit
        qubit_indices : list
            a list of qubit indices in the circuit
        coupler_indices: list
            a list of coupler indices in the circuit
        mutual_mat : ndarray
            the mutual matrix that describes how the circuit elements in
            "elements" are coupled. The matrix indices correspond to the
            objects in the "elements"
            dim=(len(elements), len(elements))
        trunc_vec : ndarray
            low-energy truncation for circuit elements: how many low energy
            eigenvalues to keep for each object in "elements"
            dim=len(elements)
        d_list : ndarray
            list of circuit element asymmetries.
            dim=total_elements
        nmax : int
            size of the total system after low-e tuncation
        L_b : ndarray
            branch inductance matrix for interacting branches
            dim=(total_elements, total_elements)
        L_b_inv : ndarray
            inverse of the branch inductance matrix for interacting branches
            dim = (total_elements, total_elements)
        low_e_dict : dictionary
            a dictionary containing low-energy eigensystem of each loaded
            circuit element.
        low_pauli_dict : dictionary
            a dictionary contatining Pauli matrices of qubits represented
            in their low-energy subspace
        low_p0_dict : dictionary
            a dictionary containing ground state projector of the couplers
            represented in their low-energy subspace.
        hq : ndarray
            effective qubit-space Hamiltonian of circuit calculated
            via Schrieffer-Wolff transformation. Has to be calculated via its
            method.
            dim=(total_elements, total_elements)
        ising_sw_dict : dictionary
            dictionary containing ising coefficients calculated using
            Schrieffer-Wolff (SW) method. Has to be calculated via its method.
        ising_bo_dict : dictionary
            dictionary containing ising coefficients calculated using
            Born-Oppenheimer method. Has to be calculated via its method.
        ising_pwsw_dict : dictionary
            dictionary containing ising coefficients calculated using
            pair-wise Schrieffer-Wolff method. Has to be calculated via
            its method.
        custom_flux_num : dictionary
            dictionary containing custom circuit fluxes that produces a given
            ising schedule, calculated using numerical optimization of fluxes
            with full SW method. Has to be calculated via its method.
        custom_flux_bo : dictionary
            dictionary containing custom circuit fluxes that produces a given
            ising schedule, calculated using Born-Oppenheimer method.
            Has to be calculated via its method.
        custom_flux_pwsw : dictionary
            dictionary containing custom circuit fluxes that produces a given
            ising schedule, calculated using pair-wise Schrieffer-Wolff
             method. Has to be calculated via its method.
        phi_x_vec : ndarray
            array holding a given set of x-bias for all circuit elements.
            dim=total_elements
        phi_z_vec : ndarray
            array holding a given set of z-bias for all circuit elements.
            dim=total_elements
        """

        self.elements = [copy.copy(element) for element in elements]
        self.total_elements = len(self.elements)
        self.qubit_indices = [
            i for i, element in enumerate(self.elements) if element.type == "qubit"
        ]
        self.coupler_indices = [
            i for i, element in enumerate(self.elements) if element.type == "coupler"
        ]

        self.mutual_mat = np.array(mutual_mat)
        self.trunc_vec = np.array(trunc_vec)

        self.d_list = np.array([element.d for element in self.elements])

        self.nmax = np.prod(self.trunc_vec)

        # branch inductance matrix for interacting branches
        self.L_b = np.diag([element.l for element in self.elements]) + self.mutual_mat
        self.L_b_inv = np.linalg.inv(self.L_b)

        # load the inductances
        for i, element in enumerate(self.elements):
            element.l = 1 / (self.L_b_inv[i, i])

        # re-initialize objects after loading their inductors
        for element in self.elements:
            element.re_init()

        # define class attributes that can be calculated later
        self.low_e_dict = {}
        self.low_pauli_dict = {}
        self.low_p0_dict = {}
        self.hq = 0
        self.ising_sw_dict = {}
        self.ising_bo_dict = {}
        self.ising_pwsw_dict = {}
        self.custom_flux_num = {}
        self.custom_flux_bo = {}
        self.custom_flux_pwsw = {}
        self.phi_x_vec = 0
        self.phi_z_vec = 0

        return

    def calculate_quantum(self, phi_x_vec, phi_z_vec, sw=True, sparse_htot=True):
        """Calculates quantum properties of the whole circuit for given fluxes.
        Specifically calculates the low-energy eigensystem of individual
        elements, and if asked also calculated the effective qubit-subspace H
        using SW transformation.
        The results are saved in the class attributes.
        For large circuits this can not be done due to memory limits.

        Arguments
        ---------
        phi_x_vec : ndarray
            array holding x-bias for all circuit elements.
            dim=total_elements
        phi_z_vec : ndarray
            array holding a given set of z-bias for all circuit elements.
            dim=total_elements
        sw : bool
            whether to calculate the effective H using SW transformation or not
            default : True
        sparse_htot : bool
            Whether to calculate eigenvectors of h_tot using sparse matrix
            or dense. For some geometries sparse fails, example is
            FM triangle with the same parameters (degenerate)
            default : True
        """
        # set attributes to they can be used in other methods for calculations
        self.phi_x_vec = np.array(phi_x_vec)
        self.phi_z_vec = np.array(phi_z_vec)

        self.calculate_low_e()

        if sw:
            self.calculate_pauli_low_e()
            self.hq = self.get_hq(sparse_htot=sparse_htot)

        return

    def calculate_low_e(self):
        """Calculates the low-energy eigensystem of individual circuit
        elements. The result is saved in "low_e_dict" attribute.
        """
        # low energy eigensystem of elements
        for i, element in enumerate(self.elements):
            self.low_e_dict["e_low_" + str(i)], self.low_e_dict[
                "v_low_" + str(i)
            ] = element.get_low_e(
                self.phi_x_vec[i], self.phi_z_vec[i], self.trunc_vec[i]
            )
        return

    def calculate_pauli_low_e(self):
        """Calculates the pauli matrices of qubits and ground state projectors
        of couplers, all projected onto their low-energy eigensystem.
        The result is saved in "low_pauli_dict" and "low_p0_dict" attributes.
        """
        # single qubit Pauli matrices represented in low energy subspace
        for i in self.qubit_indices:
            qubit = self.elements[i]
            self.low_pauli_dict["pauli_" + str(i)] = qubit.get_pauli_low_e(
                self.phi_x_vec[i],
                self.phi_z_vec[i],
                self.low_e_dict["e_low_" + str(i)],
                self.low_e_dict["v_low_" + str(i)],
            )

        for i in self.coupler_indices:
            coupler = self.elements[i]
            self.low_p0_dict["p0_" + str(i)] = coupler.get_p0_low_e(
                self.low_e_dict["v_low_" + str(i)]
            )
        return

    def get_hams(self, basis=None):
        """Calculates Hamiltonian of the combined system

        Arguments
        ---------
        basis : dictionary
            a dictionary of low-energy eigensystem of circuit elements that is
            used as a fixed basis for constructing the total circuit H.
            if "None" then the instantaneous low-energy eigensystem of elements
            is used for H construction.
            default: None
        Returns
        -------
        h_0 : sparse matrix
            Hamiltonian of the non-interacting qubits written in the chosen basis.
            dim=(nmax, nmax)
        h_tot : sparse matrix
            Total Hamiltonian of the system (includes interactions) written in
            the chosen basis
            dim=(nmax, nmax)
        """
        if basis is None:
            ham_basis = self.low_e_dict
        else:
            ham_basis = basis
        # Hamiltonian of individual (non-interacting) elements
        h_list = [0 for i in range(self.total_elements)]
        h_0 = 0
        for i, element in enumerate(self.elements):
            if basis is None:
                h_list[i] = sparse.diags(ham_basis["e_low_" + str(i)])
            else:
                h_0i = element.get_h(self.phi_x_vec[i], self.phi_z_vec[i])
                h_list[i] = (
                    ham_basis["v_low_" + str(i)].T.conj()
                    @ h_0i
                    @ ham_basis["v_low_" + str(i)]
                )
                h_list[i] = sparse.csr_matrix(h_list[i])

            for j in np.delete(np.arange(self.total_elements), i):
                h_list[j] = sparse.identity(self.trunc_vec[j])
            h_0 = h_0 + multi_krons(h_list)

        # assure Hermitianity
        h_0 = (h_0 + h_0.T.conj()) / 2

        # phi_0 of circuit elements in the low-energy subspace
        # (rotate into low_E subspace)
        phi_0_low_dict = {}
        for i, element in enumerate(self.elements):
            phi_0_low_dict[str(i)] = (
                ham_basis["v_low_" + str(i)].T.conj()
                @ element.phi_0
                @ ham_basis["v_low_" + str(i)]
            )
            phi_0_low_dict[str(i)] = sparse.csr_matrix(phi_0_low_dict[str(i)])

        # construct the interaction matrix
        h_int = 0
        for i in range(self.total_elements):
            phi_list = [0 for i in range(self.total_elements)]
            for j in range(i + 1, self.total_elements):
                phi_list[i] = phi_0_low_dict[str(i)]
                phi_list[j] = phi_0_low_dict[str(j)]
                for k in np.delete(np.arange(self.total_elements), [i, j]):
                    phi_list[k] = sparse.identity(self.trunc_vec[k])

                # prevent division by zero
                if np.abs(self.L_b_inv[i, j]) >= 1e-16:
                    h_int = h_int + 2 * e_l(1 / self.L_b_inv[i, j]) * multi_krons(
                        phi_list
                    )
                else:
                    pass

        h_tot = h_0 + h_int

        return [h_0, h_tot]

    def calculate_spectrum(self, phi_dict, levels=4, verbose=False):
        """Calculates the spectrum of the circuit along an anneal

        Arguments
        ---------
        phi_dict : dictionary
            a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
            i is the index of the circuit element. for each key there's an array
            of circuit fluxes during the anneal.
        levels : int
            number of low-energy eigenvalues to calculate
            default: 4
        verbose : bool
            whether to show the progress or not.
            default : False

        Returns
        -------
        e_list : ndarray
            array of circuit eigenenergies along the flux schedule.
            dim=(phi_dict["points"], levels)
        """
        pts = phi_dict["points"]
        phi_x_all = np.array(
            [phi_dict["phix_" + str(i)] for i in range(self.total_elements)]
        )
        phi_z_all = np.array(
            [phi_dict["phiz_" + str(i)] for i in range(self.total_elements)]
        )
        e_list = np.zeros((pts, levels))
        for p in range(pts):
            if verbose:
                print(
                    "Calculating spectrum for bias point",
                    p + 1,
                    "/",
                    pts,
                    end="\x1b[1K\r",
                )

            self.calculate_quantum(phi_x_all[:, p], phi_z_all[:, p], sw=False)
            h_tot = self.get_hams()[1]
            e, _ = sprsalg.eigsh(
                h_tot, k=levels, which="SA", v0=basis_vec(0, self.nmax)
            )
            sort_index = np.argsort(e)
            e = e[sort_index]
            e_list[p] = e - e[0]

        return e_list

    def get_hq(self, sparse_htot=True):
        """Calculates the effective qubit-subspace Hamiltonian of the combined
        system. Uses Schrieffer-Wolff transformation described in arXiv:1912.00464

        Arguments
        ---------
        sparse_htot : bool
            Whether to calculate eigenvectors of h_tot using sparse matrix
            or dense. For some geometries sparse fails, example is
            FM triangle with the same parameters (degenerate)
            default is True
        Returns
        -------
        h_q : ndarray
            Effective Ising Hamiltonian for the joint system
            dim=(nmax, namx)
        """
        h_0, h_tot = self.get_hams()

        dim = 2 ** len(self.qubit_indices)
        e_0, self.v_0 = sprsalg.eigsh(
            h_0, k=dim, which="SA", v0=basis_vec(0, self.nmax)
        )
        sort_index = np.argsort(e_0)
        self.v_0 = self.v_0[:, sort_index]

        if sparse_htot:
            e_tot, v_tot = sprsalg.eigsh(
                h_tot, k=dim, which="SA", v0=basis_vec(0, self.nmax)
            )
            sort_index = np.argsort(e_tot)
            v_tot = v_tot[:, sort_index]
        else:
            _, v_tot = np.linalg.eigh(h_tot.toarray())
            v_tot = v_tot[:, 0:dim]

        p_0 = self.v_0 @ self.v_0.T.conj()
        p_tot = v_tot @ v_tot.T.conj()
        self.u_sw = sqrtm(
            (2 * p_0 - np.eye(h_0.shape[0])) @ (2 * p_tot - np.eye(h_0.shape[0]))
        )

        h_q = p_0 @ self.u_sw @ h_tot @ self.u_sw.T.conj() @ p_0

        return h_q

    def calculate_ising_sw(self, index_list):
        """Calculates Ising spin coefficients for the joint system using full SW.
        Uses the effective Ising Hamiltonian of get_hq() method.

        Arguments
        ---------
        index_list : ndarray
            a list of indices indicating the Pauli index of each qubit.
            0: sigma_I, 1: sigma_x, 2: sigma_y, 3: sigma_z
            example: [0, 1, 3] means Ising coefficient for IXZ pauli matrix
            dim=len(qubit_indices)

        Returns
        -------
        ising : float
            Ising coefficient of the system for the given indices
        """
        pauli_list = [0 for i in range(self.total_elements)]
        for index, i in enumerate(self.qubit_indices):
            pauli_list[i] = self.low_pauli_dict["pauli_" + str(i)][index_list[index]]
        for i in self.coupler_indices:
            pauli_list[i] = self.low_p0_dict["p0_" + str(i)]

        pauli_operator = multi_krond(pauli_list)
        dim = 2 ** len(self.qubit_indices)
        ising = np.trace(self.hq @ pauli_operator) / dim

        return ising.real

    def get_ising_sw(self, phi_dict, verbose=False, sparse_htot=True):
        """Calculates the ising coefficients of the circuit along an anneal.
        Uses the full SW method.

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit element. for each key there's an array
             of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False
        sparse_htot : bool
            Whether to calculate eigenvectors of h_tot using sparse matrix
            or dense. For some geometries sparse fails, example is
            FM triangle with the same parameters (degenerate)
            default is True

         Returns
         -------
         ising_sw_dict : dictionary
             a dictionary of ising coefficients. Keys are "x_i", "z_i", and
             "zz_i,j", where i<j are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        pts = phi_dict["points"]
        phi_x_all = np.array(
            [phi_dict["phix_" + str(i)] for i in range(self.total_elements)]
        )
        phi_z_all = np.array(
            [phi_dict["phiz_" + str(i)] for i in range(self.total_elements)]
        )
        # create the empty dictionary
        self.ising_sw_dict["points"] = pts
        for i in range(len(self.qubit_indices)):
            self.ising_sw_dict["x_" + str(i)] = np.zeros(pts)
            self.ising_sw_dict["z_" + str(i)] = np.zeros(pts)
        for i, coupler_index in enumerate(self.coupler_indices):
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]
            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]
            self.ising_sw_dict["zz_" + str(index_0) + "," +
                               str(index_1)] = np.zeros(pts)

        # calculate the full SW
        for p in range(pts):
            if verbose:
                print(
                    "Calculating full SW for schedule point",
                    p + 1,
                    "/",
                    pts,
                    end="\x1b[1K\r",
                )

            self.calculate_quantum(
                phi_x_all[:, p], phi_z_all[:, p], sw=True, sparse_htot=sparse_htot
            )
            # save single qubit terms
            for i in range(len(self.qubit_indices)):
                ising_index = basis_vec(i, len(self.qubit_indices)).astype(int)
                self.ising_sw_dict["x_" + str(i)][p] = self.calculate_ising_sw(
                    1 * ising_index
                )
                self.ising_sw_dict["z_" + str(i)][p] = self.calculate_ising_sw(
                    3 * ising_index
                )

            # save interaction terms
            for i, coupler_index in enumerate(self.coupler_indices):
                # index of circuit elements that are coupled via the coupler
                index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]
                # ID of qubits that are coupled to each other
                index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
                index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]

                ising_index = np.zeros(len(self.qubit_indices), dtype=int)
                ising_index[index_0], ising_index[index_1] = 1, 1
                self.ising_sw_dict["zz_" + str(index_0) + ',' + str(index_1)][p] = \
                    self.calculate_ising_sw(3 * ising_index)

        return copy.deepcopy(self.ising_sw_dict)

    def _get_single_qubit_ising(self, phi_dict, verbose=False):
        """Calculates the ising coefficients for individual isolated qubits.
        Qubits are loaded but NOT interacting with others.

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False

         Returns
         -------
         ising_dict : dictionary
             a dictionary of ising coefficients. Keys are "x_i", "z_i",
              where i are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        pts = phi_dict["points"]
        ising_dict = {}
        for i, qubit_index in enumerate(self.qubit_indices):
            if verbose:
                print(
                    "calculating qubit isings for qubit",
                    i + 1,
                    "of",
                    len(self.qubit_indices),
                )

            x_list = np.zeros(pts)
            z_list = np.zeros(pts)
            qubit = self.elements[qubit_index]

            for p in range(pts):
                x_list[p], z_list[p] = qubit.get_ising(
                    phi_dict["phix_" + str(qubit_index)][p],
                    phi_dict["phiz_" + str(qubit_index)][p],
                )[0]

            ising_dict["x_" + str(i)] = x_list
            ising_dict["z_" + str(i)] = z_list
        return ising_dict

    @staticmethod
    def _get_bo_coupling(
        qubit0,
        coupler,
        qubit1,
        phi_x_list,
        phi_z_list,
        pauli_0,
        pauli_1,
        alpha0,
        alpha1,
    ):
        """Calculates the interaction strength between two qubits mediated
        via a tunable coupler using Born-Oppenheimer method of
        Kafri et al. Phys. Rev. A 95, 052333 (2017)

        Arguments
        ---------
        qubit0 : object
            An instance of the qubit object
        coupler : object
            An instance of the coupler object
        qubit0 : object
            An instance of the qubit object
        phi_x_list : ndarray
             array containing x-bias for circuit elements
             dim=3
        phi_z_list : ndarray
             array containing z-bias for circuit elements
             dim=3
        pauli_0 : int
            index of pauli for qubit0.
            0: sigma_I, 1: sigma_x, 2: sigma_y, 3: sigma_z
        pauli_1 : int
            index of pauli for qubit1.
            0: sigma_I, 1: sigma_x, 2: sigma_y, 3: sigma_z
        alpha0 : float
            mutual/qubit_inductance for qubit 0 and coupler
        alpha1 : float
            mutual/qubit_inductance for qubit 1 and coupler

        Returns
        -------
        out : float
            Interaction strength.
            Units of energy in GHz (i.e., omega = 2*pi*f)
         """
        out = 0
        zeta = coupler.zeta
        beta = coupler.beta_max * np.cos(phi_x_list[1] / 2)

        for nu in range(-50, 50):
            s0 = nu * alpha0
            s1 = nu * alpha1
            out = out + (
                _b(nu, beta, zeta)
                * np.exp(1j * nu * (phi_z_list[1] + np.pi))
                * qubit0._c(pauli_0, s0, phi_x_list[0], phi_z_list[0])
                * qubit1._c(pauli_1, s1, phi_x_list[2], phi_z_list[2])
            )

        out = out * e_l(coupler.l) * 2
        return out

    def _get_coupler_zz_bo(self, phi_dict, verbose=False):
        """Calculates the ZZ interaction Ising coefficients between qubits using
        Born-Oppenheimer method.

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False

         Returns
         -------
         ising_bo_dict : dictionary
             a dictionary of ising coefficients. Keys are "zz_i,j",
              where i<j are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        pts = phi_dict["points"]
        for i, coupler_index in enumerate(self.coupler_indices):
            zz_list = np.zeros(pts)
            coupler = self.elements[coupler_index]
            if verbose:
                print(
                    "calculating coupling strength for coupler",
                    i + 1,
                    "of",
                    len(self.coupler_indices),
                )

            # index of circuit elements that are coupled via the coupler
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]
            # Below we assumes each coupler is connected to only 2 qubits
            # CAUTION: Kafri uses unloaded qubits here (for single qubit Ising)
            # but I'm using loaded for now
            qubit0 = self.elements[index_of_coupled[0]]
            qubit1 = self.elements[index_of_coupled[1]]

            mutual_list = -self.mutual_mat[coupler_index][index_of_coupled]
            # CAUTION: Kafri uses unloaded qubit l here
            # but I'm using loaded for now
            alpha0 = mutual_list[0] / qubit0.l
            alpha1 = mutual_list[1] / qubit1.l

            for p in range(pts):
                if verbose:
                    print("schedule point", p + 1, "/", pts, end="\x1b[1K\r")

                phi_x_list = [
                    phi_dict["phix_" + str(index_of_coupled[0])][p],
                    phi_dict["phix_" + str(coupler_index)][p],
                    phi_dict["phix_" + str(index_of_coupled[1])][p],
                ]
                phi_z_list = [
                    phi_dict["phiz_" + str(index_of_coupled[0])][p],
                    phi_dict["phiz_" + str(coupler_index)][p],
                    phi_dict["phiz_" + str(index_of_coupled[1])][p],
                ]

                zz_list[p] = self._get_bo_coupling(
                    qubit0,
                    coupler,
                    qubit1,
                    phi_x_list,
                    phi_z_list,
                    3,
                    3,
                    alpha0,
                    alpha1,
                ).real

            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]
            self.ising_bo_dict["zz_" + str(index_0) + ',' +
                               str(index_1)] = zz_list

        return self.ising_bo_dict

    def get_ising_bo(self, phi_dict, verbose=False):
        """Calculates all the Ising coefficients of the system using
        Born-Oppenheimer method of Kafri et al. Phys. Rev. A 95, 052333 (2017)

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False

         Returns
         -------
         ising_bo_dict : dictionary
             a dictionary of ising coefficients. Keys are "x_i", "z_i",
             where i are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        self.ising_bo_dict = self._get_single_qubit_ising(phi_dict, verbose=verbose)
        _ = self._get_coupler_zz_bo(phi_dict, verbose=verbose)
        self.ising_bo_dict["points"] = phi_dict["points"]
        return copy.deepcopy(self.ising_bo_dict)

    def _get_pwsw_coupling(
        self,
        qubit0_index,
        coupler_index,
        qubit1_index,
        phi_x_vec,
        phi_z_vec,
        pauli_0,
        pauli_1,
    ):
        """Calculates the interaction strength between two qubits mediated
        via a tunable coupler using pair-wise Schrieffer-Wolff.
        It essentially construct a qubit-coupler-qubit joint system and
        calculates the interaction strength for it.

        Arguments
        ---------
        qubit0_index : int
            index of qubit0 in the circuit elements
        coupler_index : int
            index of coupler in the circuit elements
        qubit1_index : int
            index of qubit1 in the circuit elements
        phi_x_vec : ndarray
             array containing x-bias for circuit elements
             dim=3
        phi_z_vec : ndarray
             array containing z-bias for circuit elements
             dim=3
        pauli_0 : int
            index of pauli for qubit0.
            0: sigma_I, 1: sigma_x, 2: sigma_y, 3: sigma_z
        pauli_1 : int
            index of pauli for qubit1.
            0: sigma_I, 1: sigma_x, 2: sigma_y, 3: sigma_z

        Returns
        -------
        ising : float
            Interaction strength.
            Units of energy in GHz (i.e., omega = 2*pi*f)
         """
        indices = [qubit0_index, coupler_index, qubit1_index]
        # these elements will e loaded
        elements = [
            self.elements[qubit0_index],
            self.elements[coupler_index],
            self.elements[qubit1_index],
        ]
        trunc_vec = self.trunc_vec[[qubit0_index, coupler_index, qubit1_index]]

        # low energy eigensystem of elements
        low_e_dict = {}
        for i, element in enumerate(elements):
            low_e_dict["e_low_" + str(i)], low_e_dict[
                "v_low_" + str(i)
            ] = element.get_low_e(phi_x_vec[i], phi_z_vec[i], trunc_vec[i])

        # single qubit Pauli matrices represented in low energy subspace
        low_pauli_dict = {}
        for i in [0, 2]:
            qubit = elements[i]
            low_pauli_dict["pauli_" + str(i)] = qubit.get_pauli_low_e(
                phi_x_vec[i],
                phi_z_vec[i],
                low_e_dict["e_low_" + str(i)],
                low_e_dict["v_low_" + str(i)],
            )

        # projector onto ground state for the coupler
        low_p0_dict = {}
        for i in [1]:
            coupler = elements[i]
            low_p0_dict["p0_" + str(i)] = coupler.get_p0_low_e(
                low_e_dict["v_low_" + str(i)]
            )

        # diagonal Hamiltonian of individual elements
        h_list = [0 for i in range(3)]
        h_0 = 0
        for i in range(3):
            h_list[i] = sparse.diags(low_e_dict["e_low_" + str(i)])
            for j in np.delete(np.arange(3), i):
                h_list[j] = sparse.identity(trunc_vec[j])
            h_0 = h_0 + multi_krons(h_list)

        # phi_0 of circuit elements in the low-energy subspace
        # (rotate into low_E subspace)
        phi_0_low_dict = {}
        for i, element in enumerate(elements):
            phi_0_low_dict[str(i)] = (
                low_e_dict["v_low_" + str(i)].T.conj()
                @ element.phi_0
                @ low_e_dict["v_low_" + str(i)]
            )
            phi_0_low_dict[str(i)] = sparse.csr_matrix(phi_0_low_dict[str(i)])

        # construct the interaction matrix
        h_int = 0
        for i in range(3):
            phi_list = [0 for i in range(3)]
            for j in range(i + 1, 3):
                phi_list[i] = phi_0_low_dict[str(i)]
                phi_list[j] = phi_0_low_dict[str(j)]
                for k in np.delete(np.arange(3), [i, j]):
                    phi_list[k] = sparse.identity(trunc_vec[k])

                # prevent division by zero
                if np.abs(self.L_b_inv[indices[i], indices[j]]) >= 1e-16:
                    h_int = h_int + 2 * e_l(
                        1 / self.L_b_inv[indices[i], indices[j]]
                    ) * multi_krons(phi_list)
                else:
                    pass

        h_tot = h_0 + h_int

        # calculate effective qubit hamiltonian (h_q) using SW
        e_0, v_0 = sprsalg.eigsh(
            h_0, k=4, which="SA", v0=basis_vec(0, np.prod(trunc_vec))
        )
        sort_index = np.argsort(e_0)
        v_0 = v_0[:, sort_index]
        e_tot, v_tot = sprsalg.eigsh(
            h_tot, k=4, which="SA", v0=basis_vec(0, np.prod(trunc_vec))
        )
        sort_index = np.argsort(e_tot)
        v_tot = v_tot[:, sort_index]

        p_0 = v_0 @ v_0.T.conj()
        p_tot = v_tot @ v_tot.T.conj()
        u = sqrtm((2 * p_0 - np.eye(h_0.shape[0])) @ (2 * p_tot - np.eye(h_0.shape[0])))

        h_q = p_0 @ u @ h_tot @ u.T.conj() @ p_0

        # calculate ising coefficient
        index_list = [pauli_0, pauli_1]
        pauli_list = [0 for i in range(3)]
        for index, i in enumerate([0, 2]):
            pauli_list[i] = low_pauli_dict["pauli_" + str(i)][index_list[index]]
        for i in [1]:
            pauli_list[i] = low_p0_dict["p0_" + str(i)]

        pauli_operator = multi_krond(pauli_list)
        ising = np.trace(h_q @ pauli_operator) / 4

        return ising.real

    def _get_coupler_zz_pwsw(self, phi_dict, verbose=False):
        """Calculates the ZZ interaction Ising coefficients between qubits using
        pair-wise Schrieffer-Wolff method.

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False

         Returns
         -------
         ising_pwsw_dict : dictionary
             a dictionary of ising coefficients. Keys are "zz_i,j",
              where i<j are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        pts = phi_dict["points"]
        for i, coupler_index in enumerate(self.coupler_indices):
            zz_list = np.zeros(pts)
            if verbose:
                print(
                    "calculating coupling strength for coupler",
                    i + 1,
                    "of",
                    len(self.coupler_indices),
                )

            # index of circuit elements that are coupled via the coupler
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]

            for p in range(pts):
                if verbose:
                    print("schedule point", p + 1, "/", pts, end="\x1b[1K\r")

                phi_x_list = [
                    phi_dict["phix_" + str(index_of_coupled[0])][p],
                    phi_dict["phix_" + str(coupler_index)][p],
                    phi_dict["phix_" + str(index_of_coupled[1])][p],
                ]
                phi_z_list = [
                    phi_dict["phiz_" + str(index_of_coupled[0])][p],
                    phi_dict["phiz_" + str(coupler_index)][p],
                    phi_dict["phiz_" + str(index_of_coupled[1])][p],
                ]

                zz_list[p] = self._get_pwsw_coupling(
                    index_of_coupled[0],
                    coupler_index,
                    index_of_coupled[1],
                    phi_x_list,
                    phi_z_list,
                    3,
                    3,
                )

            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]
            self.ising_pwsw_dict["zz_" + str(index_0) + ',' +
                                 str(index_1)] = zz_list

        return self.ising_pwsw_dict

    def get_ising_pwsw(self, phi_dict, verbose=False):
        """Calculates all the Ising coefficients of the system using
        pair-wise Schrieffer-Wolff method.

         Arguments
         ---------
         phi_dict : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         verbose : bool
             whether to show the progress or not.
             default : False

         Returns
         -------
         ising_pwsw_dict : dictionary
             a dictionary of ising coefficients. Keys are "x_i", "z_i",
             where i are indexes of qubits from 0 to len(qubit_indices)
             For each key there is an array of coefficients during the anneal.
         """
        self.ising_pwsw_dict = self._get_single_qubit_ising(phi_dict, verbose=verbose)
        _ = self._get_coupler_zz_pwsw(phi_dict, verbose=verbose)
        self.ising_pwsw_dict["points"] = phi_dict["points"]
        return copy.deepcopy(self.ising_pwsw_dict)

    def _residual_phi_x(self, params, phi_x_sym, d):
        """Residual function for finding asymmetric junction x-bias that yields
        the same persistent current as in the case of symmetric junctions.

        Arguments
        ---------
        params : object
            an instance of lmfit.Parameters(), which stores optimization
            parameters. It's a dictionary of lmfit.Parameter() objects.
            Optimization parameter is asymmetric x-bias
        phi_x_sym : float
            the x-bias for the symmetric junctions
        d : float
            junction asymmetry

        Returns
        -------
        float
            Difference between target and calculated persistent current prefactors
        """
        values = params.valuesdict()
        phi_x_asym = values["phi_x_asym"]
        residual = np.cos(phi_x_sym / 2) - np.cos(phi_x_asym / 2) * np.sqrt(
            1 + d ** 2 * np.tan(phi_x_asym / 2) ** 2
        )
        return np.abs(residual)

    def _get_phi_x_d(self, phi_x_sym, d):
        """ Calculates the x-bias for asymmetric junctions that yields the same
        persistent current as the case of symmetric junctions.

        Arguments
        ---------
        phi_x_sym : ndarray
            x-bias (phase NOT flux) of the symmetric junction
        d : float
            Asymmetry parameter

        Returns
        -------
        phi_x_asym : ndarray
            x-bias (phase NOT flux) for asymmetric junctions
        """
        p_tot = len(phi_x_sym)
        phi_x_asym = np.zeros(p_tot)
        phi_x_asym_init = phi_x_sym[0]
        params = lmfit.Parameters()
        params["phi_x_asym"] = lmfit.Parameter(
            name="phi_x_asym", value=phi_x_asym_init, min=0, max=2 * np.pi
        )
        for i, phi_x in enumerate(phi_x_sym):
            minner = lmfit.Minimizer(self._residual_phi_x, params, fcn_args=(phi_x, d))
            result = minner.minimize(
                method="nelder", options={"fatol": 1e-10, "xatol": 1e-10}
            )
            phi_x_asym[i] = result.params["phi_x_asym"].value
        return phi_x_asym

    @staticmethod
    def _get_phi_z_d(phi_z_sym, phi_x_asym, d, asym_sign):
        """ Calculates the z-bias for asymmetric junctions given the symmetric
        results

        Arguments
        ---------
        phi_z_sym : ndarray
            z-bias (phase NOT flux) of the symmetric junction
        phi_x_asym : ndarray
            x-bias (phase NOT flux) of the asymmetric junction
        d : float
            Asymmetry parameter
        asym_sign : int
            The sign of asymmetry, can be either +1 or -1

        Returns
        -------
        phi_z_asym : ndarray
            z-bias (phase NOT flux) for asymmetric junctions
        """
        # handle the condition when phi_x_asym is pi
        # Forcing tan(pi/2) to be always positive
        where_pi = np.where(np.abs(phi_x_asym - np.pi) < 1e-6)[0]
        phi_x_asym[where_pi] = np.pi + 1e-8

        phi_z_asym = phi_z_sym + asym_sign * np.arctan(d * np.tan(phi_x_asym / 2))
        return phi_z_asym

    def _apply_asymmetry_shifts(self, phi_dict_sym):
        """ Calculates the circuit biases for asymmetric junctions
        given the biases for symmetric junctions.
        This essentially applies asymmetry shifts to the biases

        Arguments
        ---------
        phi_dict_sym : dictionary
            dictionary of circuit biases for symmetric x-loop junctions
            Keys are "phix_i" and "phiz_i" where i is the index of the circuit
            elements.

        Returns
        -------
        phi_dict_asym : dictionary
            dictionary of circuit biases for asymmetric x-loop junctions
            Keys are "phix_i" and "phiz_i" where i is the index of the circuit
            elements.
        """
        phi_dict_asym = {"points": phi_dict_sym["points"]}
        for index, element in enumerate(self.elements):
            if self.d_list[index] != 0:
                phi_dict_asym["phix_" + str(index)] = self._get_phi_x_d(
                    phi_dict_sym["phix_" + str(index)], self.d_list[index]
                )
                phi_dict_asym["phiz_" + str(index)] = self._get_phi_z_d(
                    phi_dict_sym["phiz_" + str(index)],
                    phi_dict_asym["phix_" + str(index)],
                    self.d_list[index],
                    element.asym_sign,
                )
            else:
                phi_dict_asym["phix_" + str(index)] = phi_dict_sym["phix_" + str(index)]
                phi_dict_asym["phiz_" + str(index)] = phi_dict_sym["phiz_" + str(index)]
        return phi_dict_asym

    def _residual_custom_flux_num(self, params, ising_target, null):
        """ Calculates the residual between calculated Isings and desired ones.
        Used for numerically finding circuit fluxes for custom schedules.

        Arguments
        ---------
        params : object
            an instance of lmfit.Parameters(), which stores optimization
            parameters. It's a dictionary of lmfit.Parameter() objects.
            Optimization parameters are circuit biases
        ising_target : dictionary
            a dictionary containing desired Ising coefficients
        null : null
            dummy variable to get argument passing working.

        Returns
        -------
        residual : ndarray
            Difference between target and desired Ising coefficients.
        """
        values = params.valuesdict()
        phi_x_all = np.array(
            [values["phix_" + str(i)] for i in range(self.total_elements)]
        )
        phi_z_all = np.array(
            [values["phiz_" + str(i)] for i in range(self.total_elements)]
        )

        self.calculate_quantum(phi_x_all, phi_z_all, sw=True)
        x_all, z_all = np.zeros((2, len(self.qubit_indices)))
        xx_all, yy_all, zz_all = np.zeros((3, len(self.coupler_indices)))
        # calculate the single qubit Isings for the trial biases
        for i in range(len(self.qubit_indices)):
            ising_index = basis_vec(i, len(self.qubit_indices)).astype(int)
            x_all[i] = self.calculate_ising_sw(1 * ising_index)
            z_all[i] = self.calculate_ising_sw(3 * ising_index)

        # calculate the two-qubit Isings for the trial biases
        for i, coupler_index in enumerate(self.coupler_indices):
            # index of circuit elements that are coupled via the coupler
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]

            # ID of qubits that are coupled to each other
            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]

            ising_index = np.zeros(len(self.qubit_indices), dtype=int)
            ising_index[index_0], ising_index[index_1] = 1, 1
            xx_all[i] = self.calculate_ising_sw(1 * ising_index)
            yy_all[i] = self.calculate_ising_sw(2 * ising_index)
            zz_all[i] = self.calculate_ising_sw(3 * ising_index)

        # always try to make XX and YY coupling as small as possible
        # albeit with a smaller weight in the cost function.

        # lmfit library uses the sum of squares of the array elements as
        # its cost function
        residual = []
        # 10 MHz error in X, helps with finding bias for small X (10x speedup)
        # X field decreases exponentially with barrier bias and becomes hard to
        # find the bias value for small X-fields without this error
        x_err = 10 / 1000 * 2 * np.pi
        # no error in Z for now, but include for completeness.
        z_err = 1
        residual = np.append(residual, (x_all - ising_target["x_all"]) / x_err)
        residual = np.append(residual, (z_all - ising_target["z_all"]) / z_err)
        residual = np.append(residual, zz_all - ising_target["zz_all"])
        residual = np.append(residual, xx_all * 0.1)
        residual = np.append(residual, yy_all * 0.1)

        return residual

    def get_custom_fluxes_num(
        self, schedule_dict, verbose=False, optimizer_method="leastsq"
    ):
        """Calculates custom circuit biases that yields a desired Ising schedule.
        Uses numerical optimization of full SW to find fluxes.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of ising coefficients. Keys are "x_i", "z_i", and
            "zz_i,j" where i<j are indexes of circuit elements.
            For each key there is an array of coefficients during the anneal.
        verbose : bool
            whether to show the progress or not.
            default : False
        optimizer_method : string
            Method used for optimization. Typical options are "leastsq" and
            "nelder". "leastsq" is much faster.
            default : "leastsq"

         Returns
         -------
         custom_flux_num : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         """
        # set all asymmetries to zero for easier calculations
        for element in self.elements:
            element.d = 0

        # create an schedule dictionary with only the initial points
        initial_ising_dict = {}
        for key in schedule_dict.keys():
            if key != "points":
                initial_ising_dict[key] = np.array([schedule_dict[key][0]])
            else:
                initial_ising_dict[key] = 1

        # use pair-wise SW to find an approximate for initial flux values.
        # good guess of initial flux values helps with finding solutions
        initial_fluxes = self.get_custom_fluxes_pwsw(
            initial_ising_dict, optimizer_method=optimizer_method
        )
        # create the lmfit optimizer parameter dictionary
        params = lmfit.Parameters()
        for i in range(self.total_elements):
            if self.elements[i].type == "qubit":
                params["phix_" + str(i)] = lmfit.Parameter(
                    name="phix_" + str(i),
                    value=initial_fluxes["phix_" + str(i)][0],
                    min=np.pi,
                    max=2 * np.pi,
                )
                params["phiz_" + str(i)] = lmfit.Parameter(
                    name="phiz_" + str(i),
                    value=initial_fluxes["phiz_" + str(i)][0],
                    min=-0.01 * np.pi,
                    max=0.01 * np.pi,
                )
            elif self.elements[i].type == "coupler":
                params["phix_" + str(i)] = lmfit.Parameter(
                    name="phix_" + str(i),
                    value=initial_fluxes["phix_" + str(i)][0],
                    min=np.pi,
                    max=2 * np.pi,
                )
                params["phiz_" + str(i)] = lmfit.Parameter(
                    name="phiz_" + str(i),
                    value=initial_fluxes["phiz_" + str(i)][0],
                    min=-0.01 * np.pi,
                    max=0.01 * np.pi,
                    vary=False,
                )

        pts = schedule_dict["points"]
        x_target, z_target = np.zeros((2, len(self.qubit_indices)))
        zz_target = np.zeros(len(self.coupler_indices))
        ising_target = {}
        self.custom_flux_num["points"] = pts
        for i in range(self.total_elements):
            self.custom_flux_num["phix_" + str(i)] = np.zeros(pts)
            self.custom_flux_num["phiz_" + str(i)] = np.zeros(pts)

        for p in range(pts):
            if verbose:
                print(
                    "Calculating fluxes for schedule point",
                    p + 1,
                    "/",
                    pts,
                    end="\x1b[1K\r",
                )

            # prepare the dictionary of target isings for single qubit terms
            for j in range(len(self.qubit_indices)):
                x_target[j] = schedule_dict["x_" + str(j)][p]
                z_target[j] = schedule_dict["z_" + str(j)][p]
            ising_target["x_all"], ising_target["z_all"] = x_target, z_target

            # prepare the dictionary of target isings for two-qubit terms
            for j, coupler_index in enumerate(self.coupler_indices):
                # index of circuit elements that are coupled via the coupler
                index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]

                # ID of qubits that are coupled to each other
                index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
                index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]
                zz_target[j] = schedule_dict["zz_" + str(index_0) + ',' +
                                             str(index_1)][p]

            ising_target["zz_all"] = zz_target

            # use lmfit to optimize for fluxes
            minner = lmfit.Minimizer(
                self._residual_custom_flux_num, params, fcn_args=(ising_target, None)
            )
            result = minner.minimize(method=optimizer_method)

            # raise warning if the solver fails
            residual_norm = np.linalg.norm(result.residual)
            target_norm = []
            for key in ising_target.keys():
                target_norm = np.append(target_norm, ising_target[key])
            target_norm = np.linalg.norm(target_norm)
            rel_error = residual_norm / target_norm
            if rel_error > 1e-2:
                print(
                    "point #{0:d} residual norm: \n".format(p + 1),
                    residual_norm / 2 / np.pi,
                    " (GHz)",
                    "\n",
                )
                warnings.warn(
                    (
                        "For the point #{0:d}, solver found solutions that"
                        + " are not optimal. The relative error is"
                        + " {1:.2f} %"
                    ).format(p + 1, rel_error * 100)
                )
            # save the results in our own dictionary
            for j in range(self.total_elements):
                self.custom_flux_num["phix_" + str(j)][p] = result.params[
                    "phix_" + str(j)
                ].value
                self.custom_flux_num["phiz_" + str(j)][p] = result.params[
                    "phiz_" + str(j)
                ].value

            # set initial value of next search to the solution of this search
            # speeds up the solver by ~2x, but could risk getting the solver
            # stuck in the wrong solution
            for j in range(self.total_elements):
                params["phix_" + str(j)].set(
                    value=self.custom_flux_num["phix_" + str(j)][p]
                )
                params["phiz_" + str(j)].set(
                    value=self.custom_flux_num["phiz_" + str(j)][p]
                )

        # apply asymmetry shifts
        self.custom_flux_num = self._apply_asymmetry_shifts(self.custom_flux_num)
        # retrieve circuit element asymmetries
        for i, element in enumerate(self.elements):
            element.d = self.d_list[i]

        return copy.deepcopy(self.custom_flux_num)

    def _get_single_qubit_custom_bias(
        self, schedule_dict, verbose=False, optimizer_method="leastsq"
    ):
        """Calculates the circuit biases that yields a given single qubit Ising
        schedule for individual isolated qubits.
        Qubits are loaded but NOT interacting with others.
        Uses numerical optimization of biases for individual qubits.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of ising coefficients. Keys are "x_i", "z_i",
            where i are indexes of qubits from 0 to len(qubit_indices)
            For each key there is an array of coefficients during the anneal.
        verbose : bool
             whether to show the progress or not.
             default : False
        optimizer_method : string
            Method used for optimization. Typical options are "leastsq" and
            "nelder". "leastsq" is faster.
            default : "leastsq"

        Returns
        -------
        qubit_flux : dictionary
            a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
            i is the index of the circuit elements (here qubits). for each key
            there's an array of circuit fluxes during the anneal.
         """
        qubit_flux = {"points": schedule_dict["points"]}
        for i, qubit_index in enumerate(self.qubit_indices):
            qubit = self.elements[qubit_index]

            if verbose:
                print(
                    "calculating qubit biases for qubit",
                    i + 1,
                    "of",
                    len(self.qubit_indices),
                )

            phix, phiz = qubit.get_fluxes(
                schedule_dict["x_" + str(i)],
                schedule_dict["z_" + str(i)],
                optimizer_method=optimizer_method,
                verbose=verbose,
            )
            qubit_flux["phix_" + str(qubit_index)] = phix
            qubit_flux["phiz_" + str(qubit_index)] = phiz

        return qubit_flux

    def _get_coupler_custom_bias_bo(self, schedule_dict, verbose=False):
        """Calculates the coupler biases that yields a desired ZZ interaction
        between qubits. qubit circuit biases are from "_get_single_qubit_custom_bias"
        Uses the Born-Oppenheimer method to calculate interactions.
        Result is saved in the "custom_flux_bo" attribute.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of Ising coefficients. Keys are "x_i", "z_i", "zz_i,j"
            where i<j are indexes of circuit elements.
            For each key there is an array of coefficients during the anneal.
        verbose : bool
             whether to show the progress or not.
             default : False
         """
        pts = schedule_dict["points"]
        phi_xc_list = np.linspace(0.5, 1, 10) * 2 * np.pi

        for i, coupler_index in enumerate(self.coupler_indices):
            coupler = self.elements[coupler_index]

            if verbose:
                print(
                    "\n calculating coupler bias for coupler",
                    i + 1,
                    "of",
                    len(self.coupler_indices),
                )

            self.custom_flux_bo["phix_" + str(coupler_index)] = np.zeros(pts)
            self.custom_flux_bo["phiz_" + str(coupler_index)] = np.zeros(pts)

            # index of circuit elements that are coupled via the coupler
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]

            # index of qubits that are coupled to each other
            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]

            # Below we assumes each coupler is connected to only 2 qubits
            # CAUTION: Kafri uses unloaded qubits here (for single qubit Ising)
            # but I'm using loaded for now
            qubit0 = self.elements[index_of_coupled[0]]
            qubit1 = self.elements[index_of_coupled[1]]

            mutual_list = -self.mutual_mat[coupler_index][index_of_coupled]
            # CAUTION: Kafri uses unloaded qubit l here
            # but I'm using loaded for now
            alpha0 = mutual_list[0] / qubit0.l
            alpha1 = mutual_list[1] / qubit1.l

            for p in range(pts):
                if verbose:
                    print("schedule point", p + 1, "/", pts, end="\x1b[1K\r")

                coupling_list = np.zeros(len(phi_xc_list))
                # for each point in the schedule, turn on the coupler all the
                # way and construct interpolation
                # skip the first point (x-bias at Phi_0/2) when coupler is off
                for j, phix_coupler in enumerate(phi_xc_list[1::]):
                    phi_x_list = [
                        self.custom_flux_bo["phix_" + str(index_of_coupled[0])][p],
                        phix_coupler,
                        self.custom_flux_bo["phix_" + str(index_of_coupled[1])][p],
                    ]
                    phi_z_list = [
                        self.custom_flux_bo["phiz_" + str(index_of_coupled[0])][p],
                        0,
                        self.custom_flux_bo["phiz_" + str(index_of_coupled[1])][p],
                    ]

                    coupling_list[j + 1] = self._get_bo_coupling(
                        qubit0,
                        coupler,
                        qubit1,
                        phi_x_list,
                        phi_z_list,
                        3,
                        3,
                        alpha0,
                        alpha1,
                    ).real

                    # prevent calculation of couplings if the coupler has
                    # already reached the desired strength. Speeds up
                    # calculations for typical anneals
                    if np.abs(coupling_list[j + 1]) >= np.abs(
                        schedule_dict["zz_" + str(index_0) + ',' +
                                      str(index_1)][p]
                    ):
                        cutoff_index = j + 2
                        break
                    cutoff_index = j + 2

                coupling_vs_bias = interpolate.interp1d(
                    coupling_list[0:cutoff_index],
                    phi_xc_list[0:cutoff_index],
                    kind=1,
                    fill_value="extrapolate",
                )

                phix = coupling_vs_bias(
                    schedule_dict["zz_" + str(index_0) + ',' +
                                  str(index_1)][p]
                )
                if phix > 2 * np.pi:
                    warnings.warn(
                        (
                            "Coupler {0:d} can not provide the required coupling"
                            + " strength of {1:.2f} GHz for point {2:d} in schedule."
                            + " Using maximum available strength instead."
                        ).format(
                            i + 1,
                            schedule_dict["zz_" + str(index_0) + ',' +
                                          str(index_1)][p]
                            / 2
                            / np.pi,
                            p,
                        )
                    )
                    phix = 2 * np.pi

                self.custom_flux_bo["phix_" + str(coupler_index)][p] = phix

        return None

    def get_custom_fluxes_bo(
        self, schedule_dict, verbose=False, optimizer_method="leastsq"
    ):
        """Calculates custom circuit biases that yields a desired Ising schedule.
        Uses isolated qubits for single qubit biases, and two-qubit
        Born-Oppenheimer for interactions.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of Ising coefficients. Keys are "x_i", "z_i",
            and "zz_i,j" where i<j are indexes of circuit elements.
            For each key there is an array of coefficients during the anneal.
        verbose : bool
            whether to show the progress or not.
            default : False
        optimizer_method : string
            Method used for optimization for single qubits.
            Typical options are "leastsq" and "nelder". "leastsq" is much faster.
            default : "leastsq"

         Returns
         -------
         custom_flux_bo : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         """
        self.custom_flux_bo["points"] = schedule_dict["points"]
        # set all asymmetries to zero for easier calculations
        for element in self.elements:
            element.d = 0

        self.custom_flux_bo = self._get_single_qubit_custom_bias(
            schedule_dict, verbose=verbose, optimizer_method=optimizer_method
        )
        self._get_coupler_custom_bias_bo(schedule_dict, verbose=verbose)

        # apply asymmetry shifts
        self.custom_flux_bo = self._apply_asymmetry_shifts(self.custom_flux_bo)
        # retrieve circuit element asymmetries
        for i, element in enumerate(self.elements):
            element.d = self.d_list[i]

        return copy.deepcopy(self.custom_flux_bo)

    def _get_coupler_custom_bias_pwsw(self, schedule_dict, verbose=False):
        """Calculates the coupler biases that yields a desired ZZ interaction
        between qubits. qubit circuit biases are from "_get_single_qubit_custom_bias"
        Uses the pair-wise SW method to calculate interactions.
        Result is saved in the "custom_flux_pwsw" attribute.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of Ising coefficients. Keys are "x_i", "z_i", "zz_i,j"
            where i<j are indexes of circuit elements.
            For each key there is an array of coefficients during the anneal.
        verbose : bool
             whether to show the progress or not.
             default : False
         """
        pts = schedule_dict["points"]
        phi_xc_list = np.linspace(0.5, 1, 10) * 2 * np.pi

        for i, coupler_index in enumerate(self.coupler_indices):

            if verbose:
                print(
                    "\n calculating coupler bias for coupler",
                    i + 1,
                    "of",
                    len(self.coupler_indices),
                )

            self.custom_flux_pwsw["phix_" + str(coupler_index)] = np.zeros(pts)
            self.custom_flux_pwsw["phiz_" + str(coupler_index)] = np.zeros(pts)

            # index of circuit elements that are coupled via the coupler
            index_of_coupled = np.nonzero(self.mutual_mat[coupler_index])[0]

            # index of qubits that are coupled to each other
            index_0 = np.argwhere(self.qubit_indices == index_of_coupled[0])[0, 0]
            index_1 = np.argwhere(self.qubit_indices == index_of_coupled[1])[0, 0]

            for p in range(pts):
                if verbose:
                    print("schedule point", p + 1, "/", pts, end="\x1b[1K\r")

                coupling_list = np.zeros(len(phi_xc_list))
                # for each point in the schedule, turn on the coupler all the
                # way and construct interpolation
                # skip the first point (x-bias at Phi_0/2) when coupler is off
                for j, phix_coupler in enumerate(phi_xc_list[1::]):
                    phi_x_list = [
                        self.custom_flux_pwsw["phix_" + str(index_of_coupled[0])][p],
                        phix_coupler,
                        self.custom_flux_pwsw["phix_" + str(index_of_coupled[1])][p],
                    ]
                    phi_z_list = [
                        self.custom_flux_pwsw["phiz_" + str(index_of_coupled[0])][p],
                        0,
                        self.custom_flux_pwsw["phiz_" + str(index_of_coupled[1])][p],
                    ]

                    coupling_list[j + 1] = self._get_pwsw_coupling(
                        index_of_coupled[0],
                        coupler_index,
                        index_of_coupled[1],
                        phi_x_list,
                        phi_z_list,
                        3,
                        3,
                    )

                    # prevent calculation of couplings if the coupler has
                    # already reached the desired strength. Speeds up
                    # calculations for typical anneals
                    if np.abs(coupling_list[j + 1]) >= np.abs(
                        schedule_dict["zz_" + str(index_0) + ',' +
                                      str(index_1)][p]
                    ):
                        cutoff_index = j + 2
                        break
                    cutoff_index = j + 2

                coupling_vs_bias = interpolate.interp1d(
                    coupling_list[0:cutoff_index],
                    phi_xc_list[0:cutoff_index],
                    kind=1,
                    fill_value="extrapolate",
                )

                phix = coupling_vs_bias(
                    schedule_dict["zz_" + str(index_0) + ',' +
                                  str(index_1)][p]
                )
                if phix > 2 * np.pi:
                    warnings.warn(
                        (
                            "Coupler {0:d} can not provide the required coupling"
                            + " strength of {1:.2f} GHz for point {2:d} in schedule."
                            + " Using maximum available strength instead."
                        ).format(
                            i + 1,
                            schedule_dict["zz_" + str(index_0) + ',' +
                                          str(index_1)][p]
                            / 2
                            / np.pi,
                            p,
                        )
                    )
                    phix = 2 * np.pi

                self.custom_flux_pwsw["phix_" + str(coupler_index)][p] = phix

        return None

    def get_custom_fluxes_pwsw(
        self, schedule_dict, verbose=False, optimizer_method="leastsq"
    ):
        """Calculates custom circuit biases that yields a desired Ising schedule.
        Uses isolated qubits for single qubit biases, and two-qubit
        pair-wise SW for interactions.

        Arguments
        ---------
        schedule_dict : dictionary
            a dictionary of Ising coefficients. Keys are "x_i", "z_i",
            and "zz_i,j" where i<j are indexes of circuit elements.
            For each key there is an array of coefficients during the anneal.
        verbose : bool
            whether to show the progress or not.
            default : False
        optimizer_method : string
            Method used for optimization for single qubits.
            Typical options are "leastsq" and "nelder". "leastsq" is much faster.
            default : "leastsq"

         Returns
         -------
         custom_flux_pwsw : dictionary
             a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
             i is the index of the circuit elements. for each key
             there's an array of circuit fluxes during the anneal.
         """
        self.custom_flux_pwsw["points"] = schedule_dict["points"]
        # set all asymmetries to zero for easier calculations
        for element in self.elements:
            element.d = 0

        self.custom_flux_pwsw = self._get_single_qubit_custom_bias(
            schedule_dict, verbose=verbose, optimizer_method=optimizer_method
        )
        self._get_coupler_custom_bias_pwsw(schedule_dict, verbose=verbose)

        # apply asymmetry shifts
        self.custom_flux_pwsw = self._apply_asymmetry_shifts(self.custom_flux_pwsw)
        # retrieve circuit element asymmetries
        for i, element in enumerate(self.elements):
            element.d = self.d_list[i]

        return copy.deepcopy(self.custom_flux_pwsw)

    def get_ips(self):
        """Calculates persistent currents of qubits for the truncated system.
        Eigenvalues will have units of nA.

        Arguments
        ---------

        Returns
        -------
        list of sparse
            [ip_0, ip_1, ...], which are persistent current operators of qubits
            0 through len(qubit_indices).
            Each PC operator has dim=(nmax, nmax)
        """
        self.calculate_low_e()
        # get qubit persistent current operators
        ip_dict = {}
        for i in self.qubit_indices:
            ip_dict[str(i)] = self.elements[i].get_ip(
                self.phi_x_vec[i], self.phi_z_vec[i]
            )

        # project Ips onto low-energy subspace of the system
        for i in self.qubit_indices:
            ip_dict[str(i)] = (
                self.low_e_dict["v_low_" + str(i)].T.conj()
                @ ip_dict[str(i)]
                @ self.low_e_dict["v_low_" + str(i)]
            )

        # tensor product with identity for other circuit elements
        prod_list = [0 for i in range(self.total_elements)]
        for i in self.qubit_indices:
            prod_list[i] = sparse.csr_matrix(ip_dict[str(i)])
            for j in np.delete(np.arange(self.total_elements), i):
                prod_list[j] = sparse.identity(self.trunc_vec[j])
            ip_dict[str(i)] = multi_krons(prod_list)

        return [ip_dict[str(i)] for i in self.qubit_indices]

    def get_povms(self, delta_i=10):
        """Calculates POVM operator for measuring probability of right
        circulating current, M_r.

        Arguments
        ---------
        delta_i : float
            measurement current sensitivity, in nA
            Default: 10 nA

        Returns
        -------
        list of sparse
            [povm_0, povm_1, ...], which are povms operators for right
            circulating current of qubits 0 through len(qubit_indices).
            Left circulating current is M_l = Identity - M_r.
            Each povm operator has dim=(nmax, nmax)
        """
        self.calculate_low_e()
        # get qubit POVM operators
        povm_dict = {}
        for i in self.qubit_indices:
            povm_dict[str(i)] = self.elements[i].get_povm(
                self.phi_x_vec[i], self.phi_z_vec[i], delta_i
            )

        # project onto low-energy subspace
        for i in self.qubit_indices:
            povm_dict[str(i)] = (
                self.low_e_dict["v_low_" + str(i)].T.conj()
                @ povm_dict[str(i)]
                @ self.low_e_dict["v_low_" + str(i)]
            )

        # tensor product with identity for other circuit elements
        prod_list = [0 for i in range(self.total_elements)]
        for i in self.qubit_indices:
            prod_list[i] = sparse.csr_matrix(povm_dict[str(i)])
            for j in np.delete(np.arange(self.total_elements), i):
                prod_list[j] = sparse.identity(self.trunc_vec[j])
            povm_dict[str(i)] = multi_krons(prod_list)

        return [povm_dict[str(i)] for i in self.qubit_indices]

    def _initialize_se(self, phi_dict):
        """Initialize some parameters for time-evolution simulation of circuits.
        Creates interpolating functions for circuit fluxes as class attributes.

        Arguments
        ---------
        phi_dict : dictionary
            a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
            i is the index of the circuit elements. for each key
            there's an array of circuit fluxes during the anneal.
        """
        phix_all = np.array(
            [phi_dict["phix_" + str(i)] for i in range(len(self.elements))]
        )
        phiz_all = np.array(
            [phi_dict["phiz_" + str(i)] for i in range(len(self.elements))]
        )

        s_ad = np.linspace(0, 1, phi_dict["points"])
        self.x_interp = interpolate.interp1d(
            s_ad, phix_all, axis=1, kind="linear", fill_value="extrapolate"
        )
        self.z_interp = interpolate.interp1d(
            s_ad, phiz_all, axis=1, kind="linear", fill_value="extrapolate"
        )
        return None

    def _match_phase(self, low_e_dict_1, low_e_dict_2):
        """Match the phase of eigenstates between two points.

        Arguments
        ---------
        low_e_dict_1 : dictionary
            a dictionary containing low-energy eigensystem of each loaded
            circuit element at point s1
        low_e_dict_2 : dictionary
            a dictionary containing low-energy eigensystem of each loaded
            circuit element at point s2

        Returns
        -------
        low_e_dict_2 : dictionary
            a dictionary containing low-energy eigensystem of each loaded
            circuit element at point s2, with all the phases matched to that of
            eigenstates at point s1.
        """
        # match phase of 2 with 1
        for i in range(self.total_elements):
            # inner dot of eigenvectors at different time-steps:
            inner_list = np.diag(
                low_e_dict_1["v_low_" + str(i)].T.conj()
                @ low_e_dict_2["v_low_" + str(i)]
            )

            # complex-plane phase of the inner products. Output in (-pi, pi]
            phase_list = np.angle(inner_list)
            for j, phase in enumerate(phase_list):
                low_e_dict_2["v_low_" + str(i)][:, j] = (
                    np.exp(-1j * phase) * low_e_dict_2["v_low_" + str(i)][:, j]
                )

        return low_e_dict_2

    def evolve_se_fixed(
        self, phi_dict, tf, init_state, dt=1e-2, save_at=None, basis_s=0
    ):
        """Calculates the close system time evolution of the joint system using
        Schrodinger's equation.
        Uses a basis for H construction and representation that is fixed
        throughout the anneal.

        Arguments
        ---------
        phi_dict : dictionary
            a dictionary of circuit fluxes. Keys are "phix_i" and "phiz_i" where
            i is the index of the circuit elements. for each key
            there's an array of circuit fluxes during the anneal.
        tf : float
            final anneal time in ns.
        init_state : ndarray
            initial state of the system.
            dim=(nmax, nmax)
        dt : float
            time step for discretizing the evolution equations in units of ns.
            default : 1e-2 ns
        save_at : list or None
            a list of normalized annela time (from 0 to 1) points for which the
            solver saves the result of the evolution.
            If "None" then uses the number points in the flux schedule.
            default : None
        basis_s : float
            the normalized anneal time for which the solver fixes the basis.
            Should be between (including) 0 and 1.
            initial_state should be represented in the same basis as this.
            default : 0

        Returns
        -------
        sol : ndarray
            Solution of the Schrodinger equation for the time evolution.
            sol[i] is the state vector saved at time i.
            dim=(len(save_at), nmax)
        """
        t_list = np.linspace(0, tf, num=int(tf / dt))

        if save_at is None:
            s_list = np.linspace(0, 1, phi_dict["points"])
        else:
            s_list = save_at

        self._initialize_se(phi_dict)

        self.calculate_quantum(self.x_interp(basis_s), self.z_interp(basis_s), sw=False)
        ham_basis = copy.deepcopy(self.low_e_dict)

        sol = np.zeros((len(s_list), self.nmax), dtype=complex)
        state = init_state
        sol[0] = state
        ctr = 1
        for i in range(len(t_list[:-1])):
            # use mid-point for unitary propogation, it's more accurate
            # result is the solution at t_list[i+1]
            t_mid = (t_list[i] + t_list[i + 1]) / 2
            s_mid = t_mid / tf

            phi_x_vec = self.x_interp(s_mid)
            phi_z_vec = self.z_interp(s_mid)
            self.calculate_quantum(phi_x_vec, phi_z_vec, sw=False)
            h_tot = self.get_hams(basis=ham_basis)[1]
            state = expm(-1j * h_tot * dt) @ state

            if np.abs(t_list[i + 1] / tf - s_list[ctr]) < dt / tf / 2:
                sol[ctr] = state
                ctr += 1

        return sol
