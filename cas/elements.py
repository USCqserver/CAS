"""Models of various quantum circuit elements to generate device Hamiltonians.

Currently contains different effective models for single elements:
1. 4-JJ CSFQ with an inductor using 3 modes, CSFQ
2. 4-JJ CSFQ using 1 mode, CSFQ2D
3. 4-JJ CSFQ using 2 modes, CSFQ2D
4. Coupler using 1 mode, Coupler

Right now these classes only return the gap (E_01) via a private method
_get_gap(), and calculation is deferred to the module h_to_flux.py.
This will be refactored such that these models return an interpolated
representation of the Hamiltonian, allowing the h_to_flux to focus
solely on that conversion and not eigenstate simulation.

"""
import warnings

import lmfit
import numpy as np
from scipy.linalg import expm
import scipy.sparse.linalg as sprsalg
from scipy import sparse, interpolate

from cas.utils import H_PLANCK, PHI_0, E_CHARGE
from cas.utils import multi_krons, basis_vec
from cas.utils import e_j, e_c, e_l

warnings.filterwarnings("ignore", message="splu requires CSC matrix format")
warnings.filterwarnings(
    "ignore",
    message="spsolve is more efficient when sparse b is " "in the CSC matrix format",
)


class CSFQ:
    """Creates 4-JJ CSFQ qubit with addition of an inductor.

    It has 3 modes, where the first  mode is represented in harmonic oscillator
    basis and the next two modes are represented in charge basis.
    Uses sparse matrix implementation for fast and efficient calculations
    """

    def __init__(self, i_c, c_shunt, c_z, l, alpha, d, homax_0=3, qmax_1=10, qmax_2=5):
        """
        Arguments
        ---------
        i_c : float
            junction critical current, in nA
        c_shunt : float
            shunt capacitance, in fF
        c_z : float
            z-loop junction capacitance, in fF
        l : float
            inductance, in pH
        alpha : float
            ratio of the average current in x-loop junctions to the current in
            z-loop junction. alpha*Iz = (Ix1 + Ix2)/2.
            alpha < 1
        d : float
            x-loop junction asymmetry. d = (Ix1 - Ix2)/(Ix1 + Ix2).
        homax_0: int
            maximum number of harmonic oscillator basis for mode 0.
            default: 5
        qmax_1 : int
            maximum value of charge (in Cooper pairs) in the charge basis for
            mode 1.
            default: 10
        qmax_1 : int
            maximum value of charge (in Cooper pairs) in the charge basis for
            mode 2.
            default: 10

        Attributes
        ----------
        type : sting
            type of the object
        i_c : float
            junction critical current, in nA
        c_shunt : float
            shunt capacitance, in fF
        c_z : float
            z-loop junction capacitance, in fF
        l : float
            inductance, in pH
        alpha : float
            ratio of the average current in x-loop junctions to the current in
            z-loop junction. alpha*Iz = (Ix1 + Ix2)/2.
            alpha < 1
        d : float
            x-loop junction asymmetry. d = (Ix1 - Ix2)/(Ix1 + Ix2).
        asym_sign : float
            sign of asymmetry shift in phi_z, i.e., if asymmetry increases or
            decreases the degeneracy.
            For this circuit phi_z --> phi_z + phi_d, hence asym_sign=-1
        qmax_1 : int
            maximum value of charge (in Cooper pairs) in mode 1
            default: 10
        qmax_2 : int
            maximum value of charge (in Cooper pairs) in mode 2
            default: 10
        nmax_0 : int
            maximum number of levels in the mode 0
            default: 5
        nmax_1 : int
            maximum number of levels in the mode 1
        nmax_2 : int
            maximum number of levels in the mode 2
        nmax : int
            maximum number of levels in the circuit Hamiltonian
        a : sparse matrix
            lowering operator for the mode 0
            dim=(namx, nmax)
        phi_0: sparse matrix
            phase operator for the mode 0
            dim=(nmax, nmax)
        n_0: sparse matrix
            charge operator for the mode 0
            dim=(nmax, nmax)
        n_1: sparse matrix
            charge operator for the mode 1
            dim=(nmax, nmax)
        n_2: sparse matrix
            charge operator for the mode 1
            dim=(nmax, nmax)
        d1_0 : sparse matrix
            charge displacement (increases) operator by 1 cooper pair, mode 0
            dim=(nmax, nmax)
        d1_1 : sparse matrix
            charge displacement (increases) operator by 1 cooper pair, mode 1
            dim=(nmax, nmax)
        d1_2 : sparse matrix
            charge displacement (increases) operator by 1 cooper pair, mode 2
            dim=(nmax, nmax)
        n2_sqrd : sparse matrix
            n_2**2 operator
            dim=(nmax, nmax)
        n0_plus_n1_plus_n2_sqrd : sparse matrix
            (n_0 + n_1 + n_2)**2 operator
            dim=(nmax, nmax)
        n0_plus_n1_sqrd: sparse matrix
            (n_0 + n_1)**2 operator
            dim=(nmax, nmax)
        a_dagger_a: sparse matrix
            a^dagger * a operator
            dim=(nmax, nmax)
        d1_2_dot_d1_1_dagger: sparse matrix
            d1_2 @ d1_1^dagger operator
            dim=(namx, nmax)
        cos_phi1_minus_phi0: sparse matrix
            cos(phi_1 - phi_0) operator
            dim=(namx, nmax)
        sin_phi1_minus_phi0: sparse matrix
            sin(phi_1 - phi_0) operator
            dim=(namx, nmax)
        """
        self.type = "qubit"
        self.i_c = i_c
        self.c_shunt = c_shunt
        self.c_z = c_z
        self.l = l
        self.alpha = alpha
        self.d = d
        self.asym_sign = -1

        self.qmax_1 = qmax_1
        self.qmax_2 = qmax_2
        self.nmax_0 = homax_0
        self.nmax_1 = 2 * self.qmax_1 + 1  # -self.qmax <= q <= self.qmax
        self.nmax_2 = 2 * self.qmax_2 + 1  # -self.qmax <= q <= self.qmax
        self.nmax = self.nmax_0 * self.nmax_1 * self.nmax_2

        # lowering operator for the harmonic oscillator basis
        self.a = np.sqrt(sparse.diags([i for i in range(1, self.nmax_0)], 1))
        self.a = multi_krons(
            [self.a, sparse.identity(self.nmax_1), sparse.identity(self.nmax_2)]
        )

        # phase and charge operator for the harmonic oscillator basis
        self.phi_0 = (
            (e_c(2 * self.alpha * self.c_z) / e_l(self.l)) ** (1 / 4)
            * (self.a + self.a.T)
            / (np.sqrt(2))
        )
        self.n_0 = (
            (e_l(self.l) / e_c(2 * self.alpha * self.c_z)) ** (1 / 4)
            * (self.a - self.a.T)
            / (np.sqrt(2) * 1j)
        )

        # charge operator for charge bases: n_{basis_index}
        self.n_1 = sparse.diags([i for i in range(-self.qmax_1, self.qmax_1 + 1)])
        self.n_1 = multi_krons(
            [sparse.identity(self.nmax_0), self.n_1, sparse.identity(self.nmax_2)]
        )
        self.n_2 = sparse.diags([i for i in range(-self.qmax_2, self.qmax_2 + 1)])
        self.n_2 = multi_krons(
            [sparse.identity(self.nmax_0), sparse.identity(self.nmax_1), self.n_2]
        )

        # displacement operators of charge by 1 unit: d1_{basis_index}
        self.d1_0 = sprsalg.expm(1j * self.phi_0)
        self.d1_1 = sparse.diags([1 for i in range(1, self.nmax_1)], -1)
        self.d1_1 = multi_krons(
            [sparse.identity(self.nmax_0), self.d1_1, sparse.identity(self.nmax_2)]
        )
        self.d1_2 = sparse.diags([1 for i in range(1, self.nmax_2)], -1)
        self.d1_2 = multi_krons(
            [sparse.identity(self.nmax_0), sparse.identity(self.nmax_1), self.d1_2]
        )

        # precalculating things for faster Hamiltonian construction
        self.n2_sqrd = self.n_2.dot(self.n_2)
        self.n0_plus_n1_plus_n2_sqrd = (self.n_0 + self.n_1 + self.n_2).dot(
            self.n_0 + self.n_1 + self.n_2
        )
        self.n0_plus_n1_sqrd = (self.n_0 + self.n_1).dot(self.n_0 + self.n_1)
        self.a_dagger_a = self.a.T.dot(self.a)
        self.d1_2_dot_d1_1_dagger = self.d1_2.dot(self.d1_1.T)
        self.cos_phi1_minus_phi0 = (
            self.d1_1.dot(self.d1_0.conj().T) + self.d1_1.T.dot(self.d1_0)
        ) / 2
        self.sin_phi1_minus_phi0 = (
            (self.d1_1.dot(self.d1_0.conj().T) - self.d1_1.T.dot(self.d1_0)) / 2 / 1j
        )

    def re_init(self):
        """Re-initialize the qubit object"""
        self.__init__(
            self.i_c,
            self.c_shunt,
            self.c_z,
            self.l,
            self.alpha,
            self.d,
            self.nmax_0,
            self.qmax_1,
            self.qmax_2,
        )

    def get_h(self, phi_x, phi_z):
        """Builds the CSFQ qubit Hamiltonian.

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)

        Returns
        -------
        ham : sparse matrix
            circuit Hamiltonian
            dim=(nmax, nmax)
        """
        ham = (
            2 * np.sqrt(e_l(self.l) * e_c(2 * self.alpha * self.c_z)) * self.a_dagger_a
            + e_c(2 * self.c_shunt + self.c_z) * self.n0_plus_n1_plus_n2_sqrd
            + e_c(2 * self.c_shunt + self.c_z) * self.n0_plus_n1_sqrd
            + e_c(self.c_z * (2 * self.c_shunt + self.c_z) / self.c_shunt)
            * self.n2_sqrd
        ) - e_j(self.i_c) * (
            (
                np.exp(+1j * phi_z / 2) * self.d1_2
                + np.exp(-1j * phi_z / 2) * self.d1_2.T
            )
            / 2
            + (
                np.exp(-1j * phi_z / 2) * self.d1_2_dot_d1_1_dagger
                + np.exp(+1j * phi_z / 2) * self.d1_2_dot_d1_1_dagger.T
            )
            / 2
            + 2 * self.alpha * np.cos(phi_x / 2) * self.cos_phi1_minus_phi0
            + 2 * self.alpha * np.sin(phi_x / 2) * self.sin_phi1_minus_phi0 * self.d
        )
        return ham

    def get_ip(self, phi_x, phi_z):
        """Calculates the persistent current operator.

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
            Unused, but kept for consistency of method across models
        phi_z : float
            z (tilt) bias phase (not flux)

        Returns
        -------
        ip_hat : sparse matrix
            Persistent current operator, eigenvalues will have units of nA.
            dim=(nmax, nmax)

        """
        ip_hat = (
            -e_j(self.i_c)
            / 2
            * (
                (
                    np.exp(+1j * phi_z / 2) * self.d1_2
                    - np.exp(-1j * phi_z / 2) * self.d1_2.T
                )
                / 2
                / 1j
                - (
                    np.exp(-1j * phi_z / 2) * self.d1_2_dot_d1_1_dagger
                    - np.exp(+1j * phi_z / 2) * self.d1_2_dot_d1_1_dagger.T
                )
                / 2
                / 1j
            )
        )
        return ip_hat * H_PLANCK / PHI_0 * 1e18

    def get_povm(self, phi_x, phi_z, delta_i=10):
        """Calculates the POVM operator for measuring probability of right
        circulating current.

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)
        delta_i : float
            measurement current sensitivity, in nA
            Default: 10 nA

        Returns
        -------
        m_right : sparse matrix
            POVM for right circulating current.
            Left circulating current is m_left = np.eye(nmax) - m_right.
            dim=(nmax, nmax)
        """
        ip = self.get_ip(phi_x, phi_z).toarray()
        eig_vals, eig_vecs = np.linalg.eigh(ip)
        f_filter = lambda x: (np.tanh(x / delta_i) + 1) / 2
        # apply QFP filter and rotate back to the initial basis
        m_right = eig_vecs @ np.diag(f_filter(eig_vals)) @ eig_vecs.conj().T
        return sparse.csr_matrix(m_right)

    def get_phi_z_cutoff(self):
        """Here we calculate the upper limit of phi_z above which
        the first two eigenstates would be locailized in the same well
        and the persistent currents in low energy subspace would have two lowest
        eigenvalues with the same sign. Hence PC measurement would not be possible.
        Refer to discussion on page 2 of arXiv:2103.06461v1"""

        num_pts = 200
        phix_val = 2*np.pi
        """ We could take any value of phix_val in the flux qubit regime since
             phix_val would not have any effect on get_phi_z_cutoff
        """
        phi_z_cutoff = None
        phi_z_list = np.linspace(0.0, 0.05, num_pts)*2*np.pi
        for (i,phiz_val) in enumerate(phi_z_list):
            ham = self.get_h(phix_val, phiz_val)
            ip  = self.get_ip(phix_val, phiz_val)
            eign_e, eign_v = sprsalg.eigsh(ham, k=2, which="SA", v0=basis_vec(0, self.nmax))
            sort_index = np.argsort(eign_e)
            eign_e = eign_e[sort_index]
            eign_v = eign_v[:, sort_index]
            ip_low_e = eign_v.T.conj() @ ip @ eign_v
            ip_low_e = (ip_low_e + ip_low_e.conj().T) / 2  # assure hermitianity
            eig_vals, u_vec = np.linalg.eigh(ip_low_e)

            if (np.sign(eig_vals[0])==np.sign(eig_vals[1])):
                phi_z_cutoff = phiz_val
                break

        if phi_z_cutoff==None:
            phi_z_cutoff = phi_z_list[-1]

        return phi_z_cutoff


    def get_ising(self, phi_x, phi_z):
        """Calculates the Ising coefficients for single qubit.
        See arXiv:1912.00464 for more details.

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
            Should be in the range -2pi <= phi_x <= 2pi
        phi_z : float
            z (tilt) bias phase (not flux)
            Should be in the range 0 <= phi_z <= 2pi

        Returns
        -------
        ising : array
            sigma_x and sigma_z coefficients
            dim=(1, 2)
        basis : ndarray
            computational basis, which is a linear combination of first two
            eigenstates. First column is |0>, second column is |1>
            dim=(2, nmax)
        """
        ham = self.get_h(phi_x, phi_z)
        ip = self.get_ip(phi_x, phi_z)
        eign_e, eign_v = sprsalg.eigsh(ham, k=2, which="SA", v0=basis_vec(0, self.nmax))
        sort_index = np.argsort(eign_e)
        eign_e = eign_e[sort_index]
        eign_v = eign_v[:, sort_index]

        ip_low_e = eign_v.T.conj() @ ip @ eign_v
        ip_low_e = (ip_low_e + ip_low_e.conj().T) / 2  # assure hermitianity
        _, u_vec = np.linalg.eigh(ip_low_e)

        # remove sigma_y term
        u_vec[:, 0] = u_vec[:, 0] * abs(u_vec[:, 0][0]) / u_vec[:, 0][0]
        u_vec[:, 1] = u_vec[:, 1] * abs(u_vec[:, 1][0]) / u_vec[:, 1][0]

        h_eff = u_vec.conj().T.dot(np.diag(eign_e[0:2])).dot(u_vec)
        h_eff = (h_eff + h_eff.T.conj()) / 2  # assure hermitianity

        sigma_x = ((h_eff[0, 1] + h_eff[1, 0]) / 2).real
        sigma_z = ((h_eff[0, 0] - h_eff[1, 1]) / 2).real

        # Make sigma_x coefficient always positive
        if sigma_x < 0:
            u_vec[:, 0] = -u_vec[:, 0]
            sigma_x = -sigma_x

        ising = np.array([sigma_x, sigma_z])
        basis = eign_v.dot(u_vec)  # first column |0>, second |1>
        # basis will have different phase on each execution, but it's fine.
        # Reason: u_vec has different phase due to small sigma_y term removal
        return ising, basis

    def get_pauli(self, phi_x, phi_z):
        """Uses the single qubit computational basis to calculate Pauli matrices
        Matrices are represented in the same basis as the qubit representation

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
            Should be in the range -2pi <= phi_x <= 2pi
        phi_z : float
            z (tilt) bias phase (not flux)
            Should be in the range 0 <= phi_z <= 2pi

        Returns
        -------
        ndarray
            sigma_i, sigma_x, sigma_y, and sigma_z pauli matrices (in order)
            of the single qubit
            dim=(4, nmax, nmax)
        """
        _, basis = self.get_ising(phi_x, phi_z)
        basis_0 = basis[:, 0].reshape(-1, 1)
        basis_1 = basis[:, 1].reshape(-1, 1)

        # construct pauli operators
        sigma_i = basis_0 @ basis_0.conj().T + basis_1 @ basis_1.conj().T
        sigma_x = basis_0 @ basis_1.conj().T + basis_1 @ basis_0.conj().T
        sigma_y = -1j * basis_0 @ basis_1.conj().T + 1j * basis_1 @ basis_0.conj().T
        sigma_z = basis_0 @ basis_0.conj().T - basis_1 @ basis_1.conj().T

        return np.array([sigma_i, sigma_x, sigma_y, sigma_z])

    def get_pauli_low_e(self, phi_x, phi_z, eign_e, eign_v):
        """Uses single qubit computational basisto calculate single qubit
        Pauli matrices, and then projects the Pauli matrices onto a
        low energy subspace that is given by eigen_v parameter

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
            Should be in the range -2pi <= phi_x <= 2pi
        phi_z : float
            z (tilt) bias phase (not flux)
            Should be in the range 0 <= phi_z <= 2pi
        eign_e : array
            list of sorted low energy eigenvalues (from smallest to large)
            calculated for the same phi_x and phi_z biases
            dim=(1, len(eign_e))
            FIXME: this argument is not used anymore in favor of shorter code
        eign_v : ndarray
            Corresponding sorted eigenvectors (eigenvectors in columns)
            calculated for the same phi_x and phi_z biases
            dim=(nmax, len(eign_e))

        Returns
        -------
        ndarray
            sigma_i, sigma_x, sigma_y, and sigma_z pauli matrices (in order)
            of the single qubit, projected onto low energy subspace
            dim=(4, len(eign_e), len(eign_e))
        """
        [sigma_i, sigma_x, sigma_y, sigma_z] = self.get_pauli(phi_x, phi_z)

        # project pauli operators onto low energy eigenspace
        sigma_i = eign_v.conj().T @ sigma_i @ eign_v
        sigma_x = eign_v.conj().T @ sigma_x @ eign_v
        sigma_y = eign_v.conj().T @ sigma_y @ eign_v
        sigma_z = eign_v.conj().T @ sigma_z @ eign_v

        return np.array([sigma_i, sigma_x, sigma_y, sigma_z])

    def _c(self, pauli_index, s, phi_x, phi_z):
        """Calculate a special function that is later used for derivation of
        coupling strength between qubits.

        Arguments
        ---------
        pauli_index : int
            index of the pauli matrix that we want to use.
            the order is 0:I, 1:X, 2:Y, 3:Z
        s : float
            argument of the special function
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)

        Returns
        -------
        out : float
            output of the special function
        """
        pauli_mat = self.get_pauli(phi_x, phi_z)[pauli_index]
        out = 1 / 2 * np.trace(pauli_mat @ sprsalg.expm(-1j * s * self.phi_0))
        return out

    def get_low_e(self, phi_x, phi_z, trunc=10):
        """Calculates low energy eigenvalues and eigenvectors of the circuit

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)
        trunc : int
            truncation: how many low energy eigenvalues and vectors to return

        Returns
        -------
        eign_e : array
            list of sorted low energy eigenvalues (from smallest to large)
            dim=(1, trunc)
        eign_v : ndarray
            list of sorted low energy eigenvectors (from smallest to large)
            Eigenvectors are in columns
            dim=(nmax, trunc)
        """
        ham = self.get_h(phi_x, phi_z)

        eign_e, eign_v = sprsalg.eigsh(
            ham, k=trunc, which="SA", v0=basis_vec(0, self.nmax)
        )
        sort_index = np.argsort(eign_e)
        eign_e = eign_e[sort_index]
        eign_v = eign_v[:, sort_index]
        return eign_e, eign_v

    def _residuals(self, params, x_trgt, z_trgt):
        """Residual function for finding single qubit custom fluxes.
        Calculates the difference between a target single qubit Ising coefficients
        and the qubit's Ising for given flux biases

        Arguments
        ---------
        params: object
            an instance of lmfit.Parameters(), which stores optimization
            parameters. It's a dictionary of lmfit.Parameter() objects.
            Optimization parameters are single qubit x and z flux biases.
        x_trgt: float
            target (desired) Pauli x coefficient
        z_trgt: float
            target (desired) Pauli z coefficient

        Returns
        -------
        array
            difference between target and calculated coefficients
            dim=(1, 2)
        """
        values = params.valuesdict()
        phi_x = values["phi_x"]
        phi_z = values["phi_z"]
        # 10 MHz error in X, helps with finding bias for small X (10x speedup)
        # X field decreases exponentially with barrier bias and becomes hard to
        # find the bias value for small X-fields without this error
        x_err = 10 / 1000 * 2 * np.pi
        x, z = self.get_ising(phi_x, phi_z)[0]
        return np.array([(x_trgt - x) / x_err, z_trgt - z])

    def get_fluxes(self, x_custom, z_custom, optimizer_method="leastsq", verbose=False):
        """Calculates the circuit fluxes for given x/z Pauli schedules.
        Numerically finds fluxes that produce the desired Pauli coefficients
        This optimizes the fluxes simultaneously.

        Arguments
        ---------
        x_custom : array
            target schedule for Pauli x coefficient
        z_custom : array
            target schedule for Pauli z coefficient
        optimizer_method : str
            Method used for optimizer
            default is "leastsq", another good option is "nelder"
        verbose : bool
            Weather to show the progress or not

        Returns
        -------
        ndarray:
            the barrier (x) and tilt (z) biases (phase NOT flux), that produce
            the x_custom and z_custom schedules.
            dim = (2, p_tot)
        """
        p_tot = len(x_custom)
        phi_x_list = np.zeros(p_tot)
        phi_z_list = np.zeros(p_tot)

        # object that stores optimization parameters:
        params = lmfit.Parameters()
        # It contains dictionary of Parameter() objects,
        # with keys corresponding to parameter name
        # initial value selected at degeneracy point of z-bias, important for
        # limiting the range of solutions found
        params["phi_x"] = lmfit.Parameter(
            name="phi_x", value=1.5 * np.pi, min=1 * np.pi, max=2 * np.pi
        )
        params["phi_z"] = lmfit.Parameter(
            name="phi_z", value=0, min=-0.01 * np.pi, max=0.01 * np.pi
        )

        for i, (x_trgt, z_trgt) in enumerate(zip(x_custom, z_custom)):
            if verbose:
                print("schedule point", i + 1, "/", p_tot, end="\x1b[1K\r")
            minner = lmfit.Minimizer(self._residuals, params, fcn_args=(x_trgt, z_trgt))
            result = minner.minimize(method=optimizer_method)
            # options={"xatol": 1e-6})

            residual_norm = np.linalg.norm(result.residual)
            target_norm = np.linalg.norm([x_trgt, z_trgt])
            rel_error = residual_norm / target_norm
            if rel_error > 1e-2:
                print(
                    "point #{0:d} single qubit residuals: \n".format(i),
                    result.residual,
                    "\n",
                )
                warnings.warn(
                    (
                        "For the point #{0:d}, solver found solutions that"
                        + " are not optimal. The relative error is"
                        + " {1:.2f} % for single qubit residuals"
                    ).format(i, rel_error * 100)
                )
            phi_x_list[i] = result.params["phi_x"].value
            phi_z_list[i] = result.params["phi_z"].value

            # set initial value of next search to the solution of this
            # only set for x cause finding small z-bias sometimes gets stuck
            # speeds up the solver
            params["phi_x"].set(value=phi_x_list[i])
            # params["phi_z"].set(value=phi_z_list[i])

        return [phi_x_list, phi_z_list]

    def evolve_se(self, phi_dict, tf, init_state, dt=1e-2, save_at=None, trunc=None):
        """Calculates the close system time evolution of the circuit using
        Schrodinger's equation

        Arguments
        ---------
        phi_dict : dictionary
            dictionary of qubit fluxes, keys are ["phix", "phiz", "points"]
        tf : float
            anneal time in ns
        init_state : array
            Initial state of the system
            dim = self.nmax
        dt : float
            time step for the evolution, in ns
            default = 1e-2
        save_at : array or None
            an array from 0 to 1, for which the solver saves the results
            If "None" uses the size of flux schedule
            default : None
        trunc : float
            truncation value for solving the low-energy Schrodinger equation
            default None does not use any truncation

        Returns
        -------
        sol : ndarray
            solution of the time evolution.
            dim = (dim of save_at, self.nmax)
        """
        t_list = np.linspace(0, tf, num=int(tf / dt))

        s_phi = np.linspace(0, 1, phi_dict["points"])
        x_interp = interpolate.interp1d(
            s_phi, phi_dict["phix"], kind="linear", fill_value="extrapolate"
        )
        z_interp = interpolate.interp1d(
            s_phi, phi_dict["phiz"], kind="linear", fill_value="extrapolate"
        )
        if save_at is None:
            s_list = s_phi
        else:
            s_list = save_at

        sol = np.zeros((len(s_list), self.nmax), dtype=complex)
        state = init_state
        sol[0] = state
        ctr = 1
        for i in range(len(t_list[:-1])):
            s1 = t_list[i] / tf
            s2 = t_list[i + 1] / tf
            s_mid = (s1 + s2) / 2
            # the method below uses mid-point for unitary propogation,
            # it's more accurate
            if trunc is not None:
                # low energy eigensystem
                ham = self.get_h(x_interp(s_mid), z_interp(s_mid))
                eign_e, eign_v = sprsalg.eigsh(
                    ham, k=trunc, which="SA", v0=basis_vec(0, self.nmax)
                )
                sort_index = np.argsort(eign_e)
                eign_e = eign_e[sort_index]
                eign_v = eign_v[:, sort_index]

                # project state onto low-e and do evolution, then rotate back
                state_low_e = eign_v.T.conj() @ state
                propagator = np.diag(np.exp(-1j * dt * eign_e))
                state_low_e = propagator @ state_low_e
                state = eign_v @ state_low_e

            elif trunc is None:
                ham = self.get_h(x_interp(s_mid), z_interp(s_mid))
                propagator = expm(-1j * dt * ham)
                state = propagator @ state

            if np.abs(t_list[i + 1] / tf - s_list[ctr]) < dt / tf / 2:
                sol[ctr] = state
                ctr += 1

        return sol


class Coupler:
    """Creates coupler circuit, represented in the harmonic oscillator basis.
    This circuit has only 1 mode and uses dense implementation
    (sparse implementation was slower due to low sparsity)
    """

    def __init__(self, i_sigma, c_sigma, l, d, homax_0=50):
        """
        Arguments
        ---------
        i_sigma : float
            sum of the two junction critical currents (I_x1 + I_x2), in nA
        c_sigma : float
            sum of the two junction capacitances (C_x1 + C_x2), in fF
        l : float
            Total inductance of the coupler loop, in pH
        d : float
            x-loop junction asymmetry. d = (I_x1 - I_x2)/(I_x1 + I_x2).
        homax_0: int
            maximum number of harmonic oscillator basis for mode 0.
            default: 50

        Attributes
        ----------
        type : string
            type of the object
        i_sigma : float
            sum of the two junction critical currents (I_x1 + I_x2), in nA
        c_sigma : float
            sum of the two junction capacitances (C_x1 + C_x2), in fF
        l : float
            Total inductance of the coupler loop, in pH
        d : float
            x-loop junction asymmetry. d = (I_x1 - I_x2)/(I_x1 + I_x2).
        asym_sign : float
            sign of asymmetry shift in phi_z, i.e., if asymmetry increases or
            decreases the degeneracy.
            For this circuit phi_z --> phi_z + phi_d, hence asym_sign=-1
        beta_max : float
            maximum coupler beta, defined as junction energy over
            inductive energy (with a factor of 1/2 to match literature)
            Another interpretation: maximum coupler nonlinearity
        zeta : float
            coupler impedance divided by resistance quantum (25.8 KiloOhm)
            (with a factor of 4pi to match literature)
        nmax : int
            maximum number of levels (H.O. basis) in the circuit
            default: 50
        a : ndarray
            lowering operator for the mode 0
            dim=(namx, nmax)
        phi_0: ndarray
            phase operator for the mode 0
            dim=(nmax, nmax)
        n_0: ndarray
            charge operator for the mode 0
            dim=(nmax, nmax)
        d1_0 : sparse matrix
            charge displacement (increases) operator by 1 cooper pair, mode 0
            dim=(nmax, nmax)
        a_dagger_a: sparse matrix
            a^dagger * a operator
            dim=(nmax, nmax)
        """
        self.type = "coupler"
        self.i_sigma = i_sigma
        self.c_sigma = c_sigma
        self.l = l
        self.d = d
        self.asym_sign = -1

        self.beta_max = e_j(self.i_sigma) / e_l(self.l) / 2
        self.zeta = np.sqrt(self.l / self.c_sigma * 1e3) * 4 * np.pi / 25812.807

        self.nmax = homax_0

        self.a = np.sqrt(np.diag([i for i in range(1, self.nmax)], 1))
        self.phi_0 = (
            (e_c(self.c_sigma) / e_l(self.l)) ** (1 / 4)
            * (self.a + self.a.T)
            / (np.sqrt(2))
        )
        self.n_0 = (
            (e_l(self.l) / e_c(self.c_sigma)) ** (1 / 4)
            * (self.a - self.a.T)
            / (np.sqrt(2) * 1j)
        )

        # displacement operators of charge by 1 unit
        self.d1_0 = expm(1j * self.phi_0)
        self.a_dagger_a = self.a.T @ self.a

    def re_init(self):
        """Re-initialize the coupler object"""
        self.__init__(self.i_sigma, self.c_sigma, self.l, self.d, self.nmax)

    def get_h(self, phi_x, phi_z):
        """Builds the coupler Hamiltonian.

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)

        Returns
        -------
        ham : ndarray
            circuit Hamiltonian
            dim=(nmax, nmax)
        """
        ham = 2 * np.sqrt(e_c(self.c_sigma) * e_l(self.l)) * self.a_dagger_a - e_j(
            self.i_sigma
        ) * (
            np.cos(phi_x / 2)
            * (
                np.exp(-1j * phi_z) * self.d1_0
                + np.exp(1j * phi_z) * self.d1_0.T.conj()
            )
            / 2
            + np.sin(phi_x / 2)
            * (
                np.exp(-1j * phi_z) * self.d1_0
                - np.exp(1j * phi_z) * self.d1_0.T.conj()
            )
            / 2
            / 1j
            * self.d
        )
        return ham

    def get_low_e(self, phi_x, phi_z, trunc=5):
        """Calculates low energy eigenvalues and eigenvectors of the circuit

        Arguments
        ---------
        phi_x : float
            x (barrier) bias phase (not flux)
        phi_z : float
            z (tilt) bias phase (not flux)
        trunc : int
            truncation: how many low energy eigenvalues and vectors to return

        Returns
        -------
        eign_e : array
            list of sorted low energy eigenvalues (from smallest to large)
            dim=(1, trunc)
        eign_v : ndarray
            list of sorted low energy eigenvectors (from smallest to large)
            Eigenvectors are in columns
            dim=(nmax, trunc)
        """
        # make sure trunc is not larger than the maximum circuit size
        trunc = min(self.nmax, trunc)
        ham = self.get_h(phi_x, phi_z)
        eign_e, eign_v = np.linalg.eigh(ham)
        return eign_e[0:trunc], eign_v[:, 0:trunc]

    def get_p0_low_e(self, eign_v):
        """Calculates projection operator onto ground state of the coupler, and
        then projects it onto low energy eigenspace that is given by eign_v

        Arguments
        ---------
        eign_v : ndarray
            Sorted low energy eigenvectors (eigenvectors in columns)
            Assume "trunc" is the number of low energy eigenvectors in eign_v
            dim=(nmax, trunc)

        Returns
        -------
        p0 : ndarray
            Projection operator onto ground state, projected onto low energy
            eigenspace
            dim=(trunc, trunc)
        """
        v_0 = eign_v[:, 0].reshape(-1, 1)
        p0 = v_0 @ v_0.conj().T
        p0 = eign_v.conj().T @ p0 @ eign_v
        return p0

    def evolve_se(self, phi_dict, tf, init_state, dt=1e-2, save_at=None, trunc=None):
        """Calculates the close system time evolution of the circuit using
        Schrodinger's equation

        Arguments
        ---------
        phi_dict : dictionary
            dictionary of qubit fluxes, keys are ["phix", "phiz", "points"]
        tf : float
            anneal time in ns
        init_state : array
            Initial state of the system
            dim = self.nmax
        dt : float
            time step for the evolution, in ns
            default = 1e-2
        save_at : array or None
            an array from 0 to 1, for which the solver saves the results
            If "None" uses the size of flux schedule
            default : None
        trunc : float
            truncation value for solving the low-energy Schrodinger equation
            default None does not use any truncation

        Returns
        -------
        sol : ndarray
            solution of the time evolution.
            dim = (dim of save_at, self.nmax)
        """
        t_list = np.linspace(0, tf, num=int(tf / dt))

        s_phi = np.linspace(0, 1, phi_dict["points"])
        x_interp = interpolate.interp1d(
            s_phi, phi_dict["phix"], kind="linear", fill_value="extrapolate"
        )
        z_interp = interpolate.interp1d(
            s_phi, phi_dict["phiz"], kind="linear", fill_value="extrapolate"
        )
        if save_at is None:
            s_list = s_phi
        else:
            s_list = save_at

        sol = np.zeros((len(s_list), self.nmax), dtype=complex)
        state = init_state
        sol[0] = state
        ctr = 1
        for i in range(len(t_list[:-1])):
            s1 = t_list[i] / tf
            s2 = t_list[i + 1] / tf
            s_mid = (s1 + s2) / 2
            # the method below uses mid-point for unitary propogation,
            # it's more accurate
            if trunc is not None:
                # low energy eigensystem
                ham = self.get_h(x_interp(s_mid), z_interp(s_mid))
                eign_e, eign_v = np.linalg.eigh(ham)
                eign_e = eign_e[0:trunc]
                eign_v = eign_v[:, 0:trunc]

                # project state onto low-e and do evolution, then rotate back
                state_low_e = eign_v.T.conj() @ state
                propagator = np.diag(np.exp(-1j * dt * eign_e))
                state_low_e = propagator @ state_low_e
                state = eign_v @ state_low_e

            elif trunc is None:
                ham = self.get_h(x_interp(s_mid), z_interp(s_mid))
                propagator = expm(-1j * dt * ham)
                state = propagator @ state

            if np.abs(t_list[i + 1] / tf - s_list[ctr]) < dt / tf / 2:
                sol[ctr] = state
                ctr += 1

        return sol
