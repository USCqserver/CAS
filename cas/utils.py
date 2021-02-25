"""Constants and helper methods for use within the schedule submodule.
"""
import numpy as np
from scipy import constants
from scipy import sparse
import scipy.special as special

BOLTZ = constants.value("Boltzmann constant in Hz/K")
E_CHARGE = constants.value("atomic unit of charge")
H_PLANCK = constants.value("Planck constant")
PHI_0 = constants.value("mag. flux quantum")


def e_j(i_c):
    """Returns the josephson energy for the given critical current
    in GHZ (omega)
    E_j = Phi_0/2pi * I_c

    Arguments
    ---------
    i_c : float
        Junction critical current, in nA

    Returns
    -------
    float
         Josephson energy in GHz (i.e., omega = 2*pi*f)

    """
    return PHI_0 * (i_c * 1e-9) / H_PLANCK / 1e9


def e_c(c):
    """Returns the charging energy for the given capacitance
     in GHZ (omega)
     E_c = (2e)^2/2 * 1/c

    Arguments
    ---------
    c : float
        capacitance in fF

    Returns
    -------
    float
         Charging energy in GHz (i.e., omega = 2*pi*f)
    """
    return (2 * np.pi) * 2 * E_CHARGE ** 2 / (c * 1e-15) / H_PLANCK / 1e9


def e_l(l):
    """Returns the inductive energy for the given inductance
     in GHZ (omega)
     E_l = 1/2 * (Phi_0/2pi)^2 * 1/L

    Arguments
    ---------
    l : float
        inductance in pH

    Returns
    -------
    float
         inductive energy in GHz (i.e., omega = 2*pi*f)
    """
    return PHI_0 ** 2 / (4 * np.pi) / (l * 1e-12) / H_PLANCK / 1e9


def multi_krond(arr_list):
    """Calculates Kronecker product (tensor product) between multiple
    dense arrays.

    The arrays are multiplied in order arr_list[0]*arr_list[1]*... arr_list[-1]

    Arguments
    ---------
    arr_list: list of ndarrays
        list of arrays to be multiplied (tensor product) together

    Returns
    -------
    product: ndarray
        Tensor product of dense arrays in the arr_list
    """
    product = arr_list[0]
    for array in arr_list[1::]:
        product = np.kron(product, array)
    return product


def multi_krons(arr_list):
    """Calculates Kronecker product (tensor product) between multiple
    sparse arrays.

    The arrays are multiplied in order arr_list[0]*arr_list[1]*... arr_list[-1]

    Arguments
    ---------
    arr_list: list of sparse arrays
        list of arrays to be multiplied (tensor product) together

    Returns
    -------
    product: sparse matrix
        Tensor product of sparse arrays in the arr_list
    """
    product = arr_list[0]
    for array in arr_list[1::]:
        product = sparse.kron(product, array)
    return product


def basis_vec(index, size):
    """Create a unit basis vector, with 1 for given index and zeros elsewhere.
     This is used for sparse matrix eigenvector search initial seed
    Arguments
    ---------
    index: int
        Index of the vector that should be set to 1
    size: int
        Size of the unit vector

    Returns
    -------
    v: array
        The unit basis vector
        dim=(1, size)
     """
    v = np.zeros(size)
    v[index] = 1
    return v


def _g(mu, beta):
    """ Special function used in calculation of the coupling strength via the
    Born-Oppenheimer method.
    Arguments
    ---------
    mu: int
        could be from 0 to inf
    beta: float
        coupler effective beta

    Returns
    -------
    out: float
        output of the special function
    """
    out = (
        (-beta / 2) ** mu
        * special.binom(1 / 2, mu)
        * special.hyp2f1(mu / 2 - 1 / 4, mu / 2 + 1 / 4, 1 + mu, beta ** 2)
    )
    return out


def _b(nu, beta, zeta):
    """ Special function used in calculation of the coupling strength via the
    Born-Oppenheimer method.
    Arguments
    ---------
    nu: int
        could be from -inf to inf
    beta: float
        coupler effective beta
    zeta: float
        coupler effective zeta

    Returns
    -------
    out: float
        output of the special function
    """
    if nu == 0:
        out = -beta ** 2 / 4 + zeta * (_g(0, beta) - beta * _g(1, beta))
    else:
        summation = sum(
            [
                mu / nu * _g(mu, beta) * special.jv(nu - mu, beta * nu)
                for mu in range(20)
            ]
        )
        out = special.jv(nu, beta * nu) / nu ** 2 + zeta * summation
    return out
