import numpy as np


def colebrook(x0, e, D, Re):
    """
    Colebrook-White equation.

    Parameters
    ------
    e : float
        Pipe's effective roughness height.
    D : float
        Hidraulic duct_diameter.
    Re : float
        Reynolds number.
    x0 : float
        Initial guess for the inverse of the square root of the Darcy
        friction factor.

    Returns
    ------
    float
        The inverse of the square root of the Darcy friction factor:
        `1 / sqrt(f)`

    """
    return -2 * np.log10((e / (3.7 * D)) + (2.51 / (Re * x0)))


def haaland(e, D, Re):
    """
    Haaland equation.

    Parameters
    ------
    e : float
        Pipe's effective roughness height.
    D : float
        Hidraulic duct_diameter.
    Re : float
        Reynolds number.

    Returns
    ------
    float
        The inverse of the square root of the Darcy friction factor:
        `1 / sqrt(f)`

    """
    return -1.8 * np.log10(((e / D) / 3.7) ** 1.11 + (6.9 / Re))


def darcy_friction(f_invsqrt):
    """

    Parameters
    ------
    f_invsqrt : float
        The inverse of the square root of the Darcy friction factor:
        `1 / sqrt(f)`

    Returns
    ------
    float
        Darcy friction factor `f`.

    """
    return f_invsqrt ** -2


def fixed_point(fun, x0, args=(), rtol=1e-4, maxiter=1000):
    """
    Fixed-point iteration.

    Parameters
    ------
    fun : callable
    x0 : float
    args : tuple
    rtol : float
    maxiter : int

    Returns
    ------
    float

    Raises
    ------
    Exception

    """
    p1 = x0

    for _ in range(1, maxiter):
        p0 = p1
        p1 = fun(p0, *args)

        if np.all((p1 - p0) / p0 < rtol):
            return p1

    raise Exception('No convergence achieved')


def solve_friction_coeff(e, D, Re, rtol=1e-4, maxiter=1000):
    x0 = haaland(e, D, Re)
    return fixed_point(colebrook, x0, args=(e, D, Re), rtol=rtol, maxiter=maxiter)
