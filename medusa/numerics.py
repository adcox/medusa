"""
Numerics
========

Routines for numerical approximations of calculus evaluations are included here.

.. autosummary::
   derivative
   derivative_multivar

.. autofunction:: derivative
.. autofunction:: derivative_multivar
"""
import numpy as np


def derivative(func, x, step, nIter=10, maxRelChange=2.0, meta={}):
    """
    Compute the derivative of a function at ``x``.

    Args:
        func: a function handle that accepts a single argument, ``x``, and
            returns a single object (either a scalar :class:`float` or an
            :class:`~numpy.ndarray`).
        x (float, numpy.ndarray): the state at which to evaluate ``func``
        step (float, numpy.ndarray): the initial step size; it need not be small
            but rather should be an increment in ``x`` over which ``func`` changes
            *substantially*.
        meta (dict): a dictionary in which to store metadata about the derivative
            calculations
        nIter (int): maximum number of iterations, i.e., the max size of the
            Neville tableau
        maxRelChange (float): Return when the error is worse than the best so
            far by this factor.


    Returns:
        float, numpy.ndarray: the derivative of ``func`` with respect to ``x``,
        evaluated at the given value of ``x``.

    The algorithm implemented in this function is "Richardson's deferred approach
    to the limit," detailed in Numerical Recipes in C, 2nd Edition by Press,
    Teukolsky, Vetterling, and Flannery (1996)
    """
    CON = 1.4  # step size is decreased by CON at each iteration
    CON2 = CON * CON

    # Generalize to arbitrary dimensions
    x = np.asarray(x)
    step = np.asarray(step)
    stepSz = np.linalg.norm(step)

    if np.all(step == 0.0):
        raise ValueError("Step must be greater than zero")

    step = abs(step)
    tableau = np.empty((nIter, nIter), dtype=object)

    # Initial tableau entry is central difference
    tableau[0, 0] = (func(x + step) - func(x - step)) / (2 * stepSz)

    # TODO add check for subsequent derivatives being equal (to some tolerance?)

    row, col = 0, 0
    err = 1e30  # init error to huge value so that "deriv" is saved at least once
    for col in range(1, nIter):
        # Successive columns in the Neville tableu will go to smaller stepsizes
        # and higher orders of extrapolation
        step /= CON
        stepSz /= CON
        tableau[0, col] = (func(x + step) - func(x - step)) / (2 * stepSz)
        fac = CON2

        for row in range(1, col + 1):
            # Compute extrapolations of various orders, requiring no new func evals
            tableau[row, col] = (
                tableau[row - 1, col] * fac - tableau[row - 1, col - 1]
            ) / (fac - 1.0)
            fac *= CON2
            errt = max(
                np.linalg.norm(tableau[row, col] - tableau[row - 1, col]),
                np.linalg.norm(tableau[row, col] - tableau[row - 1, col - 1]),
            )

            # The error strategy is to compare each new extrapolation to one
            # order lower, both at the present step size and the previous one
            if errt <= err:
                # If error is decreased, save the improved answer
                err = errt
                deriv = tableau[row, col]

        if (
            np.linalg.norm(tableau[row, col] - tableau[row - 1, col - 1])
            >= maxRelChange * err
        ):
            # If higher order is worse by a signficant factor SAFE, quit early
            break

    # Save metadata and return best derivative approximation
    meta.update({"err": err, "count": col, "tableau": tableau})
    return deriv


def derivative_multivar(func, x, step, nIter=10, maxRelChange=2.0):
    """
    Multivariate version of :func:`derivative`

    The inputs are identical to :func:`derivative` with the exception of the
    ``step``. If ``step`` is an array, it must have the same size as ``x`` and
    each entry is the step size corresponding to an entry in ``x``. If ``step``
    is instead a scalar, that single step size is used for all entries in ``x``.
    """
    x = np.asarray(x)

    # Ensure "step" is a vector the same size as x
    if isinstance(step, float):
        step = np.full(x.shape, step)
    else:
        step = np.asarray(step)
        if not step.shape == x.shape:
            raise ValueError(
                f"'step' is a vector with shape {step.shape}, but that shape"
                f" doesn't match the state vector {x.shape}"
            )

    deriv = np.zeros(((func(x)).size, x.size))
    for ix in range(x.size):
        pert = np.zeros(x.shape)
        pert[ix] = step[ix]
        deriv[:, ix] = np.squeeze(
            derivative(func, x, pert, nIter=nIter, maxRelChange=maxRelChange)
        )

    # TODO can we report out the metadata for each variable?
    return deriv
