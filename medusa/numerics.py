"""
Numerics
========

Routines for numerical approximations of calculus evaluations are included here.

Reference
-----------

.. autosummary::
   derivative
   derivative_multivar

.. autofunction:: derivative
.. autofunction:: derivative_multivar
.. autofunction:: linesearch

Sources
---------

.. [NumRecipes] 
   Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P.,
   "Numerical Recipes in C - 2nd Edition", Cambridge University Press, 1996

.. [Ridders]
   Ridders, C.J.F., "Accurate computation of F'(x)  and F'(x)F''(x)", 
   Advances in Engineering Software, vol 4, no. 2, April 1982, 
   pp. 75--76. doi: 10.1016/S0141-1195(82)80057-0

.. [Neville]
   Neville, E.H., "Iterative Interpolation", Journal of the Indian Mathematical Society,
   1934, pp. 87--120
"""
import logging
from typing import Callable, Union, overload

import numpy as np
from numpy.typing import NDArray

from medusa.typing import FloatArray

logger = logging.getLogger(__name__)


@overload
def derivative(
    func: Callable,
    x: float,
    step: float,
    nIter: int = 10,
    maxRelChange: float = 2.0,
    meta: dict = {},
) -> Union[float, NDArray[np.double]]:
    pass


@overload
def derivative(
    func: Callable,
    x: FloatArray,
    step: FloatArray,
    nIter: int = 10,
    maxRelChange: float = 2.0,
    meta: dict = {},
) -> Union[float, NDArray[np.double]]:
    pass


def derivative(func, x, step, nIter=10, maxRelChange=2.0, meta={}):
    """
    Compute the derivative of a function at ``x``.

    Given a function, :math:`\\vec{y} = \\vec{f}(\\vec{z})`, this method numerically
    computes the scalar derivative :math:`\mathrm{d}f / \mathrm{d}\zeta` evaluated
    at the ``x`` value. Because the function argument, :math:`\\vec{z}`, is only
    perturbed along one direction (``step``), the derivative only reflects variations
    along that vector.

    Note that :math:`f` can be a univariate function
    (:math:`f(z)`) or multivariate (:math:`f(\\vec{z})`) and may return a scalar
    or vector output.

    Args:
        func: a function handle that accepts a single argument, ``x``, and
            returns a single object (either a scalar :class:`float` or an
            :class:`~numpy.ndarray`).
        x: the state at which to evaluate ``func``
        step: the initial step size; it need not be small but rather should be an
            increment in ``x`` over which ``func`` changes *substantially*. The
            step must have the same shape as ``x`` (i.e., both scalars or both
            arrays of the same shape).
        meta: a dictionary in which to store metadata about the derivative
            calculations
        nIter: maximum number of iterations, i.e., the max size of the
            Neville tableau
        maxRelChange: Return when the error is worse than the best so
            far by this factor.

    Returns:
        the derivative of ``func`` with respect to ``x``, evaluated at the given
        value of ``x``.

    Raises:
        RuntimeError: if ``x`` and ``step`` have different shapes
        ValueError: if ``step`` includes only zeros

    **Algorithm**

    This method leverages the idea of "Richardson's deferred appraoch to the limit."
    We aim to extrapolate, to a step size of :math:`h \\to 0`, the result of finite
    difference calculations with smaller and smaller finite values of :math:`h`.
    Using Neville's algorithm [Neville]_, each new finite-difference calculation
    produces both an extrapolation of higher order and extrapolations of previous,
    lower-orders but with smaller scales :math:`h`.

    The code implemented here is transcribed from section 5.7 of [NumRecipes]_,
    which itself is based on [Ridders]_.
    """

    CON = 1.4  # step size is decreased by CON at each iteration
    CON2 = CON * CON

    # Generalize to arbitrary dimensions
    x, step = np.array(x, copy=True), np.array(step, copy=True)
    if not x.shape == step.shape:
        raise RuntimeError(
            f"x {x.shape} and step {step.shape} must have the same shapes"
        )

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
            errt = float(
                max(
                    np.linalg.norm(tableau[row, col] - tableau[row - 1, col]),
                    np.linalg.norm(tableau[row, col] - tableau[row - 1, col - 1]),
                )
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


def derivative_multivar(
    func: Callable,
    x: FloatArray,
    step: Union[float, FloatArray],
    nIter: int = 10,
    maxRelChange: float = 2.0,
) -> NDArray[np.double]:
    """
    Iteratively perturbs each element of ``x`` to compute the multivariate derivative.

    Given a function, :math:`\\vec{y} = \\vec{f}(\\vec{z})`, this method numerically
    computes the derivative :math:`\mathrm{d}\\vec{f} / \mathrm{d} \\vec{z}`,
    evaluated at the ``x`` value. Note that the function must multivariate
    (accepts a vector input) but can return a scalar or a vector.

    The inputs are identical to :func:`derivative` with the exception of the
    ``step``. If ``step`` is an array, it must have the same size as ``x`` and
    each entry is the step size corresponding to an entry in ``x``. If ``step``
    is instead a scalar, that single step size is used for all entries in ``x``.
    """
    x = np.array(x, copy=True)

    # Ensure "step" is a vector the same size as x
    if isinstance(step, float):
        step = np.full(x.shape, step)
    else:
        step = np.array(step, copy=True)
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


def linesearch(
    func: Callable,
    x: NDArray[np.double],
    funcVal: float,
    grad: NDArray[np.double],
    xStep: NDArray[np.double],
) -> tuple[NDArray[np.double], float, bool]:
    """
    Find a step size that sufficiently decreases the cost function.

    Args:
        func: a function handle that accepts an input state vector, :math:`\\vec{x}`,
            and returns a scalar, :math:`f`
        x: the current value of the state vector, :math:`\\vec{x}_0`.
        funcVal: the current value of ``func``, i.e., :math:`f_0 = f(\\vec{x}_0)`
        grad: the gradient of the function evaluated at the current 
            input vector, i.e., :math:`g_0 = \\nabla f(\\vec{x}_0)`
        xStep: the proposed step, :math:`\\delta \\vec{x}`. This
            is usually the Newton step, which is guaranteed to decrease :math:`f`
            for some small, scalar multiple of this step.

    Returns:
        a tuple with the new ``x`` vector, the new value of ``func``, and a boolean
        flag indicating whether the caller should check for a local minimum.

    **Algorithm**

    This algorithm and its derivation are transcribed from section 9.7 of 
    [NumRecipes]_ .

    The goal is to find an **attenuation factor**, :math:`\lambda`, that yields
    a new variable vector along the direction of the full step (but not necessarily
    all the way),

    .. math::
       \\vec{X}^* = \\vec{X}_0 + \lambda \delta \\vec{X}, \qquad 0 < \lambda \leq 1

    such that :math:`f(\\vec{X}_1)` has decreased sufficiently relative to 
    :math:`f(\\vec{X}_0)`. The algorithm proceeds as follows:

    1. First try :math:`\lambda = 1`, the full step. When :math:`\delta \\vec{X}`
       is the Newton step, this will lead to quadratic convergence when 
       :math:`\\vec{X}` is sufficiently close to the solution.
    2. If :math:`f(\\vec{X}_1)` does not meet acceptance criteria, *backtrack*
       along the step direction, trying a smaller value of :math:`\lambda` until
       a suitable point is found.

    To avoid constructing a sequence of steps that decrease :math:`f` too slowly
    relative to the step lengths,
    the average rate of decrease of :math:`f` is required to be at least some 
    fraction :math:`\\alpha` of the *initial* rate of decrease, 
    :math:`\\nabla f \cdot \delta \\vec{X}`:

    .. math::
       f(\\vec{X}^*) \leq f(\\vec{X}_0) + \\alpha \\nabla f \cdot (\\vec{X}^* - \\vec{X}_0)

    where :math:`0 < \\alpha < 1`. A small value, e.g., 1e-4, works well.

    To understand the backtracking routine, define

    .. math::
       g(\lambda) = f(\\vec{X}_0 + \lambda \delta \\vec{X})

    so that

    .. math::
       g'(\lambda) = \\nabla f \cdot \delta \\vec{X}

    If backtracking is needed, :math:`g` is modeled with the most current information
    available and :math:`\lambda` is selected to minimize the model. Initially,
    :math:`g(0) = \\vec{X}_0` and :math:`g'(0)` are available, as is 
    :math:`g(1) = \\vec{X} + \delta \\vec{X}`. A quadratic model can be defined,

    .. math::
       g(\lambda) \\approx [g(1) - g(0) - g'(0)]\lambda^2 + g'(0)\lambda + g(0)

    The derivative of the quadratic is zero (and, thus, the quadratic is minimized)
    when

    .. math::
       \lambda = -\\frac{ g'(0) }{2[g(1) - g(0) - g'(0)]}

    On the second and subsequent backtracks, :math:`g` is modeled as a cubic in
    :math:`\lambda`, using the previous value :math:`g(\lambda_1)` and second
    most recent value :math:`g(\lambda_2)`:

    .. math::
       g(\lambda) \\approx a \lambda^3 + b \lambda^2 + g'(0)\lambda + g(0)

    Requiring this expression to give the correct values of :math:`g` at 
    :math:`\lambda_1` and :math:`\lambda_2` gives two equations that can be solved
    for :math:`a` and :math:`b`:

    .. math::
       \\begin{bmatrix}a \\\\ b\end{bmatrix} =
       \\frac{1}{\lambda_1 - \lambda_2}
       \\begin{bmatrix}
         1/\lambda_1^2 & -1/\lambda_2^2\\\\
         -\lambda_2/\lambda_1^2 & \lambda_1/\lambda_2^2
       \end{bmatrix}
       \\begin{bmatrix}
         g(\lambda_1) - g'(0)\lambda_1 - g(0)\\\\
         g(\lambda_2) - g'(0)\lambda_2 - g(0)
       \end{bmatrix}

    The minimum of the cubic is at

    .. math::
       \lambda = \\frac{-b + \sqrt{b^2 - 3ag'(0)}}{3a}

    The computed :math:`\lambda` is enforced to remain between :math:`0.5\lambda_1`
    and :math:`0.1\lambda_1`.
    """
    tolX = 1e-7  # convergence criterion on x. TODO user input
    maxIt = 10  # TODO user input
    ALF = 1e-4  # ensures sufficient decrease in function value. TODO user input
    maxStep = 100  # TODO user input

    checkLocalMin = False

    # Initialize the full step; scale if attempted step is too big, e.g.,
    #   if there is some unbounded value thing going on
    xStep = np.array(xStep, copy=True)  # make a copy
    if np.linalg.norm(xStep) > maxStep:
        xStep *= maxStep / np.linalg.norm(xStep)

    # Compute initial rate of descent (ROD)
    initROD = sum(grad * xStep)
    if initROD > 0:
        # TODO use analytical rate of descent
        raise NotImplementedError

    # Compute minimum step size
    temp = np.asarray([max(abs(v), 1.0) for v in xStep])
    temp = abs(x) / temp
    minLam = tolX / max(temp)  # smallest permissible attenuation factor

    # Attenuation factor begins at 1.0 to try the full step first
    lam = 1.0
    lam_prev, funcVal_prev = 1.0, funcVal  # will be set later
    for it in range(maxIt):
        xNew = x + lam * xStep  # take a step
        funcValNew = func(xNew)
        logger.debug(
            f"Line search iteration {it:02d}: lambda = {lam:.2e}, funcVal = {funcValNew:.2e}"
        )

        if lam < minLam:
            xNew = np.array(xStep, copy=True)
            checkLocalMin = True
            logger.debug(f"Reached local minimum; lambda < minimum = {minLam:.2e}")
            return xNew, funcValNew, checkLocalMin
        elif funcValNew <= funcVal + ALF * lam * initROD:
            logger.debug("Line search converged")
            return xNew, funcValNew, checkLocalMin
        else:
            if lam == 1.0:
                # approximate step size as a quadratic
                lam_next = -initROD / (2 * (funcValNew - funcVal - initROD))
            else:
                # approximate step size as a cubic
                term1 = funcValNew - funcVal - lam * initROD
                term2 = funcVal_prev - funcVal - lam_prev * initROD
                a = (term1 / (lam * lam) - term2 / (lam_prev * lam_prev)) / (
                    lam - lam_prev
                )
                b = (
                    -lam_prev * term1 / (lam * lam)
                    + lam * term2 / (lam_prev * lam_prev)
                ) / (lam - lam_prev)

                if a == 0.0:
                    lam_next = -initROD / (2.0 * b)
                else:
                    disc = b * b - 3 * a * initROD
                    if disc < 0.0:
                        logger.warning("Roundoff error issue in linesearch")
                        lam_next = 0.5 * lam
                    elif b < 0.0:
                        lam_next = (-b + np.sqrt(disc)) / (3 * a)
                    else:
                        lam_next = -initROD / (b + np.sqrt(disc))

                # Ensure that the attenuation factor decreases by at least a factor of 2
                if lam_next > 0.5 * lam:
                    lam_next = 0.5 * lam

        lam_prev = lam
        funcVal_prev = funcValNew
        lam = max(lam_next, 0.1 * lam)  # keep lambda >= 0.1 lambda_prev

    if funcValNew <= funcVal + ALF * lam * initROD:
        raise RuntimeError("Line search has diverged")
    else:
        raise RuntimeError("Line search has reached max iterations with convergence??")
