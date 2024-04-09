"""
Core Corrections Class
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.ma as ma

logger = logging.getLogger(__name__)


class AbstractConstraint(ABC):
    @property
    @abstractmethod
    def size(self):
        """
        Get the number of constraint rows, i.e., the number of scalar constraint
        equations

        Returns:
            int: the number of constraint rows
        """
        pass

    @abstractmethod
    def evaluate(self, freeVarIndexMap, freeVarVec):
        """
        Evaluate the constraint

        Args:
            freeVarIndexMap (dict): maps the first index (:class:`int`) of each⋅
                :class:`Variable` within ``freeVars`` to the variable object
            freeVarVec (numpy.ndarray<float>): free variable vector

        Returns:
            numpy.ndarray<float> the value of the constraint funection; evaluates
            to zero when the constraint is satisfied

        Raises:
            RuntimeError: if the constraint cannot be evaluated
        """
        pass

    @abstractmethod
    def partials(self, freeVarIndexMap, freeVarVec):
        """
        Compute the partial derivatives of the constraint vector with respect to
        all variables

        Args:
            freeVarIndexMap (dict): maps the first index (:class:`int`) of each⋅
                :class:`Variable` within ``freeVars`` to the variable object
            freeVarVec (numpy.ndarray<float>): free variable vector

        Returns:
            dict: a dictionary mapping a :class:`Variable` object to the partial
            derivatives of this constraint with respect to that variable⋅
            (:class:`numpy.ndarray<float>`). The partial derivatives with respect
            to variables that are not included in the returned dict are assumed
            to be zero.
        """
        pass


class Variable:
    """
    Contains a variable vector with an optional mask to flag non-variable values

    Args:
        values (float, [float], np.ndarray<float>): scalar or array of variable
            values
        mask (bool, [bool], np.ndarray<bool>): ``True`` flags values as excluded
            from the free variable vector; ``False`` flags values as included
    """

    def __init__(self, values, mask=False, name=""):
        self.values = ma.array(values, mask=mask, ndmin=1)
        self.name = name

    @property
    def numFree(self):
        """
        Get the number of un-masked values, i.e., the number of free variables
        within the vector

        Returns:
            int: the number of un-masked values
        """
        return int(sum(~self.values.mask))


class CorrectionsProblem:
    """
    Defines a mathematical problem to be solved by a corrections algorithm

    Attributes:
        freeVarIndexMap (dict): maps the first index (:class:`int`) of a variable
            within ``freeVarVec`` to the corresponding :class:`Variable` object.
        constraintIndexMap (dict): maps the first index (:class:`int`) of the
            constraint equation(s) in ``constraintVec`` to the corresponding
            :class:`AbstractConstraint` object.
        freeVarVec (numpy.ndarray<float>): N-element free variable vector
        constraintVec (numpy.ndarray<float>): M-element constraint vector
        jacobian (numpy.ndarray<float>): MxN Jacobian matrix. Each row contains
            the partial derivatives of a constraint equation with respect to
            the free variable vector. Thus, rows correspond to constraints and
            columns correspond to free variables.
    """

    def __init__(self):
        self._freeVarIndexMap = {}
        self._constraintIndexMap = {}

        self._freeVarVec = np.empty((0,))
        self._constraintVec = np.empty((0,))
        self._jacobian = np.empty((0,))

    # -------------------------------------------
    # Variables

    def addVariable(self, variable):
        if not isinstance(variable, Variable):
            raise ValueError("Can only add Variable objects")
        self._freeVarIndexMap[variable] = None

    def rmVariable(self, variable):
        if variable in self._freeVarIndexMap:
            del self._freeVarIndexMap[variable]
        else:
            logger.error(f"Could not remove variable {variable}")

    def clearVariables(self):
        self._freeVarIndexMap = {}

    def freeVarVec(self, recompute):
        if recompute:
            self._freeVarVec = np.zeros((self.numFreeVars,))
            for var, ix in self._freeVarIndexMap.items():
                self._freeVarVec[ix : ix + var.numFree] = var.values[~var.values.mask]

        return self._freeVarVec

    def freeVarIndexMap(self, recompute):
        if recompute:
            # TODO sort variables by type?
            count = 0
            for var in self._freeVarIndexMap:
                self._freeVarIndexMap[var] = count
                count += var.numFree

        return self._freeVarIndexMap

    @property
    def numFreeVars(self):
        return sum([var.numFree for var in self._freeVarIndexMap])

    # TODO (internal?) function to update variable objects with values from
    #   freeVarVec? Not sure if needed

    # -------------------------------------------
    # Constraints

    def addConstraint(self, constraint):
        if not isinstance(constraint, AbstractConstraint):
            raise ValueError("Can only add AbstractConstraint objects")
        self._constraintIndexMap[constraint] = None

    def rmConstraint(self, constraint):
        if constraint in self._constraintIndexMap:
            del self._constraintIndexMap[constraint]
        else:
            logger.error(f"Could not remove constraint {constraint}")

    def clearConstraints(self):
        self._constraintIndexMap = {}

    def constraintVec(self, recompute):
        if recompute:
            self._constraintVec = np.zeros((self.numConstraints,))
            for constraint, ix in self._constraintIndexMap.items():
                self._constraintVec[ix : ix + constraint.size] = constraint.evaluate(
                    self._freeVarIndexMap, self._freeVarVec
                )
        return self._constraintVec

    def constraintIndexMap(self, recompute):
        if recompute:
            # TODO sort constraints by type?
            count = 0
            for con in self._constraintIndexMap:
                self._constraintIndexMap[con] = count
                count += con.size

        return self._constraintIndexMap

    @property
    def numConstraints(self):
        return sum([con.size for con in self._constraintIndexMap])

    # -------------------------------------------
    # Jacobian

    def jacobian(self, recompute):
        if recompute:
            # Loop through constraints and compute partials with respect to all
            #   of the free variables
            for constraint, cix in self._constraintIndexMap.items():
                # Compute the partials of the constraint with respect to the free
                #   variables
                partials = constraint.partials(self._freeVarIndexMap, self._freeVarVec)

                for partialVar, partialMat in partials.items():
                    # Mask the partials to remove columns associated with variables
                    #   that are not free variables
                    maskedMat = partialVar.maskPartials(partialMat)

                    if maskedMat.size > 0:
                        # TODO insert partialMat into Jacobian
                        pass

        return self._jacobian
