"""
This module contains functions to calculate the errors of a binomial distribution.

This module follows:
    "Confidence Intervals for a binomial proportion and asymptotic expansions",
    Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta
    Ann. Statist. Volume 30, Number 1 (2002), 160-201.
    http://projecteuclid.org/euclid.aos/1015362189

"""

import numpy
from scipy.stats import beta


def _normalize_(a, n):
    """
    This function is used by all.
    """
    N = numpy.atleast_1d(n)
    N = numpy.ma.MaskedArray(N, mask=(N == 0))
    A = numpy.atleast_1d(a.astype(float))
    p = numpy.ones_like(A) * numpy.nan
    p[~N.mask] = A[~N.mask] / N[~N.mask]
    return A, N, p


def binomial_errors(a, n, coverage=0.682689):
    """Binomial error calculation using the Jeffreys prior.

    It returns a tuple with the ratio, the lower error, and the upper error.

    This is an equal-tailed Bayesian interval with a Jeffreys prior (Beta(1/2,1/2)).

    "Statistical Decision Theory and Bayesian Analysis" 2nd ed.
    Berger, J.O. (1985)
    Springer, New York

    Parameters
    ----------
    a : numpy array
        The number of 'successes'
    n : numpy array
        The number of trials
    coverage : float
        The coverage of the confidence interval. The default corresponds to
        '1-sigma' gaussian errors.
    """

    shape = numpy.broadcast_shapes(numpy.shape(a), numpy.shape(n))

    a = numpy.atleast_1d(a)
    n = numpy.atleast_1d(n)

    try:
        float(coverage)
    except ValueError:
        raise Exception("The coverage must be a float!") from None

    A, N, p = _normalize_(a, n)
    alpha_2 = (1 - coverage) / 2
    low = numpy.zeros_like(p)
    up = numpy.ones_like(p)
    low[p > 0] = beta.isf(1 - alpha_2, A[p > 0] + 0.5, N[p > 0] - A[p > 0] + 0.5)
    up[p < 1] = beta.isf(alpha_2, A[p < 1] + 0.5, N[p < 1] - A[p < 1] + 0.5)
    p, low, up = (p.reshape(shape), low.reshape(shape), up.reshape(shape))
    return p, low, up
