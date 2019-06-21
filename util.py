import numpy as np
import theano
import theano.tensor as tt
import warnings
import scipy
import numpy as np

import pymc3 as pm

from pymc3.theanof import floatX
from pymc3.distributions import transforms
from pymc3.util import get_variable_name
from pymc3.distributions.distribution import (Continuous, Discrete, draw_values, generate_samples,
                           _DrawValuesContext)
from pymc3.distributions.continuous import ChiSquared, Normal
from pymc3.distributions.special import gammaln, multigammaln
from pymc3.distributions.dist_math import bound, logpow, factln
from ..model import (
    Model, get_named_nodes_and_relations, FreeRV,
    ObservedRV, MultiObservedRV, Context, InitContextMeta
)


class StickBreaking(Continuous): 
    R"""
    Stick-Breaking Weights log-likelihood.

    ========  ===============================================
    Support   :math:`w_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum w_i = 1`
    Mean      :math:`\dfrac{w_i}{\sum w_i}`
    ========  ===============================================

    Parameters
    ----------
    a : array
        Concentration parameters (a > 0).
    """

    def __init__(self, a, weights, transform=transforms.stick_breaking,
                 *args, **kwargs):
        shape = np.atleast_1d(a.shape)[-1]

        kwargs.setdefault("shape", shape)
        super().__init__(transform=transform, *args, **kwargs)

        self.size_prefix = tuple(self.shape[:-1])
        self.k = tt.as_tensor_variable(shape)
        self.a = a = tt.as_tensor_variable(a)
        self.wts = wts = tt.as_tensor_variable(weights)
        self.mean = wts / tt.sum(wts)

        """self.mode = tt.switch(tt.all(a > 1),
                              (a - 1) / tt.sum(a - 1),
                              np.nan)"""

    def _random(self, a, size=None):
        gen = stats.dirichlet.rvs
        shape = tuple(np.atleast_1d(self.shape))
        if size[-len(shape):] == shape:
            real_size = size[:-len(shape)]
        else:
            real_size = size
        if self.size_prefix:
            if real_size and real_size[0] == 1:
                real_size = real_size[1:] + self.size_prefix
            else:
                real_size = real_size + self.size_prefix

        if a.ndim == 1:
            samples = gen(alpha=a, size=real_size)
        else:
            unrolled = a.reshape((np.prod(a.shape[:-1]), a.shape[-1]))
            samples = np.array([gen(alpha=aa, size=1) for aa in unrolled])
            samples = samples.reshape(a.shape)
        return samples

    def random(self, point=None, size=None):
        """
        Draw random values from Dirichlet distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        a = draw_values([self.a], point=point, size=size)[0]
        samples = generate_samples(self._random,
                                   a=a,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, value):
        """
        Calculate log-probability of the given set of weights.

        Parameters
        ----------
        value : 1-D array, having numeric values
            Set of weights for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        k = self.k
        a = self.a
        wts = self.wts
        #wts_delayed = np.concatenate([-999, wts])
        wts_delayed = wts



        
        #index = theano.shared(0)

        # only defined for sum(weights) 
        def inv_trans(prev_beta, weight, weight_prev):
                    #index_inc = function([], index, updates=[(index, index+1)])
                    #function()
                    #return tt.switch(T.gt(index-1,0),prev_beta*weight/(weight_prev*(1-prev_beta)), weight)
                    return tt.switch(tt.ge(wts_delayed, 0), prev_beta * weight / (weight_prev * (1 - prev_beta)), weight)

        betas, updates = theano.scan(fn=inv_trans,
                              outputs_info=[wts[0]],
                              sequences=[wts, wts_delayed])

        print(betas)
        return bound(1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        return r'${} \sim \text{{Stick-Breaking}}(\mathit{{a}}={})$'.format(name,
                                                get_variable_name(a))