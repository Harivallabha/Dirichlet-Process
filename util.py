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
from pymc3.distributions.continuous import ChiSquared, Normal, Beta
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
    a : numeric
        Concentration parameter (a > 0).
    weights : 1-D array, numeric

    num_comp : integer
        Number of components of the truncated stick-breaking process (Truncation-level).
    """

    def __init__(self, a, num_comp, transform=transforms.stick_breaking,
                 *args, **kwargs):
        
        shape = num_comp #Num_comp should be equal to the number of weights
        kwargs.setdefault("shape", shape)
        super().__init__(transform=transform, *args, **kwargs)
        self.a = tt.as_tensor_variable(a)
        self.k = tt.as_tensor_variable(shape)

        """
        self.size_prefix = tuple(self.shape[:-1])
        self.k = tt.as_tensor_variable(shape)
        self.weights = weights
        self.wts = wts = tt.as_tensor_variable(weights)
        #self.wts = weights
         #= tt.as_tensor_variable(weights)
        #self.mean = a / tt.sum(a)
        #self.mean = sum(wts) / len(wts)
        self.mean = wts / tt.sum(wts)
        """

        """self.mode = tt.switch(tt.all(a > 1),
                              (a - 1) / tt.sum(a - 1),
                              np.nan)"""
        self.mean = 1.  #Just to get stuff running. Testing the logp right now.

    def _random(self, size=None):
        """shape = tuple(np.atleast_1d(self.shape))
        if size[-len(shape):] == shape:
            real_size = size[:-len(shape)]
        else:
            real_size = size
        if self.size_prefix:
            if real_size and real_size[0] == 1:
                real_size = real_size[1:] + self.size_prefix
            else:
                real_size = real_size + self.size_prefix

        if wts.ndim == 1:
            samples = np.arange(0,100)
        else:
            unrolled = wts.reshape((np.prod(wts.shape[:-1]), wts.shape[-1]))
            samples = np.array(np.arange(0,100))
            samples = samples.reshape(wts.shape)"""


        samples = np.array(np.arange(0,100))
        samples = samples.reshape(wts.shape)
        return samples

    def random(self, point=None, size=None):
        """
        Draw random values (weights) from the Stick-Breaking Process

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
        wts = draw_values([self.a], point=point, size=size)
        samples = generate_samples(self._random,
                                   wts=wts,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def get_betas(self, weights):
        # wt_shape = tt.shape(weights) #Note that this should be k - 1, we could've directly set wt_shape to k-1 as well
        # We want to index to go till k - 1, so arange should have input as wt_shape + 1
        betas, _ = theano.scan(fn=lambda prior_beta, index, weights: (prior_beta * weights[index] / (1 - prior_beta) * weights[index - 1]),
                              outputs_info=weights[0],
                              sequences = theano.tensor.arange(1 , k),
                              non_sequences=weights,
                              n_steps=k - 2) #Also, since we've used 'sequences', we needn't have n_steps set to k - 2
        return betas

    def logp(self, weights):
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
        k = self.shape
        a = self.a
        wts = tt.as_tensor_variable(weights)
        
        #len_ = len(weights)
        """
        beta_values = [weights[0]]
        for i in range(1, len_):
            beta_new = beta_values[i-1] * weights[i] / ((1 - beta_values[i-1]) * weights[i - 1])
            beta_values.append(beta_new)
        
        Beta_ = scipy.stats.beta(1, a)
        logp_betas = [Beta_.logpdf(x) for x in beta_values]
        print(logp_betas)
        return bound(sum(logp_betas))
        """
        weights_from_2_to_n_minus_one = wts[1:-1]
        betas_from_2_to_n_minus_one = self.get_betas(weights_from_2_to_n_minus_one)
        beta_values=tt.concatenate(wts[0], betas_from_2_to_n_minus_one)
        return tt.sum(continuous.Beta.dist(1,a).logp(beta_values))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        return r'${} \sim \text{{Stick-Breaking}}(\mathit{{a}}={})$'.format(name,
                                                get_variable_name(a))
