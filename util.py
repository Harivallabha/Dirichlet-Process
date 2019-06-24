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
from pymc3.distributions.continuous import Beta
from pymc3.distributions.dist_math import bound
from ..model import (
    Model, get_named_nodes_and_relations, FreeRV,
    ObservedRV, MultiObservedRV, Context, InitContextMeta
)


class StickBreaking(Continuous):
    R"""
    Stick-Breaking Weights log-likelihood.

    ========  ===============================================
    Support   :math:`w_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum w_i = 1``
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
        shape = num_comp   # Num_comp should be equal to the number of weights
        kwargs.setdefault("shape", shape)
        super().__init__(*args, **kwargs)
        self.a = tt.as_tensor_variable(a)
        self.k = tt.as_tensor_variable(shape)
        self.mean = 1.  # Just to get stuff running. Testing the logp right now.
        self.trans = transform

    def _random(self, size=None):
        samples = np.array(np.arange(0, 100))
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

        Continuous.__init__(self, transform=self.trans)
        wts = tt.as_tensor_variable(weights)
        wt = wts[:-1]
        wt_sum = tt.extra_ops.cumsum(wt)
        denom = 1 - wt_sum
        denom_shift = tt.concatenate([[1.], denom])
        betas = wts/denom_shift
        Beta_ = pm.Beta.dist(1, a)
        return bound(Beta_.logp(betas).sum(), tt.all(betas > 0))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        return r'${} \sim \text{{Stick-Breaking}}(\mathit{{a}}={})$'.format(name,
                                                get_variable_name(a))
