"""Class for defining the parameters to be sampled during a Genetic Algorithm optimization run.

This module defines the class for defining the parameters and their
ranges that are to be sampled during the Genetic Algorithm optimization run.


"""

import numpy as np
from scipy.stats import uniform

class SampledParameter(object):
    """A parameter that will be sampled during a Nested Sampling run.
    Attributes:
        name (str,int): The name of this parameter.
        loc (float): The prior distribution object. This can be a fixed
            distribution from scipy.stats (e.g., scipy.stats.uniform) or
            a user-defined distribution class with the appropriate
            functions.
        width (float)
    """

    def __init__(self, name, loc, width):
        """Initialize the sampled parameter.
        Args:
            name (str,int): set the name Attribute.
            prior (:obj:): set the prior_dist Attribute.
        """
        self.name = name
        self.loc = loc
        self.width = width
        self._dist = uniform(loc, width)
        return

    def random(self, sample_shape):
        """Random variate sample.
        Args:
            sample_shape (int, tuple): The array size/shape for the random
            variate.
        Returns:
            (numpy.array): The set of random variate samples with length/shape
                sample_shape drawn form the prior distrbution.
        """
        return self._dist.rvs(sample_shape)

    def unit_transform(self, value):
        """The inverted cumulative density/mass function.
        Args:
            value (float, numpy.array): A value from the unit line [0:1].
        Returns:
            float, numpy.array: The transform of the unit line value to the
                associated value in the parameter's search space.
        """
        return self._dist.ppf(value)
