from math import log, sqrt

import pandas as pd
import numpy as np
import torch
import torch.tensor as tensor


class MonteCarlo(object):

    def __init__(self):
        self.samples = None
        self.variables = None

    def _sampler(self, samples=1000):
        """This function should r
        """
        pass

    def sample(self, samples=1000):
        s_ = torch.stack(self._sampler(samples=samples), dim=1)
        self.samples = pd.DataFrame(s_.numpy(),
                                    columns=self.variables)
        return self.samples

    def cache_samples(self, samples=5000):
        self.samples = self.sample(samples)

    def conditional(self, X, Y):
        """
        Calculates P(X | Y=y)

        Parameters
        ----------
        X : str
            Column name
        Y : str or iterable of strings
            Comma deliminated string or list of filters in the form:
            >>>.condition("Sz", "D = 1, Schools > 100")

        """
        if self.samples is None:
            print("No cached samples... Sampling with defaults...")
            self.cache_samples()
        else:
            return self.samples[self._parse(Y)][X].mean()

    def marginal(self, X):
        """
        Calculates P(X=x)

        Parameters
        ----------
        X : str or iterable of strings
            Comma deliminated string or list of filters in the form:
            >>>.marginal("D = 1, Schools > 100")
        """
        if self.samples is None:
            print("No cached samples... Sampling with defaults...")
            self.cache_samples()
        else:
            d = self.samples.shape[0]
            n = self.samples[self._parse(X)].shape[0]
            return n/d

    def _parse(self, query):
        """
        Parses the query, a filter statement of the sample written either as a
        comma deliminated list or in an iterable and returns the corresponding
        portion of the cached sample.

        A helper function for marginal and conditional.
        """
        if isinstance(query, str):
            clauses = query.split(", ")
        elif isinstance(query, (list, tuple)):
            clauses = query
        filters = []
        for clause in clauses:
            try:
                field, op, f = clause.split(" ")
                fil = self.translate(op)(self.samples[field],
                                         self.fix_type(f))
                filters.append(fil)
            except KeyError:
                print(f"Field {field} was not found in the\
                        sample... Did you spell it right?")
            except ValueError:
                print(f"You probably fucked {f} up...")
        return np.all(np.array(filters), axis=0)

    @staticmethod
    def _lognormal_params(m, s):
        """
        Torch requires that the lognormal distribution's parameters be given in
        terms of the non-logarithmized normal distribution.  This funciton
        converts the expected value and variance desired in the lognormal
        distribution to those parameters.
        """
        mu = log(m/sqrt(1+(s**2/m**2)))
        sigma = sqrt(log(1 + (s**2/m**2)))
        return tensor([mu, sigma])

    @staticmethod
    def _beta_params(m, k):
        """
        As is typical, we have to give pytorch Beta in terms of the normal
        parameters a, b. However, we typically have an intuition about the
        mode and variance we want from this distribution, not a, b.

        Parameters
        ----------
        m: float
            The mode of the distribution. Between 0 and 1.
        k: float
            A measure of the dispersion of the distribution. Greater than or
            equal to 2.
        """
        a = m * (k-2) + 1
        b = (1-m) * (k-2) + 1
        return tensor([a, b])

    @staticmethod
    def translate(op):
        if op == '>':
            return lambda x, y: x > y
        elif op == '>=':
            return lambda x, y: x >= y
        elif op == '<':
            return lambda x, y: x < y
        elif op == '<=':
            return lambda x, y: x <= y
        elif op == '=':
            return lambda x, y: x == y

    @staticmethod
    def fix_type(f):
        try:
            return float(f)
        except ValueError:
            return f
