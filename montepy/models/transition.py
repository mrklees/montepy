from math import log, sqrt

from montepy.montecarlo import MonteCarlo

import torch
import torch.tensor as tensor
from torch.distributions import Bernoulli, LogNormal,\
                                Poisson, Gamma, NegativeBinomial,\
                                Normal, Beta


class TransitionModel(MonteCarlo):

    def __init__(self):
        self.samples = None
        self.variables = ['D', 'Dist', 'Schools', 'Sz', 'Az', 'Sf', 'Fa', 'Dt']
        # Define distribution parameters
        self.n_decisions = 2
        self.n_districts = tensor([6.])
        # System & Infrastructure
        az_params = [self._lognormal_params(10000., 2000.),
                     self._lognormal_params(7000., 2000.)]
        self.az_means = [params[0] for params in az_params]
        self.az_sds = [params[1] for params in az_params]
        fa_params = [self._beta_params(0.1, 25),
                     self._beta_params(0.05, 15)]
        self.fa_ms = [params[0] for params in fa_params]
        self.fa_ks = [params[1] for params in fa_params]
        dt_params = [self._beta_params(0.05, 25),
                     self._beta_params(0.05, 15)]
        self.dt_ms = [params[0] for params in dt_params]
        self.dt_ks = [params[1] for params in dt_params]
    
    def _sampler(self, samples=1000):
        d_ = torch.ones(samples)
        if d == 1:
            # If SZ is adopted, then some Districts and Schools buy in
            dist = Poisson(self.n_districts)\
                    .sample([samples])\
                    .reshape([samples])
            schools = NegativeBinomial(tensor([3.]),
                                       tensor([0.8]))\
                        .sample([samples, self.n_districts.int()])\
                        .sum(dim=1)\
                        .reshape([samples])
            sz = 15000. * dist + 2430 * schools
        else:
            dist, schools, sz = torch.zeros(samples),\
                                torch.zeros(samples),\
                                torch.zeros(samples)
        if d < 2:
            sf = LogNormal(
                    *self._lognormal_params(300000., 10000.))\
                        .sample([samples])
        else:
            sf = torch.zeros(samples)
        # System & Infrastructure
        az = LogNormal(self.az_means[d], self.az_sds[d]).sample([samples])
        salary_estimate = Normal(70000., 5000.).sample([samples])        
        fa = Beta(self.fa_ms[d], self.fa_ks[d]).sample([samples])
        dt = Beta(self.dt_ms[d], self.dt_ks[d]).sample([samples])
        return d_, dist, schools, sz, az, sf, fa, dt
