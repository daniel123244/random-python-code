import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

class TransitionKernel:
    """ Abstract parent class to ensure transition
        kernels keep required form."""
    def __init__(self, symm):
        self.symm = symm

    def generate(self, sample):
        """ Generate a sample from P(-|sample). """
        pass

    def pdf(self, x, y):
        """ Computes P(y|x) """
        pass

class GaussKernel(TransitionKernel):
    """ Normal random walk. """
    def __init__(self, std):
        TransitionKernel.__init__(self, symm = True)
        self.std = std

    def generate(self, sample):
        """ Generates a new step sample + eps
            eps ~N(0, std^2)"""
        return sample + st.norm.rvs(loc=0, scale=self.std)

    def pdf(self, y, x):
        """ Computes P(y|x), the prob that x + eps < y"""
        return st.norm.pdf(y, loc=x, scale=self.std)

class MetropolisHastings:
    """ Class of MCMC methods based on the Metropolis-Hastings algorithm.
        -Sample from a distribution
        -Compute expected values E[f]
        Does not require distribution to be normalized"""

    class NotFitException(Exception):
        pass

    def __init__(self, dist, kernel, iters = 400):
        """ cdf is the desired cdf to approximate
            kernel is a transition kernel for MH"""
        if not isinstance(kernel, TransitionKernel):
            raise ValueError("Kernel must be a transition kernel.")
        self.dist = dist
        self.kernel = kernel
        self.iters = iters
        self.seed = None
        self.mean_ = None
        self.var_ = None

    def __stepChain(self, steps=5):
        """ """
        x_0 = self.seed

        for i in range(steps):
            x_1 = self.kernel.generate(x_0)

            if self.kernel.symm:
                alpha = min(1, self.dist(x_1)/self.dist(x_0))

            else:
                alpha = min(1, self.dist(x_1)*self.kernel.pdf(x_0, x_1)\
                               /self.dist(x_0)*self.kernel.pdf(x_1, x_0))

            x_0 = x_1 if np.random.rand() < alpha else x_0

        # keep updating the seed every call to converge to stationary
        self.seed = x_0

    def fit(self, seed=None):
        """ Step through the algorithm far enough to get
            a good approximation to cdf.  Store the resulting
            seed state """
        if seed:
            self.seed = seed

        else:
            self.seed = np.random.rand()

            for i in range(self.iters):
                #update x_0
                self.__stepChain()

    def plot(self, n_samples = 1000, kde = False):
        if self.seed == None:
            raise MetropolisHastings.NotFitException('MetropolisHastings instance not fit.')

        xs = list(self.generate(n_samples))
        sns.distplot(xs, kde=kde)
        plt.show()

    def sample(self):
        """ Draw a single sample from a distribution. """
        if self.seed == None:
            raise MetropolisHastings.NotFitException('MetropolisHastings instance not fit.')

        self.__stepChain()
        return self.seed

    def generate(self, size = 1):
        """ Create generator object of samples from a fit distribution."""
        if self.seed == None:
            raise MetropolisHastings.NotFitException('MetropolisHastings instance not fit.')

        return (self.sample() for _ in range(size))

    def expectedValue(self, f, courseness = 100):
        """ Compute the expected value E[f] using the ergodic theorem. 
            P[lim(1/N)Sum(f(x_i)) = E[f]] = 1."""

        samples = self.generate(courseness)

        return  sum( (f(sample) for sample in samples) )/courseness

    def mean(self, courseness = 100):
        """ Compute expected value of the distribution."""
        if self.mean_:  return self.mean_ 
        else: return self.expectedValue(lambda i: i, courseness)

    def var(self, courseness = 100):
        """ Compute the variance of the distribution."""
        if self.var_: return self.var_

        mu = self.mean(courseness)
        return self.expectedValue(lambda i : (i - mu)**2, courseness )



def exp(x):
    return 0 if x < 0 else np.exp(-x)

def norm(x):
    return np.exp(-x**2/2)


def main():

    kern = GaussKernel(std = 1)

    mh = MetropolisHastings(norm, kern)

    mh.fit()

    mh.plot()

if __name__ == '__main__':
    main()