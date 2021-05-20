from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Distribution(stats.rv_continuous):
    def _pdf(self, x, *args):
        return 0.5 * x

def monte_carlo_simulation():
    np.random.seed(54321)

    nmax = 10000
    # number of samples
    n_samples = np.arange(100, nmax + 100, 100)

    # To store each approximation of the expected value
    mu_estim = []
    mu_true = 4 / 3
    my_distribution = Distribution(a=0, b=2)
    for n in tqdm(n_samples):
        # Obtain a sample of size n
        # sample contains the x_i's
        sample = my_distribution.rvs(size=n)
        # Calculate the average of the sample
        mu_estim.append(sample.mean())

    # Create the convergence plot
    plt.plot(n_samples, mu_estim, '-g', alpha=0.5)
    plt.hlines(y=mu_true, xmin=n_samples[0],
               xmax=n_samples[-1], colors='blue', lw=6.5)
    plt.title(f'Convergence to $Np = ${mu_true}')
    plt.xlabel('Sample size')
    plt.ylabel('$E[X]$')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    monte_carlo_simulation()
