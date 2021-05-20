import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def monte_carlo_simulation():
    np.random.seed(54321)
    nmax = 5000000
    # number of samples
    n_samples = np.arange(10000, nmax + 10000, 10000)

    # To store each approximation of the expected value
    mu_estim = []
    for n in tqdm(n_samples):
        # Obtain a sample of size n
        # sample contains the x_i's and y_i's
        x = np.random.uniform(low=0, high=2 * np.pi, size=(n, 1))
        y = np.random.uniform(low=0, high=2 * np.pi, size=(n, 1))
        s = np.sin(np.sqrt(np.square(x) + np.square(y)) + x + y)
        # Calculate the average of the sample
        mu_estim.append(s.mean())
    print(mu_estim[-1] * 4 * np.square(np.pi))
    # Create the convergence plot
    plt.plot(n_samples, mu_estim, '-g', alpha=0.5)
    plt.xlabel('Sample size')
    plt.ylabel('$E[X]$')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    monte_carlo_simulation()