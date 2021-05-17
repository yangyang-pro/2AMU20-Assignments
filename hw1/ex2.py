from scipy.stats import poisson
import random
import scipy.stats as sct
import numpy as np
import matplotlib.pyplot as plt

#function for ex. 2b
def aprox():
    np.random.seed(54321)
    
    nmax = 10000
    #X axis
    x_axis = np.arange(100, nmax + 100, 100)

    #To store each approximation of the expected value
    mu_estim = []
    mu = 2.65
    r = sct.poisson(mu)
    mu_true = r.mean()
    print(mu_true)

    for n in x_axis:
        #Obtain a sample of size n
        #sample contains the x_i's
        sample = sct.poisson.rvs(mu, size=n)
        #sample = rng.poisson(lam=(4), size=(10000, 2))
        
        #Calculate the average of the sample
        mu_estim.append(sample.mean())

    #Create the convergence plot
    plt.plot(x_axis, mu_estim, '-g', alpha = 0.5)
    plt.hlines(y = mu_true, xmin = x_axis[0], 
                xmax = x_axis[-1], colors = 'blue', lw = 6.5)
    plt.title(f'Convergence to $Np = ${mu_true}')
    plt.xlabel('Sample size')
    plt.ylabel('$E[X]$')
    plt.xticks(rotation = 90)
    plt.grid()
    plt.show()


if __name__ == '__main__': 
    aprox()

