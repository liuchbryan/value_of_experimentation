import numpy as np

def get_bootstrap_mean(samples, n_bootstraps):
    bootstrap_mean = []
    for i in range(0, n_bootstraps, 1):
        bootstrap_mean.append(
            np.mean(np.random.choice(samples, len(samples), 
                                     replace=True))
        )
    return bootstrap_mean

def get_bootstrap_var(samples, n_bootstraps):
    bootstrap_var = []
    for i in range(0, n_bootstraps, 1):
        bootstrap_var.append(
            np.var(np.random.choice(samples, len(samples), 
                                    replace=True))
        )
    return bootstrap_var