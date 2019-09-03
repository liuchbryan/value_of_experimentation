import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

def get_partial_noise_reduction_value_samples(
    n_samples, N, M, mu_X, mu_epsilon, 
    sigma_sq_X, sigma_sq_1, sigma_sq_2, p,
    verbose=True):
    # Numpy's normal RNG takes sigmas instead of sigma squares
    # We need to convert the sigma squares to prevent unexpected results
    sigma_X = np.sqrt(sigma_sq_X)
    sigma_1 = np.sqrt(sigma_sq_1)
    sigma_2 = np.sqrt(sigma_sq_2)
    
    # The lists to hold the samples
    noisy_mean = []
    clean_mean = []
    improvement = []

    for i in range(0, n_samples):
        # Step 1
        propositions = (
            pd.DataFrame(np.random.normal(mu_X, sigma_X, N), 
                         columns=['true']))

        # Step 2
        propositions['observed_noisy'] = (
            propositions['true'] + 
            np.random.normal(mu_epsilon, sigma_1, N))

        # Step 3
        propositions['observed_noisy_rank'] = (
            rankdata(propositions['observed_noisy']))

        # Step 4
        noisy_chosen_true = (
            propositions[propositions['observed_noisy_rank'] > (N-M)]
                        ['true']
        )

        noisy_chosen_true_mean = noisy_chosen_true.mean()
        noisy_mean.append(noisy_chosen_true_mean)

        # Step 5- repeat step 2 for sigma^2_2
        selector = np.random.binomial(1, p, N)
        propositions['observed_clean'] = (
            propositions['true'] +
            (selector * np.random.normal(mu_epsilon, sigma_2, N) +
             (1 - selector) * np.random.normal(mu_epsilon, sigma_1, N))
        )

        # Step 5- repeat 3 for sigma^2_2
        propositions['observed_clean_rank'] = (
            rankdata(propositions['observed_clean']))

        # Step 5- repeat 4 for sigma^2_2
        clean_chosen_true = (
            propositions[propositions['observed_clean_rank'] > (N-M)]
                        ['true']
        )

        clean_chosen_true_mean = clean_chosen_true.mean()
        clean_mean.append(clean_chosen_true_mean)

        # Step 6
        improvement.append(clean_chosen_true_mean - noisy_chosen_true_mean)

        # Reporting and progress tracking - print results of 20 samples
        if ((verbose==True) and (i % (n_samples / 20) == 0)):
            print("Noisy: {}, Clean: {}. Improvement: {} ({}%)".format(
                  np.round(noisy_chosen_true_mean, 4),
                  np.round(clean_chosen_true_mean, 4),
                  np.round(clean_chosen_true_mean - noisy_chosen_true_mean, 4),
                  np.round(100.0 * ((clean_chosen_true.mean() / 
                                     noisy_chosen_true.mean()) - 1), 3)
            ))

    # Making sure there is no off-by-one error on ranks
    # by checking the value of the last cycle
    assert(np.min(propositions['observed_noisy_rank']) == 1)
    assert(np.max(propositions['observed_noisy_rank']) == N) 
    assert(len(noisy_chosen_true) == M)
    assert(len(clean_chosen_true) == M)
    
    return(noisy_mean, clean_mean, improvement)