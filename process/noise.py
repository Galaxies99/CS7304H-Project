# Add noise to validation features.

import numpy as np

N_samples = 6666
N_validation = 667
N_stratified_validation = 675
M = 1536
mu = 0.0
sigma = 0.8

epsilon_samples = np.random.randn(N_samples, M) * sigma + mu
epsilon_valid = np.random.randn(N_validation, M) * sigma + mu
epsilon_stratified_valid = np.random.randn(N_stratified_validation, M) * sigma + mu

np.save('data/all_manual_noise.npy', epsilon_samples)
np.save('data/validation_manual_noise.npy', epsilon_valid)
np.save('data/validation_stratified_manual_noise.npy', epsilon_stratified_valid)