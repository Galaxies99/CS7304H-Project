import numpy as np


def save_prediction(path, y):
    res = np.concatenate((np.arange(len(y)).reshape(-1, 1), y.reshape(-1, 1)), axis = -1).astype(np.int)
    np.savetxt(path, res, fmt = '%d', delimiter = ',', header = 'Id,Category', comments = '')
