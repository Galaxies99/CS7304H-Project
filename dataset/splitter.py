import numpy as np


class Splitter(object):
    """
    Training Validation Splitter.
    """
    def __init__(self, training_features, training_labels, val, split_file, val_with_noise = False, noise_file = None, **kwargs):
        super(Splitter, self).__init__()
        if val not in ['holdout', 'cross-validation']:
            raise AttributeError('Invalid validation type.')
        if val_with_noise and noise_file is None:
            raise AttributeError('Please provide the noise file when using val_with_noise option.')   
        self.X = np.load(training_features)
        self.y = np.load(training_labels)
        self.val = val
        self.split = np.load(split_file, allow_pickle = True).item()
        self.val_with_noise = val_with_noise
        if self.val_with_noise:
            self.noise = np.load(noise_file)
        if self.val == 'holdout':
            self.training_set = self.split['train']
            self.validation_set = self.split['val']
            self.X_train, self.y_train = self.X[self.training_set], self.y[self.training_set]
            self.X_val, self.y_val = self.X[self.validation_set], self.y[self.validation_set]
            if self.val_with_noise:
                assert self.noise.shape == self.X_val.shape
                self.X_val = self.X_val + self.noise
        elif self.val == 'cross-validation':
            if self.val_with_noise:
                assert self.noise.shape == self.X.shape
                self.noisy_X = self.X + self.noise
            self.K = len(self.split.keys())
            self.X_train = []
            self.y_train = []
            self.X_val = []
            self.y_val = []
            for k in range(self.K):
                val = self.split['{}'.format(k)]
                train = []
                for kk in range(self.K):
                    if kk != k:
                        train.append(self.split['{}'.format(kk)])
                train = np.hstack(train)
                train = train[np.random.permutation(len(train))]
                self.X_train.append(self.X[train])
                self.y_train.append(self.y[train])
                if self.val_with_noise:
                    self.X_val.append(self.noisy_X[val])
                else:
                    self.X_val.append(self.X[val])
                self.y_val.append(self.y[val])
        else:
            raise AttributeError('Invalid validation type.')
    
    def __call__(self, index: int = 0):
        if self.val == 'holdout':
            return self.X_train, self.y_train, self.X_val, self.y_val
        elif self.val == 'cross-validation':
            if index < 0 or index >= self.K:
                raise AttributeError('Index {} out of bound.'.format(index))
            return self.X_train[index], self.y_train[index], self.X_val[index], self.y_val[index]
        else:
            raise AttributeError('Invalid validation type.')

    def get_K(self):
        return self.K if self.val == 'cross-validation' else 0
    
