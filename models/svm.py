import numpy as np
from sklearn.svm import SVC


class BinarySVM(object):
    """
    Binary Support Vector Machine
    """
    def __init__(
        self, 
        kernel = "linear", 
        C = 1.0, 
        max_iter = 300,
        tol = 1e-3,
        sigma = 1.0,
        degree = 3,
        kernel_a = 1.0,
        kernel_c = 1.0,
        **kwargs
    ):
        """
        Args:
          @ kernel: the kernel type, should be in ["linear", "poly", "rbf"];
          @ C: the penalty term;
          @ max_iter: the maximum number of iterations;
          @ tol: the tolerance error;
          @ sigma: the parameter sigma in "rbf" kernel;
          @ degree: the degree parameter in "poly" kernel;
          @ kernel_a: the parameter a in "sigmoid" kernel and "poly" kernel;
          @ kernel_c: the parameter c in "sigmoid" kernel and "poly" kernel.
        """
        super(BinarySVM, self).__init__()
        if kernel == 'linear':
            self.K = lambda a, b: np.sum(a * b, axis = -1)
        elif kernel == 'rbf':
            self.K = lambda a, b: np.exp(- np.sum((a - b) ** 2, axis = -1) / (2 * sigma ** 2))
        elif kernel == 'poly':
            self.K = lambda a, b: (kernel_a * np.sum(a * b, axis = -1) + kernel_c) ** degree
        elif kernel == 'sigmoid':
            self.K = lambda a, b: np.tanh(kernel_a * np.sum(a * b, axis = -1) + kernel_c)
        else:
            raise AttributeError('Unsupported kernel type.')
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
          @ X: np.ndarray, the given training data;
          @ y: np.ndarray, the given training labels.
        """
        self.X = X
        self.y = y
        self.alpha = np.ones([len(X)])
        self.b = 0.0
        cur_iter = 0
        while self.max_iter == -1 or cur_iter < self.max_iter:
            cur_iter += 1
            with_upd = False
            err = np.array([self.__err(i) for i in range(len(X))])
            for i in range(len(X)):
                err_i = self.__err(i)
                if err_i == 0 or self.__kkt_cond(i):
                    continue
                j = np.argmin(err) if err_i > 0 else np.argmax(err)
                if i == j:
                    continue
                err_j = self.__err(j)
                if y[i] * y[j] < 0:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                if L == H:
                    continue
                eta = self.K(X[i], X[i]) + self.K(X[j], X[j]) - 2 * self.K(X[i], X[j])
                if eta <= 0:
                    continue
                with_upd = True
                alpha_j = np.clip(self.alpha[j] + y[j] * (err_i - err_j) / eta, L, H)
                alpha_i = self.alpha[i] + y[i] * y[j] * (self.alpha[j] - alpha_j)
                delta_alpha_i = alpha_i - self.alpha[i]
                delta_alpha_j = alpha_j - self.alpha[j]
                b_i = -err_i - y[i] * self.K(X[i], X[i]) * delta_alpha_i - y[j] * self.K(X[i], X[j]) * delta_alpha_j + self.b
                b_j = -err_j - y[i] * self.K(X[j], X[i]) * delta_alpha_i - y[j] * self.K(X[j], X[j]) * delta_alpha_j + self.b
                self.alpha[i] = alpha_i
                self.alpha[j] = alpha_j
                if 0 < alpha_i < self.C:
                    self.b = b_i
                elif 0 < alpha_j < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i + b_j) / 2
                err[i] = self.__err(i)
                err[j] = self.__err(j)
            if not with_upd:
                break
    
    def state_dict(self):
        """
        Fetch the state dict of the model.

        Returns:
          @ state_dict: the state dict of the model.
        """
        return {'X': self.X, 'y': self.y, 'alpha': self.alpha, 'b': self.b}
    
    def load_state_dict(self, state_dict):
        """
        Load the state dict into the model.
        
        Args:
          @ state_dict: the state dict of the model.
        """
        self.X = state_dict['X']
        self.y = state_dict['y']
        self.alpha = state_dict['alpha']
        self.b = state_dict['b']
    
    def save(self, path):
        """
        Save model.
        
        Args:
          @ path: the path to save model.
        """
        np.save(path, self.state_dict())
    
    def load(self, path):
        """
        Load model.
        
        Args:
          @ path: the path to load model.
        """
        self.load_state_dict(np.load(path, allow_pickle = True).item())
    
    def predict(self, X):
        """
        Args:
          @ X: np.ndarray, the inference data.
        Returns:
          @ y_pred: np.ndarray, the predicted labels.
        """
        y_pred = np.array([self.__g(x) for x in X])
        return np.where(y_pred > 0, 1, -1)

    def __g(self, x):
        return np.sum(self.alpha * self.y * self.K(self.X, x)) + self.b

    def __err(self, i):
        return self.__g(self.X[i]) - self.y[i]
    
    def __kkt_cond(self, i):
        g = self.__g(self.X[i])
        y = self.y[i]
        if np.abs(self.alpha[i]) < self.tol:
            return g * y >= 1
        if np.abs(self.alpha[i]) > self.C - self.tol:
            return g * y <= 1
        return np.abs(g * y - 1) < self.tol


class SVM(object):
    """
    Categorical Support Vector Machine using OvR
    """
    def __init__(
        self, 
        num_classes,
        kernel = "linear", 
        C = 1.0, 
        max_iter = 300,
        tol = 1e-3,
        sigma = 1.0,
        degree = 3,
        kernel_a = 1.0,
        kernel_c = 1.0,
        **kwargs
    ):
        self.svms = []
        self.num_classes = num_classes
        for _ in range(num_classes):
            self.svms.append(BinarySVM(kernel = kernel, C = C, max_iter = max_iter, tol = tol, sigma = sigma, degree = degree, kernel_a = kernel_a, kernel_c = kernel_c, **kwargs))
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
          @ X: np.ndarray, the given training data;
          @ y: np.ndarray, the given training labels.
        """
        for c in range(self.num_classes):
            yy = np.where(y == c, 1, -1)
            self.svms[c].fit(X, yy)
    
    def predict(self, X):
        """
        Args:
          @ X: np.ndarray, the inference data.
        Returns:
          @ y_pred: np.ndarray, the predicted labels.
        """
        res = []
        for x in X:
            y = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                y[c] = self.svms[c].__g(x)
            res.append(y.argmax())
        return np.array(res, dtype = np.int)  

    def state_dict(self):
        """
        Fetch the state dict of the model.

        Returns:
          @ state_dict: the state dict of the model.
        """
        state_dict = {}
        for c in range(self.num_classes):
            state_dict['{}'.format(c)] = self.svms[c].state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Load the state dict into the model.
        
        Args:
          @ state_dict: the state dict of the model.
        """
        for c in range(self.num_classes):
            self.svms[c].load_state_dict(state_dict['{}'.format(c)])
    
    def save(self, path):
        """
        Save model.
        
        Args:
          @ path: the path to save model.
        """
        np.save(path, self.state_dict())
    
    def load(self, path):
        """
        Load model.
        
        Args:
          @ path: the path to load model.
        """
        self.load_state_dict(np.load(path, allow_pickle = True).item())
