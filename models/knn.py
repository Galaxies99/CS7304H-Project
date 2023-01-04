import numpy as np


class KNN(object):
    """
    K Nearest Neighbor
    """
    def __init__(
        self,
        k = 10,
        distance_type = 'minkowski',
        order = 2,
        **kwargs
    ):
        """
        Args:
          @ k: the parameter "k" in K Nearest Neighbor;
          @ distance_type: the type of the distances, should be one of "minkowski" and "cosine";
          @ order: the order of the distance in Minkowski distance.
        """
        super(KNN, self).__init__()
        assert distance_type in ['minkowski', 'cosine']
        self.k = k
        self.order = order
        self.distance_type = distance_type
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
          @ X: np.ndarray, the given training data;
          @ y: np.ndarray, the given training labels.
        """
        self.X = X
        self.y = y

    def state_dict(self):
        """
        Fetch the state dict of the model.

        Returns:
          @ state_dict: the state dict of the model.
        """
        return {'X': self.X, 'y': self.y}
    
    def load_state_dict(self, state_dict):
        """
        Load the state dict into the model.
        
        Args:
          @ state_dict: the state dict of the model.
        """
        self.X = state_dict['X']
        self.y = state_dict['y']
    
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
        y_pred = np.zeros([len(X)]).astype(np.int)
        for i, x in enumerate(X):
            if self.distance_type == 'minkowski':
                dist = np.linalg.norm(self.X - x, ord = self.order, axis = 1)
            elif self.distance_type == 'cosine':
                dist = 1.0 - np.dot(self.X, x) / (np.linalg.norm(self.X, axis = 1) * np.linalg.norm(x))
            else:
                raise AttributeError('Unimplemented distance type.')
            topk = np.argsort(dist)[: self.k]
            y_pred[i] = np.bincount(self.y[topk]).argmax()
        return y_pred

