import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class MLP_Network(nn.Module):
    def __init__(
        self,
        arch = [['Linear', 1536, 32], ['ReLU', True], ['Dropout', 0.2], ['Linear', 32, 20], ['Softmax', 1]],
        num_classes = 20,
        **kwargs
    ):
        super(MLP_Network, self).__init__()
        self.layers = []
        for layer in arch:
            name = layer[0]
            args = layer[1:]
            self.layers.append(getattr(nn, name)(*args))
        self.layers = nn.Sequential(*self.layers)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.layers(x)
    
    def classify(self, x):
        y_prob = self.forward(x)
        return torch.argmax(y_prob, 1)
    
    def loss(self, x, y):
        y_prob = self.forward(x)
        loss = F.cross_entropy(y_prob, y)
        return loss


class SimpleDataset(Dataset):
    def __init__(self, X, y, device = torch.device('cpu')):
        super(SimpleDataset, self).__init__()
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.device = device
        self.size = len(self.X)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        data_dict = {
            'X': torch.from_numpy(self.X[index]).float().to(self.device), 
            'y': torch.from_numpy(np.array(self.y[index])).long().to(self.device)
        }
        return data_dict



class MLP(object):
    def __init__(
        self,
        arch = [['Linear', 1536, 32], ['ReLU', True], ['Dropout', 0.2], ['Linear', 32, 20], ['Softmax', 1]],
        num_classes = 20,
        regularization = False,
        regularization_lambda = 0.01,
        epoch = 10,
        batch_size = 16,
        learning_rate = 1e-3,
        device = torch.device("cpu"),
        **kwargs
    ):
        super(MLP, self).__init__()
        self.mlp = MLP_Network(arch = arch, num_classes = num_classes, **kwargs).to(device)
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.mlp.parameters(), lr = learning_rate, weight_decay = regularization_lambda if regularization else 0)
        self.device = device

    def fit(self, X, y, X_val, y_val):
        self.mlp.apply(weight_reset)
        dataset = SimpleDataset(X, y, self.device)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        max_acc = 0
        best_mlp_state_dict = None
        for e in range(self.epoch):
            print('==> Epoch: {}'.format(e))
            self.mlp.train()
            with tqdm(dataloader) as pbar:
                for data_dict in pbar:
                    X_batch = data_dict['X']
                    y_batch = data_dict['y']
                    loss = self.mlp.loss(X_batch, y_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description('Loss: {}'.format(loss))
            self.mlp.eval()
            acc = 0
            for i in range(len(X_val)):
                X_sample = torch.from_numpy(X_val[i]).reshape(1, -1).float().to(self.device)
                y_sample = torch.from_numpy(np.array(y_val[i])).reshape(1).long().to(self.device)
                with torch.no_grad():
                    y_pred = self.mlp.classify(X_sample).item()
                if y_sample == y_pred:
                    acc += 1
            acc = acc / len(X_val)
            if acc > max_acc:
                max_acc = acc
                best_mlp_state_dict = self.mlp.state_dict()
            print('Accuracy: {} %'.format(acc * 100))
        self.mlp.load_state_dict(best_mlp_state_dict)
        return max_acc
    
    def predict(self, X):
        self.mlp.eval()
        y_pred = np.zeros(len(X), dtype = np.int)
        for i in range(len(X)):
            X_sample = torch.from_numpy(X[i]).reshape(1, -1).float().to(self.device)
            with torch.no_grad():
                y_pred_sample = self.mlp.classify(X_sample).item()
            y_pred[i] = y_pred_sample
        return y_pred

    def state_dict(self):
        return self.mlp.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.mlp.load_state_dict(state_dict)
    
    def save(self, path):
        """
        Save model.
        
        Args:
          @ path: the path to save model.
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """
        Load model.
        
        Args:
          @ path: the path to load model.
        """
        self.load_state_dict(torch.load(path, map_location = self.device))
