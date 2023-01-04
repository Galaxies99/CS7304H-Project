import os
import yaml
import argparse
import numpy as np
from dataset.splitter import Splitter
from utils import save_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = None, help = 'model name')
    parser.add_argument('--data', type = str, default = 'data', help = 'data path')
    parser.add_argument('--cfg', type = str, default = None, help = 'path to configuration file')
    parser.add_argument('--val', type = str, default = 'holdout', help = 'type of validation')
    parser.add_argument('--split', type = str, default = None, help = 'path to the validation split file.')
    parser.add_argument('--train_only', action = 'store_true', help = 'training (and validation) only')
    parser.add_argument('--inference_only', action = 'store_true', help = 'inference only')
    parser.add_argument('--ckpt', type = str, default = None, help = 'path to save/load the checkpoint')
    parser.add_argument('--output', type = str, default = None, help = 'output path')
    parser.add_argument('--val_with_noise', action = 'store_true', help = 'whether to add noise to validation set.')
    parser.add_argument('--noise_path', type = str, default = None, help = 'noise path')
    args = parser.parse_args()

    if args.train_only and args.inference_only:
        raise AttributeError('train_only and inference_only options cannot be used at the same time.')
    if not args.inference_only and args.split is None:
        raise AttributeError('Please provide the validation split file.')
    if not args.inference_only and args.val_with_noise and args.noise_path is None:
        raise AttributeError('Please provide the noise path when using val_with_noise option.')

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfg_file:
            cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)
    else:
        cfgs = {}

    if args.model.lower() == 'knn':
        from models.knn import KNN
        model = KNN(**cfgs)
    elif args.model.lower() == 'svm-sklearn' or args.model.lower() == 'svm':
        from models.svm import SVC
        model = SVC(**cfgs)
    elif args.model.lower() == 'svm-ours':
        print('[Warning] You are selecting the SVM implemented by ourselves. This can be VERY slow (several hours). Consider selecting SVM implemented by sklearn.')
        from models.svm import SVM
        model = SVM(**cfgs)
    else:
        raise AttributeError('Unimplemented Model.')
    
    if args.inference_only:
        if args.ckpt is None:
            raise AttributeError('Please provide checkpoint path when using inference_only option.')
        model.load(args.ckpt)
    else:
        splitter = Splitter(
            training_features = os.path.join(args.data, 'train_features.npy'), 
            training_labels = os.path.join(args.data, 'train_labels.npy'), 
            val = args.val, 
            split_file = args.split, 
            val_with_noise = args.val_with_noise, 
            noise_file = args.noise_path
        )
        if args.val == 'holdout':
            X_train, y_train, X_val, y_val = splitter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = (y_pred == y_val).sum() / len(y_val)
            print('Validation Accuracy = {} %'.format(acc * 100))
        elif args.val == 'cross-validation':
            acc = 0.0
            for k in range(splitter.get_K()):
                X_train, y_train, X_val, y_val = splitter(index = k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc_part = (y_pred == y_val).sum() / len(y_val)
                acc += acc_part
            acc = acc / splitter.get_K()
            print('Validation Accuracy = {} %'.format(acc * 100))
    
    if args.train_only:
        if args.ckpt is not None:
            model.save(args.ckpt)
    else:
        if args.ckpt is not None:
            model.save(args.ckpt)
        if args.output is None:
            raise AttributeError('Please provide the output path when inference.')
        X = np.load(os.path.join('data', 'test_features.npy'))
        y = model.predict(X)
        save_prediction(args.output, y)
