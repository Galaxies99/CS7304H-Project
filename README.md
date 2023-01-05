# CS7304H Final Project

**Author**: [Hongjie Fang](http://github.com/galaxies99/)

[[Report]](assets/report.pdf)

## Introduction

This is the codebase of the final project of "CS7304H Statistical Learning" course in SJTU.

## Preliminary

Our codebase relies on several Python packages, including: `numpy`, `tqdm`, `pytorch`, `sklearn`. Please install the packages before running our codes.

Our code is tested under MacOS Big Sur with CPU and Python 3.8 environment.

## Checkpoint and Results

Download checkpoints and results at [Baidu Netdisk](https://pan.baidu.com/s/1USGhKaoGsZtNmcxz--R1QA) (extraction code: 83c3). Choose `logs.zip`, download it, and then extract the zipped file into a folder `logs`, and put it in the root directory of our codebase.

Here are the correspondence of the results in Tab. 9 of the report.

- **KNN** (cosine, `k = 20`): `logs/knn/result_k_20_cosine.csv`;
- **SVM** (poly, `d = 3`, `C = 0.5`): `logs/svm/result_poly_d_3_C_0.5.csv`;
- **MLP** ((1536, 36, 20), wo. dropout): `logs/mlp/result_h_64_wo_drop_wo_reg.csv`, and its corresponding checkpoint is `logs/mlp/h_64_wo_drop_wo_reg_3.pth`.

## Data Processing

**Recommendation**. Please download all data that we used at [Baidu Netdisk](https://pan.baidu.com/s/1USGhKaoGsZtNmcxz--R1QA) (extraction code: 83c3). Choose `data.zip`, download it, and then extract the zipped file into a folder `data`, and put it in the root directory of our codebase.

**Process the Data by Yourselves**. You can also use the scripts in `process` to process the data by yourselves, but notice that the processing scripts might lead to different results in different device at different time. Therefore, the final result might be a little different. Here are the detailed explanations.

- `process/noise.py`: Generate noise for validation set;
- `process/holdout.py`: Generate the train-validation split for holdout method;
- `process/cross-validation.py`: Divide the data into several parts, which are used in cross-validation.

## Configuration

We provide all the configuration files we used in the experiments in `configs` folder. You may choose the configuration you want in inference or training process.

## Inference

For inference, use the following command:

```bash
python run.py (--data [Data Path])
              --model [Model Type] 
              --cfg [Configuration File] 
              --inference_only
              --ckpt [Checkpoint Path]
              --output [Output Path]
```

where

- `[Data Path]` (optional) is the path to the data, default is `data`;
- `[Model Type]` is one of `knn`, `svm`, `svm-ours` (our slow implementation, not recommended) and `mlp`;
- `[Configuration File]` is the path to the configuration file;
- `[Checkpoint Path]` is the path to the checkpoint;
- `[Output Path]` is the output file path of the inference process.

## Training (Optional)

For training (optional, since we provide trained checkpoints), use the following command:

```bash
python run.py (--data [Data Path])
              --model [Model Type] 
              --cfg [Configuration File] 
              --train_only
              --val [Model Selection Method]
              --split [Split File]
              (--ckpt [Checkpoint Path])
              (--val_with_noise)
              (--noise_path [Noise Path])
```

where

- `[Data Path]` (optional) is the path to the data, default is `data`;
- `[Model Type]` is one of `knn`, `svm`, `svm-ours` (our slow implementation, not recommended) and `mlp`;
- `[Configuration File]` is the path to the configuration file;
- `[Model Selection Method]` is one of `holdout` and `cross-validation`;
- `[Split File]` is the path to the split file of model selection methods, generated in data processing scripts, usually in `data` folder.
  - For holdout method with random sampling, use `data/holdout_random_split.npy`;
  - For holdout method with stratified sampling, use `data/holdout_stratified_split.npy`;
  - For cross validation method, use `data/cross_validation_split.npy`.
- `[Checkpoint Path]` (optional) is the path to the checkpoint, if not specified, then the training process won't save any checkpoints;
- The `--val_with_noise` option (optional) controls whether to add noise in the validation set;
- `[Noise Path]` (optional) specifies the path of the generated noise, usually in `data` folder.
  - For holdout method with random sampling, use `data/validation_manual_noise.npy`;
  - For holdout method with stratified sampling, use `data/validation_stratified_manual_noise.npy`;
  - For cross validation method, use `all_manual_noise.npy`.

## References

[1] E. Fix and J. L. Hodges, “Discriminatory analysis. nonparametric discrimination: Consistency properties,” International Statistical Review/Revue Internationale de Statistique, vol. 57, no. 3, pp. 238–247, 1989.

[2] T. Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE transactions on information theory, vol. 13, no. 1, pp. 21–27, 1967.

[3] C. Cortes and V. Vapnik, “Support-vector networks,” Machine learning, vol. 20, no. 3, pp. 273–297, 1995.

[4] H. Kuhn and A. Tucker, “Nonlinear programming in proceedings of 2nd berkeley symposium (pp. 481–492),” Berkeley: University of California Press, 1951.

[5] J. Mercer, “Functions ofpositive and negativetypeand theircommection with the theory of integral equations,” Philos. Trinsdictions Rogyal Soc, vol. 209, pp. 4–415, 1909.

[6] T. Hastie, R. Tibshirani, J. H. Friedman, and J. H. Friedman, The elements of statistical learning: data mining, inference, and prediction, vol. 2. Springer, 2009.

[7] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” nature, vol. 323, no. 6088, pp. 533–536, 1986.

[8] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The journal of machine learning research, vol. 15, no. 1, pp. 1929–1958, 2014.

[9] I. Loshchilov and F. Hutter, “Fixing weight decay regularization in adam,” 2018.