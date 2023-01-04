# Holdout script with random sampling.
import numpy as np


np.random.seed(2333)

N_samples = 6666
N_classes = 20
p = 0.1

training_set_random, validation_set_random = np.split(np.random.permutation(N_samples), [int(N_samples * (1 - p))])
random_split = {
    'train': training_set_random,
    'val': validation_set_random
}
np.save('data/holdout_random_split.npy', random_split)


labels = np.load('data/train_labels.npy')
classes = []
for i in range(N_classes):
    classes.append([])
for i, label in enumerate(labels.tolist()):
    classes[label].append(i)
training_set_stratified = []
validation_set_stratified = []
for c in range(N_classes):
    N_class_samples = len(classes[c])
    training_set_c, validation_set_c = np.split(np.random.permutation(N_class_samples), [int(N_class_samples * (1 - p))])
    class_index = np.array(classes[c])
    training_set_stratified.append(class_index[training_set_c])
    validation_set_stratified.append(class_index[validation_set_c])
training_set_stratified = np.hstack(training_set_stratified)
validation_set_stratified = np.hstack(validation_set_stratified)
N_training_samples = len(training_set_stratified)
N_validation_samples = len(validation_set_stratified)
training_set_stratified = training_set_stratified[np.random.permutation(N_training_samples)]
validation_set_stratified = validation_set_stratified[np.random.permutation(N_validation_samples)]
stratified_split = {
    'train': training_set_stratified,
    'val': validation_set_stratified
}
np.save('data/holdout_stratified_split.npy', stratified_split)
