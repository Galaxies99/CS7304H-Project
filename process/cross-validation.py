import numpy as np


N_samples = 6666
N_classes = 20
K = 5

labels = np.load('data/train_labels.npy')
classes = []
for i in range(N_classes):
    classes.append([])
for i, label in enumerate(labels.tolist()):
    classes[label].append(i)
parts = []
for i in range(K):
    parts.append([])
for c in range(N_classes):
    N_class_samples = len(classes[c])
    class_parts = np.split(np.random.permutation(N_class_samples), [i * int(N_class_samples / K) for i in range(1, K)])
    class_index = np.array(classes[c])
    for i in range(K):
        parts[i].append(class_index[class_parts[i]])
cross_validation_split = {}
for i in range(K):
    parts[i] = np.hstack(parts[i])
    N_part_samples = len(parts[i])
    parts[i] = parts[i][np.random.permutation(N_part_samples)]
    cross_validation_split['{}'.format(i)] = parts[i]
np.save('data/cross_validation_split.npy', cross_validation_split)
