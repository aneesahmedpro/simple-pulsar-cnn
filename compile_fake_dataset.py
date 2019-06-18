import os

import numpy as np
# import matplotlib.pyplot as plt

from local_settings import TRAINING_DATA_DIR


# 500 images, each 64x128 (HxW), pixels only 20% bright
# x = np.random.uniform(size=(500, 64, 128)) * 0.2
x = np.random.uniform(size=(500, 64, 128)) * 0.5

# All 500 images are "non-pulsar" right now
y = np.full(500, fill_value=0, dtype=np.int)

# Make first 250 images "pulsar" and change corresponding labels
for img in x[:250]:
    # img[:, [62, 65]] *= 2.5  # pixels 50% bright
    img[:, [62, 65]] *= 1.5
    # img[:, [63, 64]] *= 5.0  # pixels 100% bright
    img[:, [63, 64]] *= 2
y[:250] = 1

# Displace the positions of spikes randomly
displacements = np.random.randint(low=0, high=128, size=250)
for i in range(250):
    x[i] = np.roll(x[i], displacements[i], axis=1)

# plt.imshow(x, cmap='gray_r'); plt.show(); plt.close()
# plt.imshow(x); plt.show(); plt.close()

# Select 80% for training and 20% for testing
indices = np.arange(500, dtype=np.int)
train_indices = np.concatenate((indices[:200], indices[250:450]))
test_indices = np.concatenate((indices[200:250], indices[450:500]))
x_train = x[train_indices]
y_train = y[train_indices]
x_test = x[test_indices]
y_test = y[test_indices]

os.chdir(TRAINING_DATA_DIR/'npy')
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
