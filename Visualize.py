import matplotlib.pyplot as plt
import numpy as np


def visualize_preds(layer, batch_no, size1, size2, name):
    layer = layer[:, batch_no, :, :]
    shape1 = layer.shape[1]
    shape2 = layer.shape[2]

    img = np.zeros((shape1 * size1, shape2 * size2))
    k = 0
    for i in range(size1):
        for j in range(size2):
            img[i * shape1:(i + 1) * shape1, j * shape2:(j + 1) * shape2] = layer[k, :, :]
            k += 1

    plt.figure()
    plt.title(name)
    plt.imshow(1 - img, cmap='gray')


def visualize_vgg(layer, batch_no, size1, size2, name):
    layer = layer[batch_no, :, :, :]
    shape1 = layer.shape[1]
    shape2 = layer.shape[2]

    img = np.zeros((shape1 * size1, shape2 * size2))
    k = 0
    for i in range(size1):
        for j in range(size2):
            img[i * shape1:(i + 1) * shape1, j * shape2:(j + 1) * shape2] = layer[k, :, :]
            k += 1

    plt.figure()
    plt.title(name)
    plt.imshow(img)


