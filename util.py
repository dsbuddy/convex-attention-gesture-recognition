import codecs
import numpy as np


def load_text(path):
    try:
        with codecs.open(path, encoding='utf-8') as f:
            return f.read()
    except IOError:
        return None


def save_text(path, text):
    with open(path, 'w') as f:
        f.write(text)


def one_hot(i, n):
    a = np.zeros(n)
    a[i] = 1.0
    return a


def moving_average(a, kernel_size):
    _, n_features = a.shape
    kernel = np.ones(kernel_size) / kernel_size
    for i in range(n_features):
        a[:, i] = np.convolve(a[:, i], kernel, mode='same')
    ks_2 = int((kernel_size - 1) / 2)
    a = a[ks_2:-ks_2, :]
    return a
