from scipy.io import arff
import os
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE
from keras.datasets import mnist
import numpy as np


## Code taken from VIME github repository
def load_mnist_data():
    """MNIST data loading.

    Args:
      - label_data_rate: ratio of labeled data

    Returns:
      - x_label, y_label: labeled dataset
      - x_unlab: unlabeled dataset
      - x_test, y_test: test dataset
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # x = x.reshape(x.shape[0], -1)

    y = np.argmax(y, axis=1)

    return x, y


class MNIST(BaseDataset):
    def __init__(self, args):
        super(MNIST, self).__init__(args)
        self.is_data_loaded = False
        self.name = "mnist"
        self.args = args
        self.task_type = TASK_TYPE.MULTI_CLASS

    def load(self):

        x, y = load_mnist_data()
        x = x.reshape(x.shape[0], -1)
        self.X = x
        self.y = y

        self.N, self.D = self.X.shape

        self.cardinalities = []
        self.num_or_cat = {}

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}
        self.cls = 0

        self.is_data_loaded = True
