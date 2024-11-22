from pathlib import Path

import numpy as np
import pandas as pd
import os

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class Blog(BaseDataset):

    def __init__(self, args):
        super(Blog, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "blog"
        self.tmp_file_names = ["blog.csv"]
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):

        path = os.path.join(self.args.data_path, self.tmp_file_names[0])
        data = pd.read_csv(path, header=None)

        self.X = data.drop(columns=[280]).to_numpy()
        target = data[280] > 0
        target = target.astype(int)
        self.y = target.to_numpy()

        self.N, self.D = self.X.shape

        self.cardinalities = []
        self.num_features = list(range(self.D))
        self.cat_features = []

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}
