from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.nn.init as nn_init
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import json
from joblib import Parallel, delayed
import pandas as pd
from einops import rearrange, repeat
from sklearn.decomposition import PCA
import Models
from src.models.autoint import AutoIntBase
from src.models.dcnv2 import DCNv2, DCNv2Base
from rtdl_revisiting_models import ResNet, FTTransformer


class Model(nn.Module):
    def __init__(
        self,
        input_num,
        model_type,
        out_dim,
        info,
        topic_num,
        cluster_centers_,
        config,
        task_type,
        categories,
    ) -> None:
        super().__init__()

        self.input_num = input_num  ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        self.num_list = np.arange(info.get("n_num_features"))
        self.cat_list = (
            np.arange(
                info.get("n_num_features"),
                info.get("n_num_features") + info.get("n_cat_features"),
            )
            if info.get("n_cat_features") != None
            else None
        )
        self.topic_num = topic_num
        self.cluster_centers_ = cluster_centers_
        self.categories = categories

        self.config = config
        self.task_type = task_type

        self.build_model()

    def build_model(self):

        if self.model_type.split("_")[0] == "MLP":
            # construct parameter for centers
            self.topic = nn.Parameter(
                torch.tensor(self.cluster_centers_), requires_grad=True
            )

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            self.encoder = Models.mlp.MLP(
                self.input_num,
                self.config["model"]["d_layers"],
                self.config["model"]["dropout"],
                self.out_dim,
                self.categories,
                self.config["model"]["d_embedding"],
            )

            self.head = nn.Linear(self.config["model"]["d_layers"][-1], self.out_dim)

            self.reduce = nn.Sequential(
                nn.Linear(
                    self.config["model"]["d_layers"][-1],
                    self.config["model"]["d_layers"][-1],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_layers"][-1],
                    self.config["model"]["d_layers"][-1],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_layers"][-1],
                    self.config["model"]["d_layers"][-1],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config["model"]["d_layers"][-1], self.topic_num),
            )
        elif self.model_type.split("_")[0] == "DCNv2":
            self.topic = nn.Parameter(
                torch.tensor(self.cluster_centers_), requires_grad=True
            )

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            self.encoder = DCNv2Base(
                d_in=self.config["model"]["d_in"],
                d=self.config["model"]["d"],
                n_hidden_layers=self.config["model"]["n_hidden_layers"],
                n_cross_layers=self.config["model"]["n_cross_layers"],
                hidden_dropout=self.config["model"]["hidden_dropout"],
                cross_dropout=self.config["model"]["cross_dropout"],
                d_out=self.config["model"]["d_out"],
                stacked=self.config["model"]["stacked"],
                categories=self.config["model"].get("categories", None),
                d_embedding=self.config["model"]["d_embedding"],
            )

            self.head = nn.Linear(self.config["model"]["d"], self.out_dim)

            self.reduce = nn.Sequential(
                nn.Linear(
                    self.config["model"]["d"],
                    self.config["model"]["d"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d"],
                    self.config["model"]["d"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d"],
                    self.config["model"]["d"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config["model"]["d"], self.topic_num),
            )
        elif self.model_type.split("_")[0] == "Resnet":
            self.topic = nn.Parameter(
                torch.tensor(self.cluster_centers_), requires_grad=True
            )

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            class PreprocessInput(nn.Module):
                def __init__(self):
                    super(PreprocessInput, self).__init__()

                def forward(self, x_num, x_cat):
                    return x_num

            class EncoderWrapper(nn.Module):
                def __init__(self, config):
                    super(EncoderWrapper, self).__init__()
                    self.preprocess = PreprocessInput()
                    self.encoder = ResNet(
                        d_in=config["model"]["d_in"],
                        d_out=config["model"].get("d_out", None),
                        n_blocks=config["model"]["n_blocks"],
                        d_block=config["model"]["d_block"],
                        d_hidden=config["model"].get("d_hidden", None),
                        d_hidden_multiplier=config["model"].get(
                            "d_hidden_multiplier", None
                        ),
                        dropout1=config["model"]["dropout1"],
                        dropout2=config["model"]["dropout2"],
                    )

                def forward(self, x_num, x_cat):
                    return self.encoder(x_num)

            self.encoder = EncoderWrapper(self.config)

            self.head = nn.Linear(self.config["model"]["d_hidden"], self.out_dim)

            self.reduce = nn.Sequential(
                nn.Linear(
                    self.config["model"]["d_hidden"],
                    self.config["model"]["d_hidden"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_hidden"],
                    self.config["model"]["d_hidden"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_hidden"],
                    self.config["model"]["d_hidden"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config["model"]["d_hidden"], self.topic_num),
            )
        elif self.model_type.split("_")[0] == "AutoInt":
            self.topic = nn.Parameter(
                torch.tensor(self.cluster_centers_), requires_grad=True
            )

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            self.encoder = AutoIntBase(
                d_numerical=self.config["model"]["d_numerical"],
                categories=self.config["model"].get(
                    "categories", None
                ),  # Optional parameter
                n_layers=self.config["model"]["n_layers"],
                d_token=self.config["model"]["d_token"],
                n_heads=self.config["model"]["n_heads"],
                attention_dropout=self.config["model"]["attention_dropout"],
                residual_dropout=self.config["model"]["residual_dropout"],
                activation=self.config["model"]["activation"],
                prenormalization=self.config["model"]["prenormalization"],
                initialization=self.config["model"]["initialization"],
                kv_compression=self.config["model"].get(
                    "kv_compression", None
                ),  # Optional parameter
                kv_compression_sharing=self.config["model"].get(
                    "kv_compression_sharing"
                ),  # Optional parameter
                d_out=self.config["model"]["d_out"],
            )

            self.head = nn.Linear(self.config["model"]["d_token"], self.out_dim)

            self.reduce = nn.Sequential(
                nn.Linear(
                    self.config["model"]["d_token"],
                    self.config["model"]["d_token"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_token"],
                    self.config["model"]["d_token"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_token"],
                    self.config["model"]["d_token"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config["model"]["d_token"], self.topic_num),
            )
        elif self.model_type.split("_")[0] == "fttransformer":
            self.topic = nn.Parameter(
                torch.tensor(self.cluster_centers_), requires_grad=True
            )

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            class EncoderWrapper(nn.Module):
                def __init__(self, config):
                    super(EncoderWrapper, self).__init__()
                    self.config = config
                    self.encoder = FTTransformer(
                        n_cont_features=self.config["model"].get("n_cont_features", 0),
                        cat_cardinalities=self.config["model"].get(
                            "cat_cardinalities", None
                        ),
                        d_out=self.config["model"].get("d_out", 128),
                        n_blocks=self.config["model"].get("n_blocks", 6),
                        d_block=self.config["model"].get("d_block", 64),
                        attention_n_heads=self.config["model"].get(
                            "attention_n_heads", 4
                        ),
                        attention_dropout=self.config["model"].get(
                            "attention_dropout", 0.1
                        ),
                        ffn_d_hidden=self.config["model"].get("ffn_d_hidden", None),
                        ffn_d_hidden_multiplier=self.config["model"].get(
                            "ffn_d_hidden_multiplier", 2
                        ),
                        ffn_dropout=self.config["model"].get("ffn_dropout", 0.1),
                        residual_dropout=self.config["model"].get(
                            "residual_dropout", 0.1
                        ),
                    )

                def forward(self, x_num, x_cat):
                    if not hasattr(self.config["model"], "cat_cardinalities"):
                        x_cat = None

                    return self.encoder(x_num, x_cat)

            self.encoder = EncoderWrapper(self.config)

            self.head = nn.Linear(self.config["model"]["d_block"], self.out_dim)

            self.reduce = nn.Sequential(
                nn.Linear(
                    self.config["model"]["d_block"],
                    self.config["model"]["d_block"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_block"],
                    self.config["model"]["d_block"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(
                    self.config["model"]["d_block"],
                    self.config["model"]["d_block"],
                ),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config["model"]["d_block"], self.topic_num),
            )

    def forward(self, inputs_n, inputs_c):
        inputs_ = self.encoder(inputs_n, inputs_c)
        r_ = self.reduce(inputs_)
        if self.model_type.split("_")[1] == "ot":
            return (
                self.head(inputs_),
                torch.softmax(r_, dim=1),
                inputs_,
                torch.sigmoid(self.weight_) + 0.01,
            )
        else:
            return self.head(inputs_)
