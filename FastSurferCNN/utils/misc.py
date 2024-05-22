# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
import os
from itertools import product
from typing import List

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import yacs.config
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import color
from torchvision import utils

import FastSurferCNN.data_loader.loader


def plot_predictions(
    images_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    batch_output: torch.Tensor,
    plt_title: str,
    file_save_name: str,
) -> None:
    """
    Plot predictions from validation set.

    Parameters
    ----------
    images_batch : torch.Tensor
        Batch of images.
    labels_batch : torch.Tensor
        Batch of labels.
    batch_output : torch.Tensor
        Batch of output.
    plt_title : str
        Plot title.
    file_save_name : str
        Name the plot should be saved tp.
    """
    f = plt.figure(figsize=(20, 10))
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    img_grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(211)
    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
    plt.imshow(color_grid, alpha=0.5)
    plt.title("Ground Truth")

    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(212)
    plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
    plt.imshow(color_grid, alpha=0.5)
    plt.title("Prediction")

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches="tight")
    plt.close(f)
    plt.gcf().clear()


def plot_confusion_matrix(
    cm: npt.NDArray,
    classes: List[str],
    title: str = "Confusion matrix",
    cmap: plt.cm.ColormapRegistry = plt.cm.Blues,
    file_save_name: str = "temp.pdf",
) -> matplotlib.figure.Figure:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    cm : npt.NDArray
        Confusion matrix.
    classes : List[str]
        List of class names.
    title : str
        (Default value = "Confusion matrix").
    cmap : plt.cm.ColormapRegistry
        Colour map (Default value = plt.cm.Blues).
    file_save_name : str
        (Default value = "temp.pdf").

    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib Figure object with the confusion matrix plot.
    """
    n_classes = len(classes)

    fig, ax = plt.subplots()
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    text_ = None
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im_, cax=cax)

    tick_marks = np.arange(n_classes)
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)

    values_format = ".2f"
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_[i, j] = ax.text(
            j, i, format(cm[i, j], values_format), ha="center", va="center", color=color
        )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")

    return fig


def find_latest_experiment(path: str) -> int:
    """
    Find and load latest experiment.

    Parameters
    ----------
    path : str
        Path to the latest experiment.

    Returns
    -------
    int
        Latest experiments.
    """
    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def check_path(path: str):
    """
    Create path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def update_num_steps(
    dataloader: FastSurferCNN.data_loader.loader.DataLoader, cfg: yacs.config.CfgNode
):
    """
    Update the number of steps.

    Parameters
    ----------
    dataloader : FastSurferCNN.data_loader.loader.DataLoader
        The dataloader object that contains the training data.
    cfg : yacs.config.CfgNode
        The configuration object that contains the training configuration.
    """
    cfg.TRAIN.NUM_STEPS = len(dataloader)
