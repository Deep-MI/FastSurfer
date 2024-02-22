# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import glob
import math
from itertools import product

import torch
from torchvision import utils
import numpy as np
from skimage import color
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from FastSurferCNN.utils import logging

logger = logging.get_logger(__name__)


def plot_predictions(images_batch, labels_batch, batch_output, plt_title):
    """
    Function to plot predictions from validation set.
    :param images_batch:
    :param labels_batch:
    :param batch_output:
    :param plt_title:
    :param file_save_name:
    :return:
    """

    n, c, h, w = images_batch.shape
    f_size = min(n * 4, 40)
    f = plt.figure(figsize=(f_size, f_size))
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(131)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title("Slices")

    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(132)
    plt.imshow(color_grid)
    plt.title("Ground Truth")

    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(133)
    plt.imshow(color_grid)
    plt.title("Prediction")

    plt.suptitle(plt_title)
    plt.tight_layout()

    return f


def plot_confusion_matrix(
    cm,
    classes,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    figsize=(20, 20),
    file_save_name=None,
):
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
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

    if file_save_name is not None:
        plt.savefig(file_save_name)

    return fig


def get_score_class_dict(subject_id, scores, metric_name, class_names):
    score_dict = {"Subject": subject_id}
    if isinstance(scores, list):
        scores = np.array(scores)

    for idx in range(1, len(class_names)):
        score_dict[class_names[idx]] = scores[idx]

    score_dict[f"Mean_{metric_name}"] = scores[1:].mean()
    return score_dict


def find_latest_experiment(path):
    if not os.path.exists(path):
        return 0
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


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def update_num_steps(dataloader, cfg):
    cfg.TRAIN.NUM_STEPS = len(dataloader)


def check_nan_loss(loss, epoch):

    if math.isnan(loss):
        raise RuntimeError(f"Got NaN loss at epoch {epoch}")


def get_selected_class_ids(num_classes, ignored_classes=None):
    new_lbl_idx = np.arange(num_classes)
    if ignored_classes is not None:
        new_lbl_idx = np.delete(new_lbl_idx, ignored_classes)
    return new_lbl_idx


def set_summary_path(cfg):
    """
    Set last experiment number(EXPR_NUM) and updates the summary path accordingly.
    
    Parameters
    ----------
    cfg : [MISSING TYPE]
        [MISSING].
    """
    summary_path = check_path(os.path.join(cfg.LOG_DIR, "summary"))
    cfg.EXPR_NUM = str(find_latest_experiment(os.path.join(cfg.LOG_DIR, "summary")) + 1)
    if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
        cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
    cfg.SUMMARY_PATH = check_path(os.path.join(summary_path, "{}".format(cfg.EXPR_NUM)))


def load_classwise_weights(cfg):
    """
    Loading class-wise median frequency weights.
    """
    dataset_dir = os.path.dirname(cfg.DATA.PATH_HDF5_TRAIN)
    weight_path = glob.glob(os.path.join(dataset_dir, "*.npy"))
    if len(weight_path) > 0:
        class_weights = np.load(weight_path[0])
        return torch.from_numpy(class_weights).unsqueeze(1)
    else:
        logger.warn(f"No class-wise weight found at {dataset_dir}. Returning ones.")
        return torch.ones(cfg.MODEL.NUM_CLASSES, 1)


def update_results_dir(cfg):
    """
    It will update the results path by finding the last experiment number.

    Parameters
    ----------
    cfg : [MISSING TYPE]
        [MISSING].
    """
    cfg.EXPR_NUM = str(find_latest_experiment(cfg.TEST.RESULTS_DIR) + 1)
    cfg.TEST.RESULTS_DIR = check_path(
        os.path.join(cfg.TEST.RESULTS_DIR, "{}".format(cfg.EXPR_NUM))
    )


def update_split_path(cfg):
    """
    Updating path with respect to the split number
    
    Parameters
    ----------
    cfg : [MISSING TYPE]
        [MISSING].
    """
    from os.path import split, join

    split_num = cfg.SPLIT_NUM
    keys = [
        "RESULTS_DIR",
        "AXIAL_CHECKPOINT_PATH",
        "CORONAL_CHECKPOINT_PATH",
        "SAGITTAL_CHECKPOINT_PATH",
    ]

    for k in keys:
        cfg[k] = cfg[k].replace("split", f"split{split_num}")

    path, name = split(cfg["SUBJECT_CSV_PATH"])
    cfg["SUBJECT_CSV_PATH"] = join(path, name.replace("split", f"split{split_num}"))


def visualize_batch(img, label, idx):
    """
    For deubg
    :param batch_dict:
    :return:
    """
    from skimage import color
    import matplotlib.pyplot as plt

    plt.imshow(img[idx, 3].cpu().numpy(), cmap="gray")
    plt.imshow(color.label2rgb(label[idx].cpu().numpy(), bg_label=0), alpha=0.4)


# def visualize_np_data(img, label):
#     np.argmax(np.sum(data_dict['cereb_subseg'], axis=(0, 1)))
