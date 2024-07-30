
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
from typing import Union
import scipy

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision import utils

from FastSurferCNN.utils import logging

logger = logging.getLogger(__name__)

matplotlib.use('agg') # standard QT backend does not allow to be run in thread
#matplotlib.use('TkAgg') # standard QT backend does not allow to be run in thread


assert(os.environ['FASTSURFER_HOME'] is not None), 'Please set the environment variable FASTSURFER_HOME to the root of the CCNet repository!'
DEFAULT_LUT_PATH = os.path.join(os.environ['FASTSURFER_HOME'] ,'CCNet/config/FastSurfer_ColorLUT.tsv')


def calculate_centers_of_comissures(volume: Union[torch.Tensor, np.ndarray], axis: int = 3) -> np.ndarray :
    """
    Calculate the centers of the commisures from a probaility volume.
    
    Args:
        volume np.ndarray or torch.Tensor: 
            volume to calculate the centers from in logits or probability space.
        axis int:
            axis (counting from 1) for which the minimum coordinate value is considered to be the pc and the max the ac.
            If negative the max of coordinate value is considered to be the pc and the min the ac.
            Default: 2 (Anterior-axis in RAS space)
    
    Returns:
        tuple[int, int, int]: voxel space coordinates of the center of the anterior commisure
        tuple[int, int, int]: voxel space coordinates of the center of the posterior commisure
    
    """

    # convert to numpy
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().astype(np.float32)

    if np.sum(volume) == 0:
        raise ValueError('Volume is empty!')

    if np.min(volume) < 0:
        # map logits space to probability space
        volume = torch.nn.functional.sigmoid(torch.from_numpy(volume)).detach().cpu().numpy()

    # put gaußian filter over volume
    volume = scipy.ndimage.gaussian_filter(volume, sigma=5, mode='constant', cval=0.0)

    # calculate two centers

    # Apply maximum filter
    size = [8 for _ in range(len(volume.shape))]
    local_max = scipy.ndimage.maximum_filter(volume, size=size) == volume

    # Filter out the 0s
    local_max = np.logical_and(local_max, volume != 0)

    # Flatten the volume and get the indices of the maxima
    maxima_indices = np.argwhere(local_max)

    # Get the values at these indices
    maxima_values = volume[local_max]

    # Sort the indices by the values
    sorted_indices = np.argsort(maxima_values)[::-1]
    

    # Get the coordinates of the 5 strongest maxima
    strongest_maxima = maxima_indices[sorted_indices[:100]]
    strongest_maxima_values = maxima_values[sorted_indices[:100]]

    # Calculate the pairwise distance between all maxima
    distances = scipy.spatial.distance.cdist(strongest_maxima, strongest_maxima)

    # Create a mask of the maxima to remove
    mask = np.ones(len(strongest_maxima), dtype=bool)

    for i in range(len(strongest_maxima)):
        if mask[i] == 0:
            continue

        # Get the distances to the other maxima
        dist_to_others = distances[i]

        # Find the maxima within a distance of 5 voxels
        close_maxima = np.where((dist_to_others < 10) & (dist_to_others > 0))[0]

        # Mark the close maxima for removal
        mask[close_maxima] = False

    # Apply the mask to remove the close maxima
    strongest_maxima = strongest_maxima[mask][:2]
    strongest_maxima_values = strongest_maxima_values[mask][:2]

    # differentiate between AC and PC
    
    # PC is the one in posterior direction
    if axis < 0:
        PC_idx = int(np.argmax(strongest_maxima[:, (-axis)-1]))
    else:
        PC_idx = int(np.argmin(strongest_maxima[:, axis-1]))

    # AC is the one in anterior direction
    AC_idx = 1 - PC_idx

    if len(strongest_maxima) == 2:
        return np.array([strongest_maxima[AC_idx], strongest_maxima[PC_idx]], dtype=np.int32)
    else:
        raise ValueError('Could not find two maxima!')

    


def get_lut(lookup_table: str = DEFAULT_LUT_PATH):
    # Load lookup table
    # try: # to map with original fastsurfer lookup table
    lut_df = pd.read_csv(lookup_table, sep='\t')#.set_index('ID')
    lut_df = lut_df[['R','G','B']]

    np_lut = np.full((lut_df.index.max() +1, 3), -1, dtype=int)

    for i in lut_df.index:
        np_lut[i, :] = lut_df.iloc[i, :]
    
        # except:
    #     colors = None
    #     color_grid = color.label2rgb(label=grid.numpy(), image=img_grid, colors=colors, bg_label=0)

    return np_lut


def save_imgage_label_plot(img_grid: np.ndarray, label_grid: np.ndarray, net_output_grid: np.ndarray, plt_title: str, file_save_name: str, lut_name: str = DEFAULT_LUT_PATH):
    np_lut = get_lut(lookup_table=lut_name)

    logger.debug(f'Saving image: {file_save_name}')
    f = plt.figure(figsize=(20, 10))
    plt.subplot(211)

    logger.debug(f'Creating GT')
    logger.debug(f'Unique labels: {np.unique(label_grid)}')

    color_grid = np_lut[label_grid]
    if not (color_grid >= 0).all():
        logger.warn('Some labels are not in the lookup table!')
        color_grid[color_grid < 0] = 0

    plt.imshow(img_grid)
    plt.imshow(color_grid, alpha=0.5)

    plt.title('Ground Truth')

    plt.subplot(212)
    #color_grid = color.label2rgb(grid.numpy(), bg_label=0)

    logger.debug(f'Creating predictions')
    logger.debug(f'Unique labels: {np.unique(net_output_grid)}')

    color_grid = np_lut[net_output_grid]

    if not (color_grid >= 0).all():
        logger.warn('Some labels are not in the lookup table!')
        color_grid[color_grid < 0] = 0

    plt.imshow(img_grid)
    plt.imshow(color_grid, alpha=0.5)
    plt.title('Prediction')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight')
    #plt.close(f)
    #plt.gcf().clear()
    logger.debug(f'Successfully created {file_save_name}')
    return 0


def plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name, process_executor=None, lut = DEFAULT_LUT_PATH): #, lookup_table='./config/FastSurfer_ColorLUT.tsv'):
    """
    Function to plot predictions from validation set.
    :param images_batch: input images
    :param labels_batch: ground truth
    :param batch_output: output from the network
    :param plt_title: title of the plot
    :param file_save_name: path to save the plot
    :param lookup_table: path to lookup table for colors
    :return:
    """
    logger.debug('Plotting predictions')
    
    c = images_batch.shape[1] # slice dimension
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)

    logger.debug('Creating grid')
    img_grid = utils.make_grid(images_batch, nrow=4).cpu().numpy().transpose((1, 2, 0))
    label_grid = utils.make_grid(labels_batch.unsqueeze(1), nrow=4)[0].cpu().numpy()
    net_output_grid = utils.make_grid(batch_output.unsqueeze(1), nrow=4)[0].cpu().numpy()
    logger.debug('Done creating grid')

    #with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: # do IO in a seperate process
    if process_executor is not None:
        logger.debug(f'Saving image in process executor')
        #_ = process_executor.submit(save_imgage_label_plot, img_grid, label_grid, net_output_grid, plt_title, file_save_name)
        _ = process_executor.apply_async(save_imgage_label_plot, (img_grid, label_grid, net_output_grid, plt_title, file_save_name, lut))
    else:
        save_imgage_label_plot(img_grid, label_grid, net_output_grid, plt_title, file_save_name, lut)

        # # get the result from the task
        # exception = plot1_future.exception()
        # # handle exceptional case
        # if exception:
        #     print(exception)
        # else:
        #     result = plot1_future.result()
        #     print(result)


def save_inpainting_plot(img_grid: np.ndarray, label_grid: np.ndarray, net_output_grid: np.ndarray, plt_title: str, file_save_name: str, dpi=100):
    f = plt.figure(figsize=(20, 20))

    plt.subplot(511)
    plt.imshow(label_grid, vmin=0, vmax=1)
    plt.title('Ground Truth')

    plt.subplot(512)
    plt.imshow(net_output_grid, vmin=0, vmax=1)
    plt.title('Prediction')

    plt.subplot(513)
    plt.imshow(img_grid, vmin=0, vmax=1)
    plt.title('Input')

    plt.subplot(514)
    plt.imshow(np.abs(label_grid - net_output_grid), vmin=0, vmax=1)
    plt.title('Ground truth - Prediction')

    plt.subplot(515)
    plt.imshow(np.abs(img_grid - label_grid), vmin=0, vmax=1)
    plt.title('Input - Ground truth')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight', dpi=dpi)
    plt.close(f)
    plt.gcf().clear()

    return 0

def plot_predictions_inpainting(images_batch, labels_batch, batch_output, plt_title, file_save_name, process_executor=None, dpi=100):
    """
    Function to plot predictions from validation set.
    :param images_batch:
    :param labels_batch:
    :param batch_output:
    :param plt_title:
    :param file_save_name:
    :return:
    """
    
    c = images_batch.shape[1] # slice dimension
    mid_slice = c // 2
    img_grid = utils.make_grid(images_batch[:, mid_slice, :, :].unsqueeze(1), nrow=4).cpu().numpy().transpose((1, 2, 0))
    label_grid = utils.make_grid(labels_batch.unsqueeze(1), nrow=4).cpu().numpy().transpose((1, 2, 0))
    output_grid = utils.make_grid(batch_output.unsqueeze(1), nrow=4).cpu().numpy().transpose((1, 2, 0))

    #with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: # do IO in a seperate process

    if process_executor is not None:
        #_ = process_executor.submit(save_inpainting_plot, img_grid, label_grid, output_grid, plt_title, file_save_name)
        _ = process_executor.apply_async(save_inpainting_plot, (img_grid, label_grid, output_grid, plt_title, file_save_name))
    else:
        save_inpainting_plot(img_grid, label_grid, output_grid, plt_title, file_save_name, dpi=dpi)

        # # get the result from the task
        # exception = plot2_future.exception()
        # # handle exceptional case
        # if exception:
        #     print(exception)
        # else:
        #     result = plot2_future.result()
        #     print(result)




def plot_confusion_matrix(cm,
                          classes,
                          file_save_name=None,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ):
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(n_classes, n_classes))
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im_, cax=cax)

    tick_marks = np.arange(n_classes)
    ax.set(xticks=tick_marks,
           yticks=tick_marks,
           xticklabels=classes,
           yticklabels=classes,
           ylabel="True label",
           xlabel="Predicted label")

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)

    values_format = '.2f'
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_[i, j] = ax.text(j, i,
                              format(cm[i, j], values_format),
                              ha="center", va="center",
                              color=color)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='horizontal')

    if file_save_name is not None:
        plt.savefig(file_save_name)

    return fig


def find_latest_experiment(path):
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


def find_device(device: str = "auto", flag_name:str = "device") -> torch.device:
    """Create a device object from the device string passed, including detection of devices if device is not defined
    or "auto".
    """
    logger = logging.get_logger(__name__ + ".auto_device")
    # if specific device is requested, check and stop if not available:
    if device.split(':')[0] == "cuda" and not torch.cuda.is_available():
        logger.info(f"cuda not available, try switching to cpu: --{flag_name} cpu")
        raise ValueError(f"--device cuda not available, try --{flag_name} cpu !")
    if device == "mps" and not torch.backends.mps.is_available():
        logger.info(f"mps not available, try switching to cpu: --{flag_name} cpu")
        raise ValueError(f"--device mps not available, try --{flag_name} cpu !")
    # If auto detect:
    if device == "auto" or not device:
        # 1st check cuda
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    # Define device and transfer model
    logger.info(f"Using {flag_name}: {device}")
    return torch.device(device)
