# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from HypVINN.config.hypvinn_files import HYPVINN_LUT

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def get_lut(lookup_table_path=HYPVINN_LUT):
    from collections import OrderedDict
    lut = OrderedDict()
    with open(lookup_table_path, 'r') as f:
        for line in f:
            if line[0] == '#' or line[0] == '\n':
                pass
            else:
                clean_line = remove_values_from_list(line.split(' '), '')
                rgb = [int(clean_line[2]), int(clean_line[3]), int(clean_line[4])]
                lut[str(clean_line[0])] = rgb
    return lut

def map_hyposeg2label(hyposeg,lut_file=HYPVINN_LUT):
    import matplotlib.colors
    '''
    Function to perform look-up table mapping of aseg.mgz data to label space (continue labels)
    '''
    labels=np.unique(hyposeg)

    labels = np.int16(labels)
    # retrieve freesurfer color map lookup table
    cdict = get_lut(lut_file)
    colors = np.zeros((len(labels), 3))
    #colors = list()
    mapped_hyposeg = np.zeros_like(hyposeg)

    for idx, value in enumerate(labels):
        mapped_hyposeg[hyposeg==value]=idx
        r, g, b = cdict[str(value)]
        colors[idx]=[r, g, b]

    colors = np.divide(colors ,255)
    cmap = colors
    cmap = matplotlib.colors.ListedColormap(colors)

    return mapped_hyposeg,cmap

def plot_coronal_predictions(cmap,images_batch=None,pred_batch=None,img_per_row=8):
    import matplotlib.pyplot as plt
    import torch
    from torchvision import  utils
    plt.ioff()

    FIGSIZE = 3
    #FIGDPI = dpi

    ncols=1
    nrows=2

    fig, ax = plt.subplots(nrows,ncols)

    grid_size=(images_batch.shape[0]/img_per_row,img_per_row)

    # adjust layout
    fig.set_size_inches([FIGSIZE * ncols * grid_size[1] , FIGSIZE * nrows * grid_size[0]])
    #fig.set_dpi(FIGDPI)
    fig.set_facecolor('black')
    fig.set_tight_layout({'pad': 0})
    fig.subplots_adjust(wspace=0,hspace=0)

    pos=0


    images = torch.from_numpy(images_batch.copy())
    images = torch.unsqueeze(images,1)
    grid = utils.make_grid(images.cpu(), nrow=img_per_row,normalize=True)
    #ax[pos].imshow(grid.numpy().transpose(1, 2, 0), cmap='gray',origin='lower')
    ax[pos].imshow(grid.numpy().transpose(1, 2, 0), cmap='gray', origin='lower')
    ax[pos].set_axis_off()
    ax[pos].set_aspect('equal')
    ax[pos].margins(0, 0)
    ax[pos].set_title('T1w input image (1 to N). Coronal orientation from right (R) to left (L).',color='white')
    pos += 1

    pred=torch.from_numpy(pred_batch.copy())
    pred = torch.unsqueeze(pred, 1)
    pred_grid = utils.make_grid(pred.cpu(), nrow=img_per_row)[0] #dont take the channels axis from grid
    #pred_grid=color.label2rgb(pred_grid.numpy(),grid.numpy().transpose(1, 2, 0),alpha=0.6,bg_label=0,colors=DEFAULT_COLORS)
    #pred_grid = color.label2rgb(pred_grid.numpy(), grid.numpy().transpose(1, 2, 0), alpha=0.6, bg_label=0,bg_color=None,colors=DEFAULT_COLORS)

    alphas = np.ones(pred_grid.numpy().shape) * 0.8
    alphas[pred_grid.numpy() == 0] = 0

    ax[pos].imshow(grid.numpy().transpose(1, 2, 0),cmap='gray', origin='lower')
    ax[pos].imshow(pred_grid.numpy(), cmap=cmap,interpolation='none',alpha=alphas,origin='lower')
    ax[pos].set_axis_off()
    ax[pos].set_aspect('equal')
    ax[pos].margins(0, 0)
    ax[pos].set_title('Predictions (1 to N). Coronal orientation from right (R) to left (L).',color='white')
    ax[pos].margins(0, 0)

    return fig


def  select_index_to_plot(hyposeg,slice_step=2):
    #slices with labels
    idx = np.where(hyposeg > 0)
    idx= np.unique(idx[0])
    #get slices with 3rd ventricle
    idx_with_third_ventricle = np.unique(np.where(hyposeg == 910)[0])
    #get slices with only 3rd ventricle
    idx_only_third_ventricle = []
    for i in idx_with_third_ventricle:
        label = np.unique(hyposeg[i])
        # Background is allways at position 0
        if label[1] == 910:
            idx_only_third_ventricle.append(i)
    #Remove slices with only third ventricle from the total
    idx = list(set(list(idx)) - set(idx_only_third_ventricle))
    #get slices with hyppthalamus variables
    idx_hypo = np.where(hyposeg > 920)
    idx_hypo = np.unique(idx_hypo[0])
    #remove hypo_varaibles from the list
    idx = list(set(list(idx)) - set(idx_hypo))
    # optic nerve index
    idx_with_optic_nerve = np.unique(np.where((hyposeg <= 902) & (hyposeg > 0))[0])
    #remove index from list
    idx = list(set(list(idx)) - set(idx_with_optic_nerve))
    #take optic nerve every 4 slices
    idx_with_optic_nerve = idx_with_optic_nerve[::4]
    # from the remaining slices only take increments by slice step default 2
    idx = idx[::slice_step]
    #Add slices with hypothalamus variables and optic nerve
    idx.extend(idx_hypo)
    idx.extend(idx_with_optic_nerve)

    return sorted(idx)


def plot_qc_images(save_dir,orig_path, prediction_path, padd=45, lut_file=HYPVINN_LUT, slice_step =2):
    from HypVINN.data_loader.data_utils import transform_axial2coronal
    from HypVINN.config.hypvinn_files import HYPVINN_QC_IMAGE_NAME

    from scipy import ndimage
    import os

    image = nib.as_closest_canonical(nib.load(orig_path))
    pred  = nib.as_closest_canonical(nib.load(prediction_path))

    mod_image = transform_axial2coronal(image.get_fdata())
    mod_image = np.transpose(mod_image,(2,0,1))
    mod_pred  = transform_axial2coronal(pred.get_fdata())
    mod_pred = np.transpose(mod_pred, (2, 0, 1))

    idx = select_index_to_plot(hyposeg=mod_pred, slice_step=slice_step)

    hypo_seg,cmap = map_hyposeg2label(hyposeg=mod_pred,lut_file=lut_file)

    if len(idx) > 0:

        crop_image = mod_image[idx, :, :]

        crop_seg = hypo_seg[idx, :, :]

        cm = ndimage.center_of_mass(crop_seg > 0)

        cm = np.array(cm).astype(int)

        crop_image = crop_image[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]
        crop_seg = crop_seg[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]

    else:
        depth = hypo_seg.shape[0] // 2
        crop_image = mod_image[depth - 8:depth + 8, :, :]
        crop_seg = hypo_seg[depth - 8:depth + 8, :, :]

        cm = [crop_image.shape[0]//2,crop_image.shape[1]//2,crop_image.shape[2]//2]
        cm = np.array(cm).astype(int)


        crop_image = crop_image[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]
        crop_seg = crop_seg[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]


    crop_image = np.rot90(np.flip(crop_image,axis=0),k=-1,axes=(1,2))
    crop_seg = np.rot90(np.flip(crop_seg, axis=0),k=-1,axes=(1,2))

    fig = plot_coronal_predictions(cmap=cmap,images_batch=crop_image,pred_batch=crop_seg, img_per_row=crop_image.shape[0])

    fig.savefig(os.path.join(save_dir,HYPVINN_QC_IMAGE_NAME), transparent=False)

    plt.close(fig)




