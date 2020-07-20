
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
import optparse
import nibabel as nib
import numpy as np
import time
import sys
import glob
import os.path as op
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import label, regionprops
from skimage.measure import label

from collections import OrderedDict
from os import makedirs

from data_loader.load_neuroimaging_data import OrigDataThickSlices
from data_loader.load_neuroimaging_data import map_label2aparc_aseg
from data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from data_loader.load_neuroimaging_data import get_largest_cc
from data_loader.load_neuroimaging_data import load_and_conform_image

from data_loader.augmentation import ToTensorTest

from models.networks import FastSurferCNN


HELPTEXT = """
Script to generate aparc.DKTatlas+aseg.deep.mgz using Deep Learning 

USAGE:
python3 eval.py --i_dir <data_directory>  or --csv_file <subject_dirs_list.csv>\
                --in_name <volume name> \
                --t <search_tag> \
                --o_dir <output_directory> \
                --out_name <prediction name> \
                --log <log_file> \
                --network_sagittal_path <pretrained weights sagittal> \
                --network_coronal_path <pretrained weights coronal> \
                --network_axial_path <pretrained weights sagittal>

Dependencies:

    Torch
    Torchvision
    Skimage
    Numpy
    Matplotlib
    optparse
    time
    glob
    h5py
    sys
    os
    scipy
    Python 3.5

    Numpy
    http://www.numpy.org

    Nibabel to read and write neuroimaging data
    http://nipy.org/nibabel/


Original Author: Leonie Henschel
Date: Mar-12-2019

"""


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: fast_surfer_cnn, v 1.0 2019/09/30$',
                                   usage=HELPTEXT)
    # Requiered options
    # 1. Directory information (where to read from, where to write to)
    parser.add_option('--i_dir', '--input_directory', dest='input', help='path to directory of input volume(s).')
    parser.add_option('--csv_file', '--csv_file', help="CSV-file with directories to process", default=None)
    parser.add_option('--o_dir', '--output_directory', dest='output',
                      help='path to output directory. Will be created if it does not already exist')

    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_option('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
                      default='orig.mgz')
    parser.add_option('--out_name', '--output_name', dest='oname', default='aparc.DKTatlas+aseg.deep.mgz',
                      help='name under which segmentation will be saved. Default: aparc.DKTatlas+aseg.deep.mgz. '
                           'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                           'mri/aparc.DKTatlas+aseg.deep.mgz)')
    parser.add_option('--order', dest='order', type="int", default=1,
                      help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 3. Options for log-file and search-tag
    parser.add_option('--t', '--tag', dest='search_tag', default="*",
                      help='Search tag to process only certain subjects. If a single image should be analyzed, set the '
                           'tag with its id. Default: processes all.')
    parser.add_option('--log', dest='logfile', help='name of log-file. Default: deep-seg.log',
                      default='deep-seg.log')

    # 4. Pre-trained weights
    parser.add_option('--network_sagittal_path', dest='network_sagittal_path',
                      help="path to pre-trained weights of sagittal network",
                      default='./checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_option('--network_coronal_path', dest='network_coronal_path',
                      help="pre-trained weights of coronal network",
                      default='./checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_option('--network_axial_path', dest='network_axial_path',
                      help="pre-trained weights of axial network",
                      default='./checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')

    # 5. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_option('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true')
    parser.add_option('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_option('--batch_size', type=int, default=16, help="Batch size for inference. Default: 16")
    parser.add_option('--simple_run', action='store_true', default=False,
                      help='Simplified run: only analyse one given image specified by --in_name (output: --out_name). '
                           'Need to specify absolute path to both --in_name and --out_name if this option is chosen.')
    (sel_option, args) = parser.parse_args()

    if sel_option.input is None and sel_option.csv_file is None and not sel_option.simple_run:
        sys.exit('ERROR: Please specify data directory or input volume')

    if sel_option.output is None and not sel_option.simple_run:
        sys.exit('ERROR: Please specify data output directory (can be same as input directory)')

    return sel_option


def fast_surfer_cnn(img_filename, save_as, args):
    """
    Cortical parcellation of single image
    :param str img_filename: name of image file
    :param parser.Options args: Arguments (passed via command line) to set up networks
            * args.network_sagittal_path: path to sagittal checkpoint (stored pretrained network)
            * args.network_coronal_path: path to coronal checkpoint (stored pretrained network)
            * args.network_axial_path: path to axial checkpoint (stored pretrained network)
            * args.cleanup: Whether to clean up the segmentation (medial filter on certain labels)
            * args.no_cuda: Whether to use CUDA (GPU) or not (CPU)
    :param str save_as: name under which to save prediction.
    :return None: saves prediction to save_as
    """
    print("Reading volume {}".format(img_filename))

    header_info, affine_info, orig_data = load_and_conform_image(img_filename, interpol=args.order)

    transform_test = transforms.Compose([ToTensorTest()])

    test_dataset_axial = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Axial')

    test_dataset_sagittal = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Sagittal')

    test_dataset_coronal = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Coronal')

    start = time.time()
    
    test_data_loader = DataLoader(dataset=test_dataset_axial, batch_size=args.batch_size, shuffle=False)

    # Axial View Testing
    params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 79, 'kernel_c': 1, 'kernel_d': 1}

    # Select the model
    model = FastSurferCNN(params_network)

    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda available: {}, "
          "# Available GPUS: {}, "
          "Cuda user disabled (--no_cuda flag): {}, "
          "--> Using device: {}".format(torch.cuda.is_available(), torch.cuda.device_count(), args.no_cuda, device))

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)

    # Set up state dict (remapping of names, if not multiple GPUs/CPUs)
    print("Loading Axial Net from {}".format(args.network_axial_path))

    model_state = torch.load(args.network_axial_path, map_location=device)
    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not model_parallel:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and model_parallel:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    prediction_probability_axial = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

    print("Axial model loaded.")
    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):
            images_batch = Variable(sample_batch["image"])

            if use_cuda:
                images_batch = images_batch.cuda()

            temp = model(images_batch)

            prediction_probability_axial[start_index:start_index + temp.shape[0]] = temp.cpu()
            start_index += temp.shape[0]
            print("--->Batch {} Axial Testing Done.".format(batch_idx))

    end = time.time() - start
    print("Axial View Tested in {:0.4f} seconds".format(end))

    # Coronal View Testing
    start = time.time()

    test_data_loader = DataLoader(dataset=test_dataset_coronal, batch_size=args.batch_size, shuffle=False)

    params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 79, 'kernel_c': 1, 'kernel_d': 1}

    # Select the model

    model = FastSurferCNN(params_network)

    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)

    # Set up new state dict (remapping of names, if not multiple GPUs/CPUs)
    print("Loading Coronal Net from {}".format(args.network_coronal_path))

    model_state = torch.load(args.network_coronal_path, map_location=device)
    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not model_parallel:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and model_parallel:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    prediction_probability_coronal = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

    print("Coronal model loaded.")
    start_index = 0
    with torch.no_grad():

        for batch_idx, sample_batch in enumerate(test_data_loader):

            images_batch = Variable(sample_batch["image"])

            if use_cuda:
                images_batch = images_batch.cuda()

            temp = model(images_batch)

            prediction_probability_coronal[start_index:start_index + temp.shape[0]] = temp.cpu()
            start_index += temp.shape[0]
            print("--->Batch {} Coronal Testing Done.".format(batch_idx))

    end = time.time() - start
    print("Coronal View Tested in {:0.4f} seconds".format(end))

    start = time.time()

    test_data_loader = DataLoader(dataset=test_dataset_sagittal, batch_size=args.batch_size, shuffle=False)

    params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 51, 'kernel_c': 1, 'kernel_d': 1}

    # Select the model
    model = FastSurferCNN(params_network)

    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)

    # Set up new state dict (remapping of names, if not multiple GPUs/CPUs)
    print("Loading Sagittal Net from {}".format(args.network_sagittal_path))

    model_state = torch.load(args.network_sagittal_path, map_location=device)
    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not model_parallel:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and model_parallel:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    prediction_probability_sagittal = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

    start_index = 0
    with torch.no_grad():

        for batch_idx, sample_batch in enumerate(test_data_loader):

            images_batch = Variable(sample_batch["image"])

            if use_cuda:
                images_batch = images_batch.cuda()

            temp = model(images_batch)

            prediction_probability_sagittal[start_index:start_index + temp.shape[0]] = temp.cpu()
            start_index += temp.shape[0]
            print("--->Batch {} Sagittal Testing Done.".format(batch_idx))

    prediction_probability_sagittal = map_prediction_sagittal2full(prediction_probability_sagittal)
    end = time.time() - start

    print("Sagittal View Tested in {:0.4f} seconds".format(end))

    del model, test_dataset_axial, test_dataset_coronal, test_dataset_sagittal, test_data_loader

    start = time.time()

    # Start View Aggregation: change from N,C,X,Y to coronal view with C in last dimension = H,W,D,C
    prediction_probability_axial = prediction_probability_axial.permute(3, 0, 2, 1)
    prediction_probability_coronal = prediction_probability_coronal.permute(2, 3, 0, 1)
    prediction_probability_sagittal = prediction_probability_sagittal.permute(0, 3, 2, 1)

    intermediate_img = torch.add(prediction_probability_axial, prediction_probability_coronal)
    del prediction_probability_axial, prediction_probability_coronal

    _, prediction_image = torch.max(torch.add(torch.mul(intermediate_img, 0.4),
                                              torch.mul(prediction_probability_sagittal, 0.2)), 3)

    del prediction_probability_sagittal, intermediate_img

    prediction_image = prediction_image.numpy()

    end = time.time() - start
    print("View Aggregation finished in {:0.4f} seconds.".format(end))

    prediction_image = map_label2aparc_aseg(prediction_image)

    # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
    rh_wm = get_largest_cc(prediction_image == 41)
    lh_wm = get_largest_cc(prediction_image == 2)
    rh_wm = regionprops(label(rh_wm, background=0))
    lh_wm = regionprops(label(lh_wm, background=0))
    centroid_rh = np.asarray(rh_wm[0].centroid)
    centroid_lh = np.asarray(lh_wm[0].centroid)

    labels_list = np.array([1003, 1006, 1007, 1008, 1009, 1011,
                            1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    for label_current in labels_list:

        label_img = label(prediction_image == label_current, connectivity=3, background=0)

        for region in regionprops(label_img):

            if region.label != 0:  # To avoid background

                if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(
                        np.asarray(region.centroid) - centroid_lh):
                    mask = label_img == region.label
                    prediction_image[mask] = label_current + 1000

    # Quick Fixes for overlapping classes
    aseg_lh = gaussian_filter(1000 * np.asarray(prediction_image == 2, dtype=np.float), sigma=3)
    aseg_rh = gaussian_filter(1000 * np.asarray(prediction_image == 41, dtype=np.float), sigma=3)

    lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                            axis=3)

    # Problematic classes: 1026, 1011, 1029, 1019
    for prob_class_lh in [1011, 1019, 1026, 1029]:
        prob_class_rh = prob_class_lh + 1000
        mask_lh = ((prediction_image == prob_class_lh) | (prediction_image == prob_class_rh)) & (lh_rh_split == 0)
        mask_rh = ((prediction_image == prob_class_lh) | (prediction_image == prob_class_rh)) & (lh_rh_split == 1)

        prediction_image[mask_lh] = prob_class_lh
        prediction_image[mask_rh] = prob_class_rh

    # Clean-Up
    if args.cleanup is True:

        labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                  15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                  46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                  77, 1026, 2026]

        start = time.time()
        prediction_image_medfilt = median_filter(prediction_image, size=(3, 3, 3))
        mask = np.zeros_like(prediction_image)
        tolerance = 25

        for current_label in labels:
            current_class = (prediction_image == current_label)
            label_image = label(current_class, connectivity=3)

            for region in regionprops(label_image):

                if region.area <= tolerance:
                    mask_label = (label_image == region.label)
                    mask[mask_label] = 1

        prediction_image[mask == 1] = prediction_image_medfilt[mask == 1]
        end = time.time() - start
        print("Segmentation Cleaned up in {:0.4f} seconds.".format(end))

    # Saving image
    header_info.set_data_dtype(np.int16)
    mapped_aseg_img = nib.MGHImage(prediction_image, affine_info, header_info)
    mapped_aseg_img.to_filename(save_as)
    print("Saving Segmentation to {}".format(save_as))


if __name__ == "__main__":

    # Command Line options and error checking done here
    options = options_parse()

    if options.simple_run:

        # Check if output subject directory exists and create it otherwise
        sub_dir, out = op.split(options.oname)

        if not op.exists(sub_dir):
            makedirs(sub_dir)

        fast_surfer_cnn(options.iname, options.oname, options)

    else:

        # Prepare subject list to be processed
        if options.csv_file is not None:
            with open(options.csv_file, "r") as s_dirs:
                subject_directories = [line.strip() for line in s_dirs.readlines()]

        else:
            search_path = op.join(options.input, options.search_tag)
            subject_directories = glob.glob(search_path)

        # Report number of subjects to be processed and loop over them
        data_set_size = len(subject_directories)
        print("Total Dataset Size is {}".format(data_set_size))

        for current_subject in subject_directories:

            start_time = time.time()
            subject = current_subject.split("/")[-1]

            # Define volume to process, log-file and name under which final prediction will be saved
            if options.csv_file:

                dataset = current_subject.split("/")[-2]
                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, dataset, subject, options.logfile)
                save_file_name = op.join(options.output, dataset, subject, options.oname)

            else:

                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, subject, options.logfile)
                save_file_name = op.join(options.output, subject, options.oname)

            print("Running Fast Surfer on {}".format(subject))

            # Check if output subject directory exists and create it otherwise
            sub_dir, out = op.split(save_file_name)

            if not op.exists(sub_dir):
                makedirs(sub_dir)

            # Prepare the log-file
            old_stdout = sys.stdout
            log_file = open(logfile, "w")
            sys.stdout = log_file

            # Run network
            fast_surfer_cnn(invol, save_file_name, options)

            end_time = time.time() - start_time

            print("Total time for computation of segmentation is {:0.4f} seconds.".format(end_time))

            sys.stdout = old_stdout
            log_file.close()

            print("Total time for computation of segmentation is {:0.4f} seconds.".format(end_time))

        sys.exit(0)
