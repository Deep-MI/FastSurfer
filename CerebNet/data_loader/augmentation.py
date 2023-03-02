
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
import numbers
import random
from typing import Optional

import numpy as np
import torch
from numpy import random as npr
from scipy.ndimage import gaussian_filter, affine_transform
from scipy.stats import median_abs_deviation
from torchvision import transforms

from CerebNet.data_loader.data_utils import FLIPPED_LABELS


##
# Transformations for training
##
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def _apply_img(self, img):
        img = img.astype(np.float32)

        # Normalize image and clamp between 0 and 1
        # img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img_max = img.max()
        img_min = img.min()
        img = img - img_min

        if img_max == img_min:
            return torch.from_numpy(img)

        img = img / (img_max - img_min)
        return torch.from_numpy(img)

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['image'] = self._apply_img(sample['image'])
        else:
            return self._apply_img(sample)
        # sample['weight'] = sample['weight'].astype(np.float32)

        return sample


class ToTensorTest(ToTensor):

    def _apply_img(self, img):
        # move the "target plane" front first (see also FastSurferCNN.data_loader.augmentation.ToTest)
        return super()._apply_img(img.transpose((2, 0, 1)))


class RandomAffine(object):
    """
    Apply a random affine transformation to
    images, label and weight
    the transformation includes translation, rotation and scaling
    """
    def __init__(self, cfg):
        self.degree = cfg.AUGMENTATION.DEGREE
        self.img_size = [cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH]
        self.scale = cfg.AUGMENTATION.SCALE
        self.translate = cfg.AUGMENTATION.TRANSLATE
        self.prob = cfg.AUGMENTATION.PROB
        self.seed = None

    def _get_random_affine(self):
        '''
                Random inverse affine matrix composed of rotation matrix (of each axis)and translation.

                Parameters
                ----------
                degrees : sequence or float or int,
                     Range of degrees to select from.
                     If degrees is a number instead of sequence like (min, max), the range of degrees
                     will be (-degrees, +degrees).
                translate : tuple, float
                     if translate=(a,b), the value for translation is uniformly sampled
                     in the range
                      -column_size * a < dx < column_size * a
                      -row_size * b < dy < row_size * b
                     If translate is a number then a=b=c.
                     The value should be between 0 and 1.
                img_size : tuple
                    img_size = (column_size, row_size)
                scale: tuple, range of min and max scaling factor
                seed : int
                    random seed

                Returns
                -------
                transform_mat : 3x3 matrix
                    Random affine transformation
            '''

        if isinstance(self.degree, numbers.Number):
            if self.degree < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-self.degree, self.degree)
        else:
            assert isinstance(self.degree, (tuple, list)) and len(self.degree) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
        if isinstance(self.translate, numbers.Number):
            if not (0.0 <= self.translate <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
            translate = (self.translate, self.translate)
        else:
            assert isinstance(self.translate, (tuple, list)) and len(self.translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in self.translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")

        if self.seed is not None:
            random.seed(self.seed)

        center = np.array([self.img_size[0] * 0.5 + 0.5, self.img_size[1] * 0.5 + 0.5])
        max_dx = translate[0] * self.img_size[0]
        max_dy = translate[1] * self.img_size[1]
        translations = np.array([random.uniform(-max_dx, max_dx),
                                 random.uniform(-max_dy, max_dy)])
        angle = random.uniform(degrees[0], degrees[1])
        #     print(f"Rotation of {angle:.4f} degree along axis x")
        angle_rad = np.radians(angle)
        # inverse of rotation matrix
        rot_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                               [-np.sin(angle_rad), np.cos(angle_rad)]])
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        rot_matrix = rot_matrix / scale_factor
        # inverse of translation and of center translation
        inv_trans_center = (rot_matrix @ (-center - translations))
        transform_mat = np.insert(rot_matrix, 2, inv_trans_center, axis=1)
        # center translation
        transform_mat[:, -1] += center
        return transform_mat

    def __call__(self, sample):
        if random.random() < self.prob:
            random_affine_mat = self._get_random_affine()
            slices_num, h, w = sample['image'].shape
            sample['image'] = sample['image'].astype(np.float32)
            # fixme need to be vectorized
            for s_id in range(slices_num):
                sample['image'][s_id, :, :] = affine_transform(sample['image'][s_id, :, :], random_affine_mat, order=3)

            sample['label'] = affine_transform(sample['label'], random_affine_mat, order=0)

        return sample


class RandomFlip(object):
    """
    Random horizontal flipping
    """
    def __init__(self, cfg):
        self.prob = cfg.AUGMENTATION.PROB
        self.axis = cfg.AUGMENTATION.FLIP_AXIS
        self.swap_labels = cfg.DATA.PLANE != 'sagittal'

    def _flip_labels(self, label_map):
        flipped = np.flip(label_map, axis=self.axis).copy()
        if self.swap_labels:
            flipped = FLIPPED_LABELS[flipped.ravel()].reshape(flipped.shape)
        return flipped

    def __call__(self, sample):
        if random.random() < self.prob:
            img_flip_axis = self.axis+(1 if len(sample['image'].shape) == 3 else 0)
            sample['image'] = np.flip(sample['image'], axis=img_flip_axis).copy()
            sample['label'] = self._flip_labels(sample['label'])
        return sample


class RandomBiasField:
    r"""Add random MRI bias field artifact.

    Based on https://github.com/fepegar/torchio

    It was implemented in NiftyNet by Carole Sudre and used in
    `Sudre et al., 2017, Longitudinal segmentation of age-related
    white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        coefficients: Magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
        p: Probability that this transform will be applied.
        seed:
    """

    def __init__(
            self,
            cfg,
            seed: Optional[int] = None,
    ):
        coefficients = cfg.AUGMENTATION.BIAS_FIELD_COEFFICIENTS
        if isinstance(coefficients, float):
            coefficients = (-coefficients, coefficients)

        self.coefficients = coefficients
        self.order = cfg.AUGMENTATION.BIAS_FIELD_ORDER
        self.prob = cfg.AUGMENTATION.PROB
        if seed is not None:
            random.seed(seed)

    def generate_bias_field(
            self,
            data: np.ndarray,
            coefficients: np.ndarray,
    ) -> np.ndarray:
        # Create the bias field map using a linear combination of polynomial
        # functions and the coefficients previously sampled
        shape = np.array(data.shape)
        half_shape = shape / 2

        ranges = [np.arange(-n, n) + 0.5 for n in half_shape]

        bias_field = np.zeros(shape)
        x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges))

        x_mesh /= x_mesh.max()
        y_mesh /= y_mesh.max()
        z_mesh /= z_mesh.max()

        i = 0
        for x_order in range(self.order + 1):
            for y_order in range(self.order + 1 - x_order):
                for z_order in range(self.order + 1 - (x_order + y_order)):
                    random_coefficient = coefficients[i]
                    new_map = (
                            random_coefficient
                            * x_mesh ** x_order
                            * y_mesh ** y_order
                            * z_mesh ** z_order
                    )
                    bias_field += np.transpose(new_map, (1, 0, 2))
                    i += 1
        bias_field = np.exp(bias_field).astype(np.float32)
        return bias_field

    def get_random_params(self):
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, self.order + 1):
            for y_order in range(0, self.order + 1 - x_order):
                for _ in range(0, self.order + 1 - (x_order + y_order)):
                    number = npr.uniform(*self.coefficients)
                    random_coefficients.append(number)
        return np.asarray(random_coefficients)

    def apply_transform(self, img):
        coefficients = self.get_random_params()
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        bias_field = self.generate_bias_field(img, coefficients)
        img = img * bias_field
        return img

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['image'] = np.squeeze(self.apply_transform(sample['image']))
        return sample


class RandomLabelsToImage(object):
    """
        Generate image from segmentation
        using the dataset intensity priors.
        based on  Billot et al.:
        A Learning Strategy for Contrast-agnostic MRI Segmentation
         and Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast.
    """

    def __init__(self, mean, std, cfg, blur_factor=0.3):
        self.means = mean
        self.stds = std
        self.prob = cfg.AUGMENTATION.PROB
        self.blur_factor = blur_factor

    def _calculate_gmm(self, labels, img_shape):
        image = npr.normal(size=img_shape)
        (h, w, d) = image.shape
        lbl = np.expand_dims(labels, axis=2).repeat(d)
        means_map = np.array(np.reshape(self.means[lbl.ravel()], (h, w, d)), dtype=np.float32)
        std_devs_map = np.array(np.reshape(self.stds[lbl.ravel()], (h, w, d)), dtype=np.float32)
        image = np.multiply(std_devs_map, image)
        image = np.add(image, means_map)
        return image

    def __call__(self, sample):
        if random.random() < self.prob:
            gmm_img = self._calculate_gmm(sample['label'], sample['image'].shape)
            sample['image'] = gaussian_filter(gmm_img, sigma=self.blur_factor)
        return sample


def sample_intensity_stats_from_image(image,
                                      segmentation,
                                      labels_list,
                                      classes_list=None,
                                      keep_strictly_positive=True):
    """This function takes an image and corresponding segmentation as inputs. It estimates the mean and std intensity
    for all specified label values. Labels can share the same statistics by being regrouped into K classes.
    :param image: image from which to evaluate mean intensity and std deviation.
    :param segmentation: segmentation of the input image. Must have the same size as image.
    :param labels_list: list of labels for which to evaluate mean and std intensity.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    :param classes_list: (optional) enables to regroup structures into classes of similar intensity statistics.
    Intenstites associated to regrouped labels will thus contribute to the same Gaussian during statistics estimation.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    It should have the same length as labels_list, and contain values between 0 and K-1, where K is the total number of
    classes. Default is all labels have different classes (K=len(labels_list)).
    :param keep_strictly_positive: (optional) whether to only keep strictly positive intensity values when
    computing stats. This doesn't apply to the first label in label_list (or class if class_list is provided), for
    which we keep positive and zero values, as we consider it to be the background label.
    :return: a numpy array of size (2, K), the first row being the mean intensity for each structure,
    and the second being the median absolute deviation (robust estimation of std).
    """
    # reformat labels and classes
    if classes_list is not None:
        classes_list = np.array(classes_list, dtype='int')
    else:
        classes_list = np.arange(labels_list.shape[0])
    assert len(classes_list) == len(labels_list), 'labels and classes lists should have the same length'
    # get unique classes
    unique_classes, unique_indices = np.unique(classes_list, return_index=True)
    n_classes = len(unique_classes)
    if not np.array_equal(unique_classes, np.arange(n_classes)):
        raise ValueError('classes_list should only contain values between 0 and K-1, '
                         'where K is the total number of classes. Here K = %d' % n_classes)
    if len(image.shape) == 4:
        h, w, d, n_slices = image.shape
        mid_slice = n_slices // 2
        image = image[:,:,:, mid_slice]

    # compute mean/std of specified classes
    means = np.zeros(n_classes)
    stds = np.zeros(n_classes)
    for idx, tmp_class in enumerate(unique_classes):
        # get list of all intensity values for the current class
        class_labels = labels_list[classes_list == tmp_class]
        intensities = np.empty(0)
        for label in class_labels:
            tmp_intensities = image[segmentation == label]
            intensities = np.concatenate([intensities, tmp_intensities])
        if tmp_class:  # i.e. if not background
            if keep_strictly_positive:
                intensities = intensities[intensities > 0]
        # compute stats for class and put them to the location of corresponding label values
        if len(intensities) != 0:
            means[idx] = np.nanmedian(intensities)
            stds[idx] = median_abs_deviation(intensities, nan_policy='omit')
    return np.stack([means, stds])


def get_transform(cfg):
    transform_list = []
    for tx in cfg.AUGMENTATION.TYPES:
        if tx == 'random_affine':
            rand_aff = RandomAffine(cfg)
            transform_list.append(rand_aff)
        if tx == 'flip':
            flip = RandomFlip(cfg)
            transform_list.append(flip)
        if tx == 'bias_field':
            rbf = RandomBiasField(cfg)
            transform_list.append(rbf)

    transform_list.append(ToTensor())
    return transforms.Compose(transform_list)
