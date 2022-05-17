from argparse import ArgumentParser
import os
import pdb
from collections import OrderedDict

import numpy as np
import nibabel as nib
from scipy.special import softmax
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation, generate_binary_structure

# Need it if providing the posteriors.
ASEG_LABELS = {
    'Background': 0,
    'Right-Hippocampus': 53,
    'Left-Hippocampus': 17,
    'Right-Lateral-Ventricle': 43,
    'Left-Lateral-Ventricle': 4,
    'Right-Thalamus': 49,
    'Left-Thalamus': 10,
    'Right-Amygdala': 54,
    'Left-Amygdala': 18,
    'Right-Putamen': 51,
    'Left-Putamen': 12,
    'Right-Pallidum': 52,
    'Left-Pallidum': 13,
    'Right-Cerebrum-WM': 41,
    'Left-Cerebrum-WM': 2,
    'Right-Cerebellar-WM': 46,
    'Left-Cerebellar-WM': 7,
    'Right-Cerebrum-GM': 42,
    'Left-Cerebrum-GM': 3,
    'Right-Cerebellar-GM': 47,
    'Left-Cerebellar-GM': 8,
    'Right-Caudate': 50,
    'Left-Caudate': 11,
    'Brainstem': 16,
    '4th-Ventricle': 15,
    '3rd-Ventricle': 14,
    'Right-Accumbens': 58,
    'Left-Accumbens': 26,
    'Right-VentralDC': 60,
    'Left-VentralDC': 28,
    'Right-Inf-Lat-Ventricle': 44,
    'Left-Inf-Lat-Ventricle': 5,
}

# Need it for clustering regions under the same gaussian.
CLUSTER_DICT = {
    'Gray': [53, 17, 51, 12, 54, 18, 50, 11, 58, 26, 42, 3],
    'CSF': [4, 5, 43, 44, 15, 14, 24],
    'Thalaumus': [49, 10],
    'Pallidum': [52, 13],
    'VentralDC': [28, 60],
    'Brainstem': [16],
    'WM': [41, 2],
    'cllGM': [47, 8],
    'cllWM': [46, 7]
}

eps = np.finfo(float).eps


def main():
    print('\n\n\n\n\n')
    print('# --------------------------------------------------------------------------- #')
    print('# Bias field correction using segmentations (e.g., from FreeSurfer, Synthseg) #')
    print('# --------------------------------------------------------------------------- #')
    print('\n')
    print(
        'Credit to: Puonti et al. Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling '
        '(SAMSEG, 2016)')
    print('\n\n')
    print('[NOTE] Please, modify the CLUSTER_DICT according to your different grouping of regions. \n'
          'Current grouping (with one gaussian per group): ')
    for k, v in CLUSTER_DICT.items():
        print(' - ' + k + ': ' + ', '.join([str(lab) for lab in v]))
    print('\n')

    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--image', type=str, help='Image filepath')
    arg_parser.add_argument('--seg', type=str,
                            help='Segmentation filepath. It can be either the posteriors for each label (labels '
                                 'assumed in the last dimension) or hard  segmentations (by default, please use '
                                 'Synthseg or Aseg segmentations')
    arg_parser.add_argument('--out', type=str, default="bf_corrected_image.mgz",
                            help='Bias field corrected image name. Will be stored in the same directory as the input image.')
    arg_parser.add_argument('--bf', type=str, default="bias_field.mgz",
                            help='Bias field. Will be stored in the same directory as the input image.')
    arg_parser.add_argument('--norm', action='store_true', default=False,
                            help='Normalize image after bias field correction (WM = 110)')
    arg_parser.add_argument('--compute_distance_map', action='store_true',
                            help='Flag to compute soft segmentation using a distance map. If not specified, a '
                                 'one-hot encoding with gaussian smoothing will be used.')

    # Get parameters
    arguments = arg_parser.parse_args()
    image_file = arguments.image
    seg_file = arguments.seg
    distance_map_flag = arguments.compute_distance_map

    # Get images
    proxy = nib.load(image_file)
    mri = np.asarray(proxy.dataobj)
    proxy = nib.load(seg_file)
    seg = np.asarray(proxy.dataobj)

    is_hard = len(mri.shape) == len(seg.shape)

    if distance_map_flag and is_hard:
        seg_uni = convert_to_unified(seg)
        soft_seg = compute_distance_map(seg_uni, soft_seg=True)

    elif is_hard:
        # seg_uni = convert_to_unified(seg)
        # unique_labels = np.unique(seg_uni)
        soft_seg = np.transpose(one_hot_encoding(seg, categories=np.array(list(ASEG_LABELS.values()))),
                                axes=(1, 2, 3, 0))
        soft_seg = convert_posteriors_to_unified(soft_seg)

    else:
        soft_seg = convert_posteriors_to_unified(seg)

    # Run bias field
    mri, bias_field = bias_field_corr(mri, soft_seg, penalty=1)

    if arguments.norm:
        # Normalize the image
        if is_hard:
            wm_label = CLUSTER_DICT['WM']
        else:
            wm_label = [it_k for it_k, k in enumerate(CLUSTER_DICT.items()) if k == 'WM']
            seg = np.argmax(seg, -1)

        wm_mask = np.zeros_like(seg)
        for wl in wm_label:
            wm_mask = wm_mask | (seg == wl)
        wm_mask = wm_mask.astype('bool')

        m = np.mean(mri[wm_mask])
        mri = 110 * mri / m
        mri = mri.astype('float32')

    # Save the results
    image_dir = os.path.dirname(image_file)

    img = nib.Nifti1Image(mri, proxy.affine)
    nib.save(img, os.path.join(image_dir, arguments.out))

    img = nib.Nifti1Image(bias_field, proxy.affine)
    nib.save(img, os.path.join(image_dir, arguments.bf))


# main bias-field correction
def bias_field_corr(init_image, init_seg, penalty=0., patience=3):
    '''
    :param image: np array. Input image to correct
    :param seg: np.array. Soft segmentation or one-hot encoding of the segmentation with shape=image.shape + (num_labels).
    :param penalty: regularization term over the coefficients.
    :param patience: int, default=3. Number indicating the maximum number of iterations where improvement < 1e-6
    :return:
    '''

    vol_shape = init_image.shape

    # image = image.reshape(-1, 1)
    init_image_log = np.log(init_image[..., np.newaxis] + 1e-5)

    # seg = seg.reshape(-1, seg.shape[-1])
    init_mask = np.sum(init_seg, axis=-1) > 0
    mask, crop_coord = crop_label(init_mask, margin=10)
    processing_shape = mask.shape

    image_log = apply_crop(init_image_log, crop_coord)
    seg = apply_crop(init_seg, crop_coord)

    basis_functions = get_dct_basis_functions(processing_shape, [50] * 3)
    num_coeff = int(np.prod([b.shape[1] for b in basis_functions]))
    coeff = np.zeros((num_coeff, 1))
    llh_last = 1
    print('         # Bias field correction #')

    it_break = 0
    for it in range(100):
        bias_field_log = getBiasFields(coeff, basis_functions)
        image_log_corr = image_log[mask] - bias_field_log[mask]
        image_posteriors = seg[mask]

        u_j = np.sum(image_posteriors * image_log_corr, axis=0) / np.sum(image_posteriors, axis=0)
        s_j = np.sum(image_posteriors * (image_log_corr - u_j) ** 2, axis=0) / np.sum(image_posteriors, axis=0)
        u_j = u_j.reshape(-1, 1)
        s_j = s_j.reshape(-1, 1, 1)

        _, llh = getGaussianPosteriors(image_log_corr, image_posteriors, u_j, s_j)
        coeff = fitBiasFieldParameters(image_log, image_posteriors, u_j, s_j, basis_functions, mask, penalty)

        improv = np.max(np.abs((llh - llh_last) / llh_last))
        print('          ' + str(it) + '. Loglikelihood improvement: ' + str(improv))
        llh_last = llh
        if improv < 1e-3:
            it_break += 1
            if it_break > patience:
                break

        else:
            it_break = 0

    if it_break == 0: print('         ### Algorithm did not converge: ')
    print('         ### Ended with loglikelihood: ' + str(llh_last))

    bias_field_log = getBiasFields(coeff, basis_functions)
    image_corr, bias_field = undoLogTransformAndBiasField(image_log, bias_field_log, mask)
    image_corr[np.isnan(image_corr)] = 0

    # undo cropping
    fi_image_corr = np.zeros(vol_shape)
    fi_image_corr[crop_coord[0][0]: crop_coord[0][1],
    crop_coord[1][0]: crop_coord[1][1],
    crop_coord[2][0]: crop_coord[2][1]] = np.squeeze(image_corr)
    fi_bias_field = np.zeros(vol_shape)
    fi_bias_field[crop_coord[0][0]: crop_coord[0][1],
    crop_coord[1][0]: crop_coord[1][1],
    crop_coord[2][0]: crop_coord[2][1]] = np.squeeze(bias_field)
    return fi_image_corr, fi_bias_field


# ------------------------------------------------------- utils -------------------------------------------------------

def crop_label(mask, margin=10, threshold=0):
    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin = [margin] * ndim

    crop_coord = []
    idx = np.where(mask > threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord


def apply_crop(image, crop_coord):
    return image[crop_coord[0][0]: crop_coord[0][1],
           crop_coord[1][0]: crop_coord[1][1],
           crop_coord[2][0]: crop_coord[2][1]
           ]


def convert_posteriors_to_unified(seg):
    '''
    Converts a Freesurfer or Synthseg segmentation to unified segmentation by jointly considering both hemispheres.
    :param seg: np.array
    :return: np.array with number_of_classes = len(CLUSTER_DICT.keys()).
    '''
    out_seg = np.zeros(seg.shape[:-1] + (len(CLUSTER_DICT.keys()),))
    for it_lab, (lab_str, lab_list) in enumerate(CLUSTER_DICT.items()):
        for lab in lab_list:
            out_seg[..., it_lab] += seg[..., np.argmax(np.array(list(ASEG_LABELS.values())) == lab)]

    out_seg = out_seg / np.sum(out_seg, axis=-1, keepdims=True)
    out_seg[np.isnan(out_seg)] = 0
    return out_seg


def convert_to_unified(seg):
    '''
    Converts a Freesurfer or Synthseg segmentation to unified segmentation by jointly considering both hemispheres.
    :param seg: np.array
    :return: np.array with number_of_classes = len(CLUSTER_DICT.keys()).
    '''
    out_seg = np.zeros(seg.shape[:3] + (len(CLUSTER_DICT.keys()),))
    for it_lab, (lab_str, lab_list) in enumerate(CLUSTER_DICT.items()):
        for lab in lab_list:
            out_seg[seg == lab] = it_lab

    out_seg = out_seg / np.sum(out_seg, axis=-1, keepdims=True)

    return out_seg


def compute_distance_map(labelmap, soft_seg=True):
    '''
    Compute distance map for different labels.
    :param labelmap: np.array
    :param soft_seg: bool. If True, it applies the softmax to the output. Otherwise it will output distances.
    :return:
    '''
    unique_labels = np.unique(labelmap)
    distancemap = -200 * np.ones(labelmap.shape + (len(unique_labels),), dtype='float32')
    # print('Working in label: ', end='', flush=True)
    for it_ul, ul in enumerate(unique_labels):
        # print(str(ul), end=', ', flush=True)

        mask_label = labelmap == ul
        bbox_label, crop_coord = crop_label(mask_label, margin=5)

        d_in = (distance_transform_edt(bbox_label))
        d_out = -distance_transform_edt(~bbox_label)
        d = np.zeros_like(d_in)
        d[bbox_label] = d_in[bbox_label]
        d[~bbox_label] = d_out[~bbox_label]

        distancemap[crop_coord[0][0]: crop_coord[0][1],
        crop_coord[1][0]: crop_coord[1][1],
        crop_coord[2][0]: crop_coord[2][1], it_ul] = d

    if soft_seg:
        prior_labels = softmax(distancemap, axis=-1)
        # soft_labelmap = np.argmax(prior_labels, axis=-1).astype('uint16')
        return prior_labels
    else:
        return distancemap


def one_hot_encoding(target, num_classes=None, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None and num_classes is None:
        raise ValueError('[ONE-HOT Enc.] You need to specify the number of classes or the categories.')
    elif categories is not None:
        num_classes = len(categories)
    else:
        categories = np.arange(num_classes)

    labels = np.zeros((num_classes,) + target.shape, dtype='int')
    for it_cls, cls in enumerate(categories):
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1

    return labels


# -------------------------------------------- bias field related functions --------------------------------------------

def get_dct_basis_functions(image_shape, smoothing_kernel_size):
    '''
    Our bias model is a linear combination of a set of basis functions. We are using so-called
    "DCT-II" basis functions, i.e., the lowest few frequency components of the Discrete Cosine
    Transform.

    Credit to: SAMSEG (Freesurfer)

    :param image_shape: (tuple)
    :param smoothing_kernel_size: ()
    :return:
    '''

    biasFieldBasisFunctions = []
    for dimensionNumber in range(len(image_shape)):
        N = image_shape[dimensionNumber]
        delta = smoothing_kernel_size[dimensionNumber]
        M = (np.ceil(N / delta) + 1).astype('int')
        Nvirtual = (M - 1) * delta
        js = [(index + 0.5) * np.pi / Nvirtual for index in range(N)]
        scaling = [np.sqrt(2 / Nvirtual)] * M
        scaling[0] /= np.sqrt(2)
        A = np.array([[np.cos(freq * m) * scaling[m] for m in range(M)] for freq in js])
        biasFieldBasisFunctions.append(A)

    return biasFieldBasisFunctions


def backprojectKroneckerProductBasisFunctions(kroneckerProductBasisFunctions, coefficients):
    numberOfDimensions = len(kroneckerProductBasisFunctions)
    Ms = np.zeros(numberOfDimensions, dtype=np.uint32)  # Number of basis functions in each dimension
    Ns = np.zeros(numberOfDimensions, dtype=np.uint32)  # Number of basis functions in each dimension
    transposedKroneckerProductBasisFunctions = []
    for dimensionNumber in range(numberOfDimensions):
        Ms[dimensionNumber] = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
        Ns[dimensionNumber] = kroneckerProductBasisFunctions[dimensionNumber].shape[0]
        transposedKroneckerProductBasisFunctions.append(kroneckerProductBasisFunctions[dimensionNumber].T)
    y = projectKroneckerProductBasisFunctions(transposedKroneckerProductBasisFunctions,
                                              coefficients.reshape(Ms, order='F'))
    Y = y.reshape(Ns, order='F')
    return Y


def projectKroneckerProductBasisFunctions(kroneckerProductBasisFunctions, T):
    #
    # Compute
    #   c = W' * t
    # where
    #   W = W{ numberOfDimensions } \kron W{ numberOfDimensions-1 } \kron ... W{ 1 }
    # and
    #   t = T( : )
    numberOfDimensions = len(kroneckerProductBasisFunctions)
    currentSizeOfT = list(T.shape)
    for dimensionNumber in range(numberOfDimensions):
        # Reshape into 2-D, do the work in the first dimension, and shape into N-D
        T = T.reshape((currentSizeOfT[0], -1), order='F')
        T = (kroneckerProductBasisFunctions[dimensionNumber]).T @ T
        currentSizeOfT[0] = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
        T = T.reshape(currentSizeOfT, order='F')
        # Shift dimension
        currentSizeOfT = currentSizeOfT[1:] + [currentSizeOfT[0]]
        T = np.rollaxis(T, 0, 3)
    # Return result as vector
    coefficients = T.flatten(order='F')
    return coefficients


def computePrecisionOfKroneckerProductBasisFunctions(kroneckerProductBasisFunctions, B):
    #
    # Compute
    #   H = W' * diag( B ) * W
    # where
    #   W = W{ numberOfDimensions } \kron W{ numberOfDimensions-1 } \kron ... W{ 1 }
    # and B is a weight matrix
    numberOfDimensions = len(kroneckerProductBasisFunctions)

    # Compute a new set of basis functions (point-wise product of each combination of pairs) so that we can
    # easily compute a mangled version of the result
    Ms = np.zeros(numberOfDimensions, dtype=np.uint32)  # Number of basis functions in each dimension
    hessianKroneckerProductBasisFunctions = {}
    for dimensionNumber in range(numberOfDimensions):
        M = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
        A = kroneckerProductBasisFunctions[dimensionNumber]
        hessianKroneckerProductBasisFunctions[dimensionNumber] = np.kron(np.ones((1, M)), A) * np.kron(A,
                                                                                                       np.ones((1, M)))
        Ms[dimensionNumber] = M
    result = projectKroneckerProductBasisFunctions(hessianKroneckerProductBasisFunctions, B)
    new_shape = list(np.kron(Ms, [1, 1]))
    new_shape.reverse()
    result = result.reshape(new_shape)
    permutationIndices = np.hstack((2 * np.r_[: numberOfDimensions], 2 * np.r_[: numberOfDimensions] + 1))
    result = np.transpose(result, permutationIndices)
    precisionMatrix = result.reshape((np.prod(Ms), np.prod(Ms)))
    return precisionMatrix


def fitBiasFieldParameters(image, soft_seg, means, variances, bias_field_functions, mask, penalty=1):
    # Bias field correction: implements Eq. 8 in the paper
    #    Van Leemput, "Automated Model-based Bias Field Correction of MR Images of the Brain", IEEE TMI 1999

    #
    numberOfGaussians = means.shape[0]
    numberOfContrasts = means.shape[1]
    numberOfBasisFunctions = [functions.shape[1] for functions in bias_field_functions]
    numberOf3DBasisFunctions = np.prod(numberOfBasisFunctions)

    # Set up the linear system lhs * x = rhs
    precisions = np.zeros_like(variances)
    for gaussianNumber in range(numberOfGaussians):
        precisions[gaussianNumber, :, :] = np.linalg.inv(variances[gaussianNumber, :, :]).reshape(
            (1, numberOfContrasts, numberOfContrasts))

    lhs = np.zeros((numberOf3DBasisFunctions * numberOfContrasts,
                    numberOf3DBasisFunctions * numberOfContrasts))  # left-hand side of linear system
    rhs = np.zeros((numberOf3DBasisFunctions * numberOfContrasts, 1))  # right-hand side of linear system
    weightsImageBuffer = np.zeros(mask.shape)
    tmpImageBuffer = np.zeros(mask.shape)
    for contrastNumber1 in range(numberOfContrasts):
        # logger.debug('third time contrastNumber=%d', contrastNumber)
        contrast1Indices = np.arange(0, numberOf3DBasisFunctions) + \
                           contrastNumber1 * numberOf3DBasisFunctions

        tmp = np.zeros(soft_seg.shape[0])
        for contrastNumber2 in range(numberOfContrasts):
            contrast2Indices = np.arange(0, numberOf3DBasisFunctions) + \
                               contrastNumber2 * numberOf3DBasisFunctions

            classSpecificWeights = soft_seg * precisions[:, contrastNumber1, contrastNumber2]
            weights = np.sum(classSpecificWeights, 1)

            # Build up stuff needed for rhs
            predicted = np.sum(classSpecificWeights * means[:, contrastNumber2], 1) / (weights + eps)
            residue = image[mask, contrastNumber2] - predicted
            tmp += weights * residue

            # Fill in submatrix of lhs
            weightsImageBuffer[mask] = weights
            lhs[np.ix_(contrast1Indices, contrast2Indices)] \
                = computePrecisionOfKroneckerProductBasisFunctions(bias_field_functions,
                                                                   weightsImageBuffer)

        tmpImageBuffer[mask] = tmp
        rhs[contrast1Indices] = projectKroneckerProductBasisFunctions(bias_field_functions,
                                                                      tmpImageBuffer).reshape(-1, 1)

    # Solve the linear system x = lhs \ rhs
    solution = np.linalg.solve(lhs + penalty * np.eye(lhs.shape[0]), rhs)

    #
    biasFieldCoefficients = solution.reshape((numberOfContrasts, numberOf3DBasisFunctions)).transpose()
    return biasFieldCoefficients


def getBiasFields(biasFieldCoefficients, biasFieldBasisFunctions, mask=None):
    #
    numberOfContrasts = biasFieldCoefficients.shape[-1]
    imageSize = tuple([functions.shape[0] for functions in biasFieldBasisFunctions])
    biasFields = np.zeros(imageSize + (numberOfContrasts,), order='F')
    for contrastNumber in range(numberOfContrasts):
        biasField = backprojectKroneckerProductBasisFunctions(
            biasFieldBasisFunctions, biasFieldCoefficients[:, contrastNumber])
        if mask is not None:
            biasField *= mask
        biasFields[:, :, :, contrastNumber] = biasField

    return biasFields


def undoLogTransformAndBiasField(imageBuffers, biasFields, mask):
    #
    expBiasFields = np.zeros(biasFields.shape, order='F')
    numberOfContrasts = imageBuffers.shape[-1]
    for contrastNumber in range(numberOfContrasts):
        # We're computing it also outside of the mask, but clip the intensities there to the range
        # observed inside the mask (with some margin) to avoid crazy extrapolation values
        biasField = biasFields[:, :, :, contrastNumber]
        clippingMargin = np.log(2)
        clippingMin = biasField[mask].min() - clippingMargin
        clippingMax = biasField[mask].max() + clippingMargin
        biasField[biasField < clippingMin] = clippingMin
        biasField[biasField > clippingMax] = clippingMax
        expBiasFields[:, :, :, contrastNumber] = np.exp(biasField)

    #
    expImageBuffers = np.exp(imageBuffers) / expBiasFields

    #
    return expImageBuffers, expBiasFields


def getGaussianLikelihoods(data, mean, variance):
    squared_mahalanobis_dist = (data - mean) ** 2 / variance
    scaling = 1.0 / (2 * np.pi * variance) ** (1 / 2)
    gaussianLikelihoods = np.exp(-0.5 * squared_mahalanobis_dist) * scaling
    return gaussianLikelihoods.T


def getGaussianPosteriors(data, classPriors, means, variances):
    numberOfClasses = classPriors.shape[-1]
    numberOfVoxels = data.shape[0]

    gaussianPosteriors = np.zeros((numberOfVoxels, numberOfClasses), order='F')
    for classNumber in range(numberOfClasses):
        classPrior = classPriors[:, classNumber]
        mean = np.expand_dims(means[classNumber, :], 1)
        variance = variances[classNumber, :]

        gaussianLikelihoods = getGaussianLikelihoods(data, mean, variance)
        gaussianPosteriors[:, classNumber] = gaussianLikelihoods * classPrior

    normalizer = np.sum(gaussianPosteriors, axis=1) + eps
    gaussianPosteriors = gaussianPosteriors / np.expand_dims(normalizer, 1)

    minLogLikelihood = -np.sum(np.log(normalizer))

    return gaussianPosteriors, minLogLikelihood


# execute script
if __name__ == '__main__':
    main()


