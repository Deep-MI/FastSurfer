
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
from FastSurferCNN.utils import logging
from doctest import FAIL_FAST
from math import e, log
from numpy import NaN
import torch
from torch import nn, normal
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from pytorch_msssim import SSIM as SSIM_loss #, ms_ssim, SSIM, MS_SSIM
from torch.nn.modules.loss import KLDivLoss, CrossEntropyLoss

logger = logging.getLogger(__name__)

class DiceLoss(_Loss):
    """
    Dice Loss
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        :param output: N x C x H x W Variable
        :param target: N x C x W LongTensor with starting class at 0
        :param weights: C FloatTensor with class wise weights
        :param int ignore_index: ignore label with index x in the loss calculation
        :return:
        """
        eps = 0.001

        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))  # Channel-wise weights

        return loss_per_channel.sum() / output.size(1)


class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood
    """

    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(f"Initialized {self.__class__.__name__} with weight: {weight} and reduction: {reduction}")

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)
    

class SSIMLoss(_Loss):

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255, size_average=True, channel=1):
        """ from https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """
        super(SSIMLoss, self).__init__()
        ssim = SSIM_loss(win_size=window_size, win_sigma=window_sigma, data_range=data_range, size_average=size_average, channel=channel)
        self.ssim_loss = lambda pred, orig: 1-ssim(pred.unsqueeze(1), orig.unsqueeze(1))
        #self.mask_weight = mask_weight
        self.window_size = window_size

    def dilate_mask(self, min_x, max_x, img_size):
        # make rectangular mask bigger to enable gaussian in SSIM to work
        if torch.abs(min_x - max_x) < self.window_size * 2: 
            min_x = min_x - self.window_size 
            max_x = max_x + self.window_size 
            if min_x < 0:
                min_x = 0
            if max_x > img_size:
                max_x = img_size

        return min_x, max_x

    def ssim_loss_masked(self, pred, orig, mask):
        # get min and max values of mask
        
        mean_ssims = 0
        ssim_count = 0

        for i in range(pred.shape[0]):
            indices = torch.nonzero(mask[i])

            if len(indices) == 0:
                continue

            min_x, min_y = torch.min(indices, dim=0).values
            max_x, max_y = torch.max(indices, dim=0).values

            max_x += 1
            max_y += 1

            min_x, max_x = self.dilate_mask(min_x, max_x, pred.shape[1])
            min_y, max_y = self.dilate_mask(min_y, max_y, pred.shape[2])

            mean_ssims += self.ssim_loss(pred[i, min_x:max_x, min_y:max_y].unsqueeze(0), orig[i, min_x:max_x, min_y:max_y].unsqueeze(0))
            ssim_count += 1

        return mean_ssims / ssim_count if ssim_count > 0 else 0
        
    def forward(self, pred, orig, mask=None):
        if mask is None: #self.mask_weight <= 0 or mask is None:
            return self.ssim_loss(pred, orig) #* (1 - self.mask_weight)
        else:
            return self.ssim_loss_masked(pred, orig, mask) #* self.mask_weight # + self.ssim_loss(pred, orig)  * (1 - self.mask_weight) \
                  
        
        
class GradientLoss(_Loss):

    def __init__(self, alpha=1):
        super(GradientLoss, self).__init__()
        self.alpha = alpha

    @staticmethod
    def gradient(image):
        """Returns image gradients (dy, dx) for each color channel.

        Both output tensors have the same shape as the input: [batch_size, h, w,
        d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
        location (x, y). That means that dy will always have zeros in the last row,
        and dx will always have zeros in the last column.

        Arguments:
            image: Tensor with shape [batch_size, h, w, d].

        Returns:
            Pair of tensors (dy, dx) holding the vertical and horizontal image
            gradients (1-step finite difference).
  
        Raises:
            ValueError: If `image` is not a 4D tensor.
        """
        
        if image.dim() != 4:
            raise ValueError(
                'image_gradients expects a 4D tensor '
                '[batch_size, d, h, w], not %s.', image.shape)

        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        # gradient step=1
        left = image
        right = F.pad(image, [0, 1, 0, 0])[:, :, :, 1:]
        top = image
        bottom = F.pad(image, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0


        # ported tf implmentation
        
        # batch_size, depth, height, width,  = torch.unbind(image.shape)
        # dy = image[:,:, 1:, :] - image[:,:, :-1, :]
        # dx = image[:,:, :, 1:] - image[:,:, :, :-1]

        # # Return tensors with same size as original image by concatenating
        # # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        # shape = torch.stack([batch_size, 1, width, depth])
        # dy = torch.concat([dy, torch.zeros(shape, image.dtype)], 1)
        # dy = torch.reshape(dy, image.shape)

        # shape = torch.stack([batch_size, height, 1, depth])
        # dx = torch.concat([dx, torch.zeros(shape, image.dtype)], 2)
        # dx = torch.reshape(dx, image.shape)

        return dx, dy
    

    #loss = tf.reduce_mean(dx * dx + dy * dy, axis=axis)
    #return loss

    def masked_loss(self, gen_frames, gt_frames, mask):

        loss_per_img = torch.zeros(gen_frames.shape[0], device=gen_frames.device)

        for i in range(gen_frames.shape[0]):
            indices = torch.nonzero(mask[i])

            if len(indices) == 0:
                continue
            # get min and max values of mask
            min_x, min_y = torch.min(indices, dim=0).values
            max_x, max_y = torch.max(indices, dim=0).values

            max_x += 1
            max_y += 1

            # gradient
            loss_per_img[i] = self.gradient_loss(gen_frames[None,None,i,min_x:max_x, min_y:max_y], gt_frames[None,None,i,min_x:max_x, min_y:max_y])

        return torch.mean(loss_per_img)


    def gradient_loss(self, gen_frames, gt_frames):
        # gradient
        dx, dy = self.gradient(gen_frames - gt_frames)

        # grad_diff_x = torch.abs(gt_dx - gen_dx)
        # grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        return torch.mean(dx ** self.alpha + dy ** self.alpha)

    def forward(self, gen_frames, gt_frames, mask=None):
        if mask is None:
            return self.gradient_loss(gen_frames, gt_frames)
        else:
            return self.masked_loss(gen_frames, gt_frames, mask)
        

class MSELoss(_Loss):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, orig, mask=None):
        if mask is None:
            return self.mse_loss(pred, orig)
        else:
            return self.mse_loss(pred * mask.to(pred.device), orig * mask.to(orig.device))


        

class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputx, target, weight):
        # Typecast to long tensor --> labels are bytes initially (uint8),
        # index operations requiere LongTensor in pytorch

        target = target.to(device=target.device, dtype=torch.long)

        #target = target.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        #if inputx.is_cuda:
        #    target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        return total_loss, dice_val, ce_val

class ComissureLoss(nn.Module):

    def __init__(self, weight_seg = 1e-5, weight_loc = 1, weight_dist = 0):
        super(ComissureLoss, self).__init__()
        self.seg_loss = CombinedLoss()
        self.loc_loss = KLDivLoss(reduction='batchmean', log_target=True)
        self.dist_loss = MSELoss()
        self.weight_seg = weight_seg
        self.weight_loc = weight_loc
        self.weight_dist = weight_dist
    


    def forward(self, inputx, target, weight):
        """
        Args:
            inputx: N x C+1 x H x W
            target: N x 2H x W
            weight: N x H x W
        """

        # calculate segmentation loss
        input_seg = inputx[:, :-1, :, :]
        target_seg = target[:, :target.shape[1]//2, :] 
        
        seg_loss, dice_val, ce_val = self.seg_loss(input_seg, target_seg, weight)
        
        # calculate localisation loss
        input_loc = torch.select(inputx, 1, -1).float() # Assuming the last channel is the location map
        input_loc = torch.sigmoid(input_loc) # input_loc was in logit space. Sigmoid is its reverse function
        target_loc = target[:, target.shape[1]//2:, :].float()
        # KLDIVloss requires input to be probabilitie distributions (sum to 1) in logspace
        input_loc = torch.log_softmax(input_loc.view(input_loc.shape[0], -1), dim=1).view_as(input_loc)
        target_loc = torch.log_softmax(target_loc.view(target_loc.shape[0], -1), dim=1).view_as(target_loc)

        if torch.all(target_loc == 0): # empty slice. Would result in inf/nan total loss
            logger.warn("target_loc is 0")
            target_loc += 1e-5

        weight_loc = self.weight_loc
        assert (abs(torch.sum(torch.exp(input_loc), dim=(1,2))-1)<  2e-5).all(), f"input_loc is not a probability distribution in logspace"
        assert (abs(torch.sum(torch.exp(input_loc), dim=(1,2))-1)<  2e-5).all(), f"target_loc is not a probability distribution in logspace" 
        loc_loss = self.loc_loss(input_loc, target_loc)

        if loc_loss <= 0:
            logger.warn("loc_loss less than 0")
            loc_loss = 1e-5
            weight_loc = 0

        # calculate distance loss
        input_dist = torch.select(inputx, 1, -1).float() # Assuming the last channel is the location map
        target_dist = target[:, target.shape[1]//2:, :].float()

        input_dist = torch.sigmoid(input_dist) # input_dist was in logit space. Sigmoid is its reverse function
        input_dist = torch.clamp(input_dist, 0, 1)

        dist_loss = self.dist_loss(input_dist, target_dist)

        # calculate total loss
        #loc_loss = (1/loc_loss)
            
        total_loss = self.weight_seg * seg_loss + weight_loc * loc_loss + self.weight_dist * dist_loss
        
        return total_loss, seg_loss, dice_val, ce_val, loc_loss, dist_loss

def get_loss_func(cfg):
    if cfg.MODEL.LOSS_FUNC == 'combined':
        return CombinedLoss()
    elif cfg.MODEL.LOSS_FUNC == 'ce':
        return CrossEntropy2D()
    elif cfg.MODEL.LOSS_FUNC == "dice":
        return DiceLoss()
    elif cfg.MODEL.LOSS_FUNC == "localisation":
        return ComissureLoss(cfg.MODEL.WEIGHT_SEG, cfg.MODEL.WEIGHT_LOC, cfg.MODEL.WEIGHT_DIST)
    else:
        raise NotImplementedError(f"{cfg.MODEL.LOSS_FUNC}"
                                  f" loss function is not supported")
