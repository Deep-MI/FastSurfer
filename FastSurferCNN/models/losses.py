
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
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class DiceLoss(_Loss):
    """
    Dice Loss
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        :param torch.Tensor output: N x C x H x W Variable
        :param torch.Tensor target: N x C x W LongTensor with starting class at 0
        :param torch.Tensor weights: C FloatTensor with class wise weights
        :param int ignore_index: ignore label with index x in the loss calculation
        :return: torch.Tensor loss: FloatTensor with class wise weights
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

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


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
        target = target.type(torch.LongTensor)  # Typecast to long tensor
        if inputx.is_cuda:
            target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        return total_loss, dice_val, ce_val
