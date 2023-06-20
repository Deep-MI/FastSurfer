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
import yacs.config
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from typing import Optional, Union
from numbers import Real


class DiceLoss(_Loss):
    """
    Dice Loss

    Methods:
        forward: Calulates the DiceLoss
    """

    def forward(
            self,
            output: Tensor,
            target: Tensor,
            weights: Optional[int] = None,
            ignore_index: Optional[int] = None
    ) -> float:
        """ Calulates the DiceLoss

        Parameter:
            output: N x C x H x W Variable
            target: N x C x W LongTensor with starting class at 0
            weights: C FloatTensor with class wise weights. Defaults to None
            ignore_index: ignore label with index x in the loss calculation. Defaults to None

        Returns:
            Calculated Diceloss
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
        loss_per_channel = weights * (
            1 - (numerator / denominator)
        )  # Channel-wise weights

        return loss_per_channel.sum() / output.size(1)


class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood

    Attributes:
        nll_loss: calculated cross-entropy loss

    Methods:
        forward: returns calculated cross entropy
    """

    def __init__(self, weight: Optional[Tensor] =None, reduction: str = "none"):
        """
        Initialization of CrossEntropy2D

        Args:
            weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size `C`. Defaults to None
            reduction: Specifies the reduction to apply to the output, as in nn.CrossEntropyLoss. Defaults to 'None'
        """

        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(
            f"Initialized {self.__class__.__name__} with weight: {weight} and reduction: {reduction}"
        )

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor

    Attributes:

        cross_entropy_loss: Results of cross entropy loss
        dice_loss: Results of dice loss
        weight_dice: Weight for dice loss
        weight_ce: Weight for float
    """

    def __init__(self, weight_dice: Real = 1, weight_ce: Real = 1):
        """ Initialization of CobinedLoss

        Args:
            weight_dice: Weight for dice loss. Defaults to 1
            weight_ce:  Weight for cross entropy loss. Defaults to 1
        """

        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(
            self,
            inputx: Tensor,
            target: Tensor,
            weight: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ [help]

        Args:
            inputx: A Tensor of shape N x C x H x W containing  the input x values
            target: A Tensor of shape N x H x W of integers containing the target
            weight: A Tensor of shape N x H x W of floats containg the weights

        Returns:
           total loss, dice loss and cross entropy value
        """

        # Typecast to long tensor --> labels are bytes initially (uint8),
        # index operations require LongTensor in pytorch
        target = target.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        if inputx.is_cuda:
            target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(
            torch.mul(self.cross_entropy_loss.forward(inputx, target), weight)
        )
        total_loss = torch.add(
            torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce)
        )

        return total_loss, dice_val, ce_val


def get_loss_func(
        cfg: yacs.config.CfgNode
) -> Union[CombinedLoss, CrossEntropy2D, DiceLoss]:
    """ Gives a default object of the loss function

    Args:
        cfg: configuration node, containing searched loss function.
            The model loss function can either be 'combined', 'ce' or 'dice'

    Returns:
        an object of the class corresponding to the loss function

    Raises:
        NotImplementedError: Requested loss function is not implemented
    """

    if cfg.MODEL.LOSS_FUNC == "combined":
        return CombinedLoss()
    elif cfg.MODEL.LOSS_FUNC == "ce":
        return CrossEntropy2D()
    elif cfg.MODEL.LOSS_FUNC == "dice":
        return DiceLoss()
    else:
        raise NotImplementedError(
            f"{cfg.MODEL.LOSS_FUNC}" f" loss function is not supported"
        )
