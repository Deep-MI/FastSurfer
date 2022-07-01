
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


# Building Blocks
class InputDenseBlock(nn.Module):
    def __init__(self, params):
        super(InputDenseBlock, self).__init__()
        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = params["num_channels"]
        conv1_in_size = params["num_filters_interpol"]
        conv2_in_size = params["num_filters_interpol"]
        out_size = params["num_filters_interpol_last"] if "num_filters_interpol_last" in params else params["num_filters_interpol"]

        # learnable layers
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params["num_filters_interpol"],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params["num_filters_interpol"],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv2 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params["num_filters_interpol"],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # D \times D convolution for the last block --> with maxout this is redundant unless we want to reduce
        # the number of filter maps here compared to conv1
        self.conv3 = nn.Conv2d(in_channels=conv2_in_size, out_channels=out_size,
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))


        self.bn0 = nn.BatchNorm2d(params["num_channels"])
        self.gn1 = nn.BatchNorm2d(conv1_in_size)
        self.gn2 = nn.BatchNorm2d(conv2_in_size)
        self.gn3 = nn.BatchNorm2d(conv2_in_size)
        self.gn4 = nn.BatchNorm2d(out_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x):
        # Input batch normalization
        x0_bn = self.bn0(x)

        # Convolution block1 (RF: 3x3)
        x0 = self.conv0(x0_bn)
        x1_gn = self.gn1(x0)
        x1 = self.prelu(x1_gn)

        # Convolution block2 (RF: 5x5)
        x1 = self.conv1(x1)
        x2_gn = self.gn2(x1)

        # First Maxout
        x1_gn = torch.unsqueeze(x1_gn, 4) # RF 3x3 --> weighted with attention map 1
        x2_gn = torch.unsqueeze(x2_gn, 4)  # RF 5x5 --> weighted with attention map 2
        x2 = torch.cat((x2_gn, x1_gn), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block 3 (RF: 7x7)
        x2 = self.conv2(x2)
        x3_gn = self.gn3(x2)

        # Second Maxout
        x3_gn = torch.unsqueeze(x3_gn, 4) # RF 7x7 --> weighted with attention map 3
        x2_max = torch.unsqueeze(x2_max, 4)  # RF 3x3 and 5x5 from First Maxout (weighted with map 1 and 2)
        x3 = torch.cat((x3_gn, x2_max), dim=4)  # Concatenating along the 5th dimension
        x3_max, _ = torch.max(x3, 4)
        x3 = self.prelu(x3_max)

        # Convolution block 4 (RF: 9x9)
        x3 = self.conv3(x3)
        out = self.gn4(x3)

        return out


class CompetitiveDenseBlock(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
               }
    """

    def __init__(self, params, outblock=False):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        :param bool outblock: Flag indicating if last block (before classifier block) is set up.
                               Default: False
        :return None:
        """
        super(CompetitiveDenseBlock, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_channels'])  # num_channels
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])
        out_size = params["num_filters_last"] if "num_filters_last" in params else params["num_filters"]

        # Define the learnable layers
        # Standard conv layers
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv2 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # D \times D convolution for the last block
        self.conv3 = nn.Conv2d(in_channels=conv2_in_size, out_channels=out_size,
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn4 = nn.BatchNorm2d(num_features=out_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter
        self.outblock = outblock

    def forward(self, x):
        """
        CompetitiveDenseBlock's computational Graph
        {in (Conv - BN from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> out
        end with batch-normed output to allow maxout across skip-connections

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # Activation from pooled input
        x0 = self.prelu(x)

        # Convolution block 1 (RF: 3x3)
        x0 = self.conv0(x0)
        x1_bn = self.bn1(x0)

        # First Maxout/Addition
        x0_bn = torch.unsqueeze(x, 4) # Original input --> weighted with attention map 1
        x1_bn = torch.unsqueeze(x1_bn, 4) # RF 3x3 --> weighted with attention map 2
        x1 = torch.cat((x1_bn, x0_bn), dim=4)  # Concatenate along the 5th dimension NB x C x H x W x F
        x1_max, _ = torch.max(x1, 4)
        x1 = self.prelu(x1_max)

        # Convolution block 2
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)

        # Second Maxout/Addition
        x2_bn = torch.unsqueeze(x2_bn, 4) # RF 5x5 --> weighted with attention map 3
        x1_max = torch.unsqueeze(x1_max, 4) # Original and 3x3 weighted with attention map 1 and 2
        x2 = torch.cat((x2_bn, x1_max), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block 3
        x2 = self.conv2(x2)
        x3_bn = self.bn3(x2)

        # Third Maxout/Addition
        x3_bn = torch.unsqueeze(x3_bn, 4) # RF 7x7 --> weighted with attention map 4
        x2_max = torch.unsqueeze(x2_max, 4) # orig, 3x3, 5x5 weighted with attention map 1-3
        x3 = torch.cat((x3_bn, x2_max), dim=4)  # Concatenating along the 5th dimension
        x3_max, _ = torch.max(x3, 4)
        x3 = self.prelu(x3_max)

        # Convolution block 4 (end with batch-normed output to allow maxout across skip-connections)
        out = self.conv3(x3)

        if not self.outblock:
            out = self.bn4(out)

        return out


class CompetitiveDenseBlockInput(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU for input

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
              }
    """

    def __init__(self, params):
        """
        Constructor to initialize the Competitive Dense Block
        :param dict params: dictionary with parameters specifiying block architecture
        """
        super(CompetitiveDenseBlockInput, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_channels'])
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv2 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # 1 \times 1 convolution for the last block
        self.conv3 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.bn0 = nn.BatchNorm2d(num_features=conv0_in_size)
        self.bn1 = nn.BatchNorm2d(num_features=conv1_in_size)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn3 = nn.BatchNorm2d(num_features=conv2_in_size)
        self.bn4 = nn.BatchNorm2d(num_features=conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x):
        """
        CompetitiveDenseBlockInput's computational Graph
        in -> BN -> {Conv -> BN -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} -> {Conv -> BN} -> out

        :param tensor x: input tensor (image or feature map)
        :return tensor out: output tensor (processed feature map)
        """
        # Input batch normalization
        x0_bn = self.bn0(x)

        # Convolution block1 (RF: 3x3)
        x0 = self.conv0(x0_bn)
        x1_bn = self.bn1(x0)
        x1 = self.prelu(x1_bn)

        # Convolution block2 (RF: 5x5)
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)

        # First Maxout
        x1_bn = torch.unsqueeze(x1_bn, 4) # RF 3x3
        x2_bn = torch.unsqueeze(x2_bn, 4)  # RF 5x5
        x2 = torch.cat((x2_bn, x1_bn), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        x2 = self.prelu(x2_max)

        # Convolution block3 (RF: 7x7)
        x2 = self.conv2(x2)
        x3_bn = self.bn3(x2)

        # Second Maxout
        x3_bn = torch.unsqueeze(x3_bn, 4) # RF 7x7
        x2_max = torch.unsqueeze(x2_max, 4)  # RF 3x3 and 5x5 from First Maxout (weighted with map 1 and 2)
        x3 = torch.cat((x3_bn, x2_max), dim=4)  # Concatenating along the 5th dimension
        x3_max, _ = torch.max(x3, 4)
        x3 = self.prelu(x3_max)

        # Convolution block 4 (RF: 9x9)
        out = self.conv3(x3)
        out = self.bn4(out)

        return out


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, device="cuda"):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).to(device)
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

##
# Encoder/Decoder definitions
##
class CompetitiveEncoderBlock(CompetitiveDenseBlock):
    """
    Encoder Block = CompetitiveDenseBlock + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)  # For Unpooling later on with the indices

    def forward(self, x):
        """
        CComputational graph for Encoder Block:
          * CompetitiveDenseBlock
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        out_block = super(CompetitiveEncoderBlock, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder, indices = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block, indices


class CompetitiveEncoderBlockInput(CompetitiveDenseBlockInput):
    """
    Encoder Block = CompetitiveDenseBlockInput + Max Pooling
    """

    def __init__(self, params):
        """
        Encoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(CompetitiveEncoderBlockInput, self).__init__(params)  # The init of CompetitiveDenseBlock takes in params
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)  # For Unpooling later on with the indices

    def forward(self, x):
        """
        Computational graph for Encoder Block:
          * CompetitiveDenseBlockInput
          * Max Pooling (+ retain indices)

        :param tensor x: feature map from previous block
        :return: original feature map, maxpooled feature map, maxpool indices
        """
        out_block = super(CompetitiveEncoderBlockInput, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder, indices = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block, indices


class CompetitiveDecoderBlock(CompetitiveDenseBlock):
    """
    Decoder Block = (Unpooling + Skip Connection) --> Dense Block
    """

    def __init__(self, params, outblock=False):
        """
        Decoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        :param bool outblock: Flag, indicating if last block of network before classifier
                              is created. Default: False
        """
        super(CompetitiveDecoderBlock, self).__init__(params, outblock=outblock)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, x, out_block, indices):
        """
        Computational graph Decoder block:
          * Unpooling of feature maps from lower block
          * Maxout combination of unpooled map + skip connection
          * Forwarding toward CompetitiveDenseBlock

        :param tensor x: input feature map from lower block (gets unpooled and maxed with out_block)
        :param tensor out_block: skip connection feature map from the corresponding Encoder
        :param tensor indices: indices for unpooling from the corresponding Encoder (maxpool op)
        :return: processed feature maps
        """
        unpool = self.unpool(x, indices)
        unpool = torch.unsqueeze(unpool, 4)

        out_block = torch.unsqueeze(out_block, 4)
        concat = torch.cat((unpool, out_block), dim=4)  # Competitive Concatenation
        concat_max, _ = torch.max(concat, 4)
        out_block = super(CompetitiveDecoderBlock, self).forward(concat_max)

        return out_block


class OutputDenseBlock(nn.Module):
    """
    Dense Ooutput Block = (Upinterpolated + Skip Connection) --> Semi Competitive Dense Block
    """

    def __init__(self, params):
        """
        Decoder Block initialization
        :param dict params: parameters like number of channels, stride etc.
        :param bool outblock: Flag, indicating if last block of network before classifier
                              is created. Default: False
        """
        super(OutputDenseBlock, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_channels'])  # num_channels
        conv1_in_size = int(params['num_filters'])
        conv2_in_size = int(params['num_filters'])

        # Define the learnable layers
        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv1 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.conv2 = nn.Conv2d(in_channels=conv1_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        # D \times D convolution for the last block
        self.conv3 = nn.Conv2d(in_channels=conv2_in_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.gn1 = nn.BatchNorm2d(conv1_in_size)
        self.gn2 = nn.BatchNorm2d(conv2_in_size)
        self.gn3 = nn.BatchNorm2d(conv2_in_size)
        self.gn4 = nn.BatchNorm2d(conv2_in_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x, out_block):
        """
        Computational graph Output block
          * Maxout combination of unpooled map from previous block + skip connection
          * Forwarding toward CompetitiveDenseBlock

        :param tensor x: up-interpolated input feature map from lower block (gets maxed with out_block)
        :param tensor out_block: skip connection feature map from the corresponding Encoder
        :param tensor indices: indices for unpooling from the corresponding Encoder (maxpool op)
        :return: processed feature maps
        """
        # Concatenation along channel (different number of channels from decoder and skip connection)
        concat = torch.cat((x, out_block), dim=1)

        # Activation from pooled input
        x0 = self.prelu(concat)

        # Convolution block1 (no maxout here), could optionally add dense connection; 3x3
        x0 = self.conv0(x0)
        x1_gn = self.gn1(x0)
        x1 = self.prelu(x1_gn)

        # Convolution block2; 5x5
        x1 = self.conv1(x1)
        x2_gn = self.gn2(x1)

        # First Maxout
        x1_gn = torch.unsqueeze(x1_gn, 4)
        x2_gn = torch.unsqueeze(x2_gn, 4)  # Add Singleton Dimension along 5th
        x2 = torch.cat((x2_gn, x1_gn), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)

        x2 = self.prelu(x2_max)
        # Convolution block3; 7x7
        x2 = self.conv2(x2)
        x3_gn = self.gn3(x2)

        # Second Maxout
        x3_gn = torch.unsqueeze(x3_gn, 4)
        x2_max = torch.unsqueeze(x2_max, 4)  # Add Singleton Dimension along 5th
        x3 = torch.cat((x3_gn, x2_max), dim=4)  # Concatenating along the 5th dimension
        x3_max, _ = torch.max(x3, 4)

        x3 = self.prelu(x3_max)
        # Convolution block 4; 9x9
        out = self.conv3(x3)
        out = self.gn4(out)

        return out


class ClassifierBlock(nn.Module):
    """
    Classification Block
    """
    def __init__(self, params):
        """
        Classifier Block initialization
        :param dict params: parameters like number of channels, stride etc.
        """
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_classes'], params['kernel_c'],
                              params['stride_conv'])  # To generate logits

    def forward(self, x):
        """
        Computational graph of classifier
        :param tensor x: output of last CompetitiveDenseDecoder Block-
        :return: logits
        """
        logits = self.conv(x)

        return logits
