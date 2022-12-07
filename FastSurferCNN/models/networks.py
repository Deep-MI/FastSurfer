
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
import torch.nn as nn
import numpy as np

import FastSurferCNN.models.sub_module as sm
import FastSurferCNN.models.interpolation_layer as il


class FastSurferCNNBase(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params, padded_size=256):
        super(FastSurferCNNBase, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        params["num_filters_last"] = params["num_filters"]
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, scale_factor=None, scale_factor_out=None):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        return decoder_output1


class FastSurferCNN(FastSurferCNNBase):
    def __init__(self, params, padded_size):
        super(FastSurferCNN, self).__init__(params)
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, scale_factor=None, scale_factor_out=None):
        """

        :param x: [N, C, H, W]
        :param scale_factor: [N, 1]
        :return:
        """
        net_out = super().forward(x, scale_factor)
        output = self.classifier.forward(net_out)

        return output


class FastSurferVINN(FastSurferCNNBase):
    """
    Network Definition of Fully Competitive Network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params, padded_size=256):
        num_c = params["num_channels"]
        params["num_channels"] = params["num_filters_interpol"]
        super(FastSurferVINN, self).__init__(params)

        # Flex options
        self.height = params['height']
        self.width = params['width']

        self.out_tensor_shape = tuple(params.get('out_tensor_' + k, padded_size) for k in ['width', 'height'])

        self.interpolation_mode = params['interpolation_mode'] if 'interpolation_mode' in params else 'bilinear'
        if self.interpolation_mode not in ['nearest', 'bilinear', 'bicubic', 'area']:
            raise ValueError('Invalid interpolation mode')

        self.crop_position = params['crop_position'] if 'crop_position' in params else 'top_left'
        if self.crop_position not in ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']:
            raise ValueError('Invalid crop position')

        # Reset input channels to original number (overwritten in super call)
        params["num_channels"] = num_c

        self.inp_block = sm.InputDenseBlock(params)

        params['num_channels'] = params['num_filters'] + params['num_filters_interpol']
        self.outp_block = sm.OutputDenseBlock(params)

        self.interpol1 = il.Zoom2d((self.width, self.height),
                                   interpolation_mode=self.interpolation_mode,
                                   crop_position=self.crop_position)

        self.interpol2 = il.Zoom2d(self.out_tensor_shape,
                                   interpolation_mode=self.interpolation_mode,
                                   crop_position=self.crop_position)

        # Classifier logits options
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, scale_factor, scale_factor_out=None):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        # Input block + Flex to 1 mm
        skip_encoder_0 = self.inp_block(x)
        encoder_output0, rescale_factor = self.interpol1(skip_encoder_0, scale_factor)

        # FastSurferCNN Base
        decoder_output1 = super().forward(encoder_output0, scale_factor=scale_factor)

        # Flex to original res
        if scale_factor_out is None:
            scale_factor_out = rescale_factor
        else:
            scale_factor_out = np.asarray(scale_factor_out) * np.asarray(rescale_factor) / np.asarray(scale_factor)

        prior_target_shape = self.interpol2.target_shape
        self.interpol2.target_shape = skip_encoder_0.shape[2:]
        try:
            decoder_output0, sf = self.interpol2(decoder_output1, scale_factor_out, rescale=True)
        finally:
            self.interpol2.target_shape = prior_target_shape
        outblock = self.outp_block(decoder_output0, skip_encoder_0)

        # Final logits layer
        logits = self.classifier.forward(outblock)  # 1x1 convolution

        return logits


_MODELS = {
    "FastSurferCNN": FastSurferCNN,
    "FastSurferVINN": FastSurferVINN,
}


def build_model(cfg):
    assert (cfg.MODEL.MODEL_NAME in _MODELS.keys()),\
        f"Model {cfg.MODEL.MODEL_NAME} not supported"
    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    model = _MODELS[cfg.MODEL.MODEL_NAME](params, padded_size=cfg.DATA.PADDED_SIZE)
    return model
