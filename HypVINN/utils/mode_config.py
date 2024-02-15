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

import os
from FastSurferCNN.utils import logging

LOGGER = logging.get_logger(__name__)


def assert_image_inputs(t1_path,t2_path,mode):

    if mode == 't1':
        t1_mode = True
        t2_mode = False
    elif mode == 't2':
        t1_mode = False
        t2_mode = True
    else:
        t1_mode = True
        t2_mode = True
    
    if t1_mode:
        assert os.path.isfile(t1_path), f"T1 image not found"
        t1 = nib.load(t1_path)
        t1 = nib.as_closest_canonical(t1)
        t1_zoom = t1.header.get_zooms()
        t1_size = np.asarray(t1.header.get_data_shape())
    if t2_mode:
        assert os.path.isfile(t2_path), f"T2 image not found"
        t2 = nib.load(t2_path)
        t2 = nib.as_closest_canonical(t2)
        t2_zoom = t2.header.get_zooms()
        t2_size = np.asarray(t2.header.get_data_shape())


    if t1_mode and t2_mode:
        assert np.allclose(np.array(t1_zoom), np.array(t2_zoom),
                           rtol=0.05), "T1 : {} and T2 : {} images have different resolutions".format(t1_zoom,
                                                                                                      t2_zoom)
        assert np.allclose(np.array(t1_size), np.array(t2_size),
                           rtol=0.05), "T1 : {} and T2 : {} images have different size".format(t1_size, t2_size)

    LOGGER.info('Input data is available for the choosen HypVINN input mode')



def get_hypinn_mode_config(args):

    mode = args.mode

    if mode == 'auto':
        LOGGER.info('HypVINN mode is "auto", setting up input mode...')
        if hasattr(args, 't1') and hasattr(args, 't2'):
            if args.t1 and args.t2:
                args.mode = 'multi'
            elif args.t1:
                args.mode ='t1'
                args.t2 = None
            else:
                args.mode = 't2'
                args.t1 = None

        elif hasattr(args, 't1'):
            args.mode = 't1'
            args.t2 = None
        else:
            args.mode = 't2'
            args.t1 = None

    elif mode == 'multi':
        pass
    elif mode == 't1':
        args.t2 = None
    elif mode == 't2':
        args.t1 = None
    
    LOGGER.info('HypVINN mode is setup to {} input mode'.format(args.mode))
    LOGGER.info('Checking input data......')

    assert_image_inputs(t1_path=args.t1,t2_path=args.t2,mode =args.mode)

    return args



