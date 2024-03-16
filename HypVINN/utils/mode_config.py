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


def get_hypinn_mode_config(args):

    LOGGER.info('Setting up input mode...')
    if hasattr(args, 't1') and hasattr(args, 't2'):
        if os.path.isfile(str(args.t1)) and os.path.isfile(str(args.t2)):
            args.mode = 'multi'
        elif os.path.isfile(str(args.t1)):
            args.mode ='t1'
            args.t2 = None
        elif os.path.isfile(str(args.t2)):
            args.mode ='t2'
            args.t1 = None
            LOGGER.info('Warning: T2 mode selected. Only passing a T2 image can generate not so accurate results.\n '
                        'Best results are obtained when a T2 image is accompanied with a T1 image.')
        else:
            args.mode= None

    elif hasattr(args, 't1'):
        if os.path.isfile(str(args.t1)):
            args.mode = 't1'
            args.t2 = None
        else:
            if hasattr(args,'t2'):
                if os.path.isfile(str(args.t2)):
                    args.mode = 't2'
                    args.t1 = None
                    LOGGER.info(
                        'Warning: T2 mode selected. Only passing a T2 image can generate not so accurate results.\n '
                        'Best results are obtained when a T2 image is accompanied with a T1 image.')
                else:
                    args.mode = None
            else:
                args.mode = None
    elif hasattr(args,'t2'):
        if os.path.isfile(str(args.t2)):
            args.mode = 't2'
            args.t1 = None
            LOGGER.info('Warning: T2 mode selected. Only passing a T2 image can generate not so accurate results.\n '
                    'Best results are obtained when a T2 image is accompanied with a T1 image.')
        else:
            args.mode = None
    else:
        args.mode = None

    if args.mode:
        LOGGER.info('HypVINN mode is setup to {} input mode'.format(args.mode))

    return args



