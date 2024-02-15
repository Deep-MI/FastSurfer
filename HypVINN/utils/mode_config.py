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
def get_hypinn_mode_config(args):

    mode = args.mode

    if mode == 'auto':
        if hasattr(args, 't1') and hasattr(args, 't2'):
            if os.path.isfile(args.t1) and os.path.isfile(args.t2):
                args.mode = 'multi'
            elif os.path.isfile(args.t1):
                args.mode ='t1'
                args.t2 = None
            elif os.path.isfile(args.t2):
                args.mode = 't2'
                args.t1 = None
            else:
                raise FileNotFoundError('No t1 or t2 image found')
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

    return args



