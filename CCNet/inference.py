
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
import time
from typing import Optional, Dict, Tuple
import nibabel as nib

import numpy as np
import torch
from tqdm.contrib.logging import logging_redirect_tqdm # why is this not at the top?
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from CCNet.utils import logging
from CCNet.models.networks import build_model
from CCNet.data_loader.augmentation import ToTensorTest, CutoutTumorMaskInference
from CCNet.data_loader.data_utils import map_prediction_sagittal2full
from CCNet.data_loader.dataset import MultiScaleOrigDataThickSlices
from CCNet.config.global_var import get_class_names
from CCNet.utils.misc import calculate_centers_of_comissures


logger = logging.getLogger(__name__)


class Inference:

    permute_order: Dict[str, Tuple[int, int, int, int]]
    device: Optional[torch.device]
    default_device: torch.device

    def __init__(self, cfg, device: torch.device, ckpt: str = "", inference_weights: Optional[Dict[str, float]] = None):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        self.doing_inpainting = self.cfg.MODEL.MODEL_NAME == 'FastSurferPaint'
        self.doing_localisation = self.cfg.MODEL.MODEL_NAME == 'FastSurferLocalisation'
        
        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        self.default_device = device

        # Options for parallel run
        self.model_parallel = torch.cuda.device_count() > 1 and \
                              self.default_device.type == "cuda" and \
                              self.default_device.index is None

        # Initial model setup
        self.model = None
        self._model_not_init = None
        self.setup_model(cfg, device=self.default_device)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        self.alpha = {"sagittal": 0.6, "axial": 0.2, "coronal": 0.2} if inference_weights is None else inference_weights
        self.permute_order = {"axial": (3, 0, 2, 1), "coronal": (2, 3, 0, 1), "sagittal": (0, 3, 2, 1)}
        self.post_prediction_mapping_hook = {"sagittal": map_prediction_sagittal2full, "axial": map_prediction_sagittal2full, "coronal": map_prediction_sagittal2full}
        self.alpha_loc = {"sagittal": 1.0, "axial": 0.0, "coronal": 0.0}

        # Initial checkpoint loading
        if ckpt:
            # this also moves the model to the para
            self.load_checkpoint(ckpt)

        self._debug = False

    def setup_model(self, cfg=None, device: torch.device = None):
        if cfg is not None:
            self.cfg = cfg
        if device is None:
            device = self.default_device

        # Set up model
        self._model_not_init = build_model(self.cfg)  # ~ model = CCNet(params_network)
        self._model_not_init.to(device)
        self.device = None

    def set_cfg(self, cfg):
        self.cfg = cfg

    def to(self, device: Optional[torch.device] = None):
        if self.model_parallel:
            raise RuntimeError("Moving the model to other devices is not supported for multi-device models.")
        _device = self.default_device if device is None else device
        self.device = _device
        self.model.to(device=_device)

    def load_checkpoint(self, ckpt):
        logger.info("Loading checkpoint {}".format(ckpt))

        self.model = self._model_not_init
        # If device is None, the model has never been loaded (still in random initial configuration)
        if self.device is None:
            self.device = self.default_device

        # workaround for mps (directly loading to map_location=mps results in zeros)
        device = self.device
        if self.device.type == 'mps':
            self.model.to('cpu')
            device = 'cpu'
        else:
            # make sure the model is, where it is supposed to be
            self.model.to(self.device)

        model_state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(model_state['model_state'])

        # workaround for mps (move the model back to mps)
        if self.device.type == 'mps':
            self.model.to(self.device)

        if self.model_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def get_modelname(self):
        return self.model_name

    def get_cfg(self):
        return self.cfg

    def get_num_classes(self):
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self):
        return self.cfg.DATA.PLANE

    def get_model_height(self):
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self):
        return self.cfg.MODEL.WIDTH

    def get_max_size(self):
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self):
        return self.device

    @torch.no_grad()
    def eval(self, init_pred: torch.Tensor, 
             val_loader: DataLoader, 
             *, out_scale=None, 
             out: Optional[torch.Tensor] = None, 
             localisation : Optional[torch.Tensor]  = None, 
             out_localisation : Optional[torch.Tensor] = None, 
             init_pred_localisation : Optional[torch.Tensor] = None,
             sdir : Optional[str] = None):
        """Perform prediction and inplace-aggregate views into pred_prob. Return pred_prob.
        
        Args:
            init_pred: Initial prediction (e.g. from the segmentation network)
            val_loader: DataLoader for the test data
            out_scale: Output scale factor
            out: Output tensor to add the predictions to

        Returns:
            out: Output tensor with the predictions added
        
        """
        self.model.eval()
        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
            logger.warning("The Validation loader seems to not use the SequentialSampler. This might interfere with "
                           "the assumed sorting of batches.")

        start_index = 0
        plane = self.cfg.DATA.PLANE
        index_of_current_plane = self.permute_order[plane].index(0)
        target_shape = init_pred.shape
        ii = [slice(None) for _ in range(4)]
        pred_ii = tuple(slice(i) for i in target_shape[:3])

        if out is None:
            out = init_pred.detach().clone()

        if self.doing_inpainting and localisation is None:
            localisation = torch.zeros([*target_shape[:3], 1], dtype=float, device=out.device)

        if self.doing_localisation and out_localisation is None:
            if init_pred_localisation is not None:
                out_localisation = init_pred_localisation.detach().clone()
            else:
                out_localisation = torch.zeros([*target_shape[:3], 1], dtype=float, device=out.device)


        assert not (self.doing_localisation and out_localisation is None), "Localisation is only possible if the model is a localisation model"

        batch_idx = 0 # prevent error in throws
        with logging_redirect_tqdm():
            try:
                for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), unit="batch"):

                    # move data to the model device
                    images, scale_factors = batch['image'].to(self.device), batch['scale_factor'].to(self.device)

                    # predict the current batch, outputs logits
                        

                    pred = self.model(images, scale_factors, out_scale)
                    if self.doing_inpainting:
                        output_slice = pred[1]
                        pred = pred[0]

                    elif self.doing_localisation:
                        loc_pred = pred[:, -1:, :, :]
                        pred = pred[:, :-1, :, :]


                    # if not np.sum(output_slice.shape) == 256*2+2:
                    #     import pdb; pdb.set_trace()

                    batch_size = pred.shape[0]
                    end_index = start_index + batch_size

                    # check if we need a special mapping (e.g. as for sagittal)
                    if self.post_prediction_mapping_hook.get(plane) is not None:
                        pred = self.post_prediction_mapping_hook.get(plane)(pred, num_classes=self.get_num_classes())

                    # permute the prediction into the out slice order
                    pred = pred.permute(*self.permute_order[plane]).to(out.device)  # the to-operation is implicit

                    # cut prediction to the image size
                    pred = pred[pred_ii]

                    # add prediction logits into the output (same as multiplying probabilities)
                    ii[index_of_current_plane] = slice(start_index, end_index)

                    
                    out[tuple(ii)].add_(pred, alpha=self.alpha.get(plane))

                    if self.doing_inpainting:

                        # if plane == 'sagittal':
                        #     import pdb; pdb.set_trace()
                        #output_slice = self.post_predition_mapping_hook.get(plane, lambda x: x)(output_slice)
                        output_slice = output_slice.permute(*self.permute_order[plane]).to(out.device)
                        output_slice = output_slice[pred_ii]
                        try:
                            localisation[tuple(ii)[:3]].add_(output_slice)
                        except:
                            print(output_slice.shape)
                            print(localisation[tuple(ii)[:3]].shape)
                            import pdb; pdb.set_trace()
                    
                    # if doing localisation, add the localisation prediction
                    if self.doing_localisation:
                        # permute the prediction into the out slice order
                        loc_pred = loc_pred.permute(*self.permute_order[plane]).to(out.device)  # the to-operation is implicit

                        # cut prediction to the image size
                        loc_pred = loc_pred[pred_ii]

                        out_localisation[tuple(ii)].add_(loc_pred, alpha=self.alpha_loc.get(plane))


                        if self._debug:
                            # debug save
                            assert batch_size == 1, "Debug save only works for batch size 1"

                            orig_image = nib.load(f'{sdir}/mri/orig.mgz')
                            max_value = np.max(orig_image.get_fdata())

                            props = (loc_pred - loc_pred.min()) / (loc_pred.max() - loc_pred.min())  # Normalize values between 0 and 1
                            props = props * max_value  # Scale values to [0, 255]
                            props= torch.round(props)

                            localisation = torch.zeros_like(out_localisation)
                            localisation[tuple(ii)].add_(props, alpha=self.alpha_loc.get(plane))

                            #print(torch.unique(props))
                            localisation_img = nib.MGHImage(localisation.cpu()[:,:,:,0], orig_image.affine, orig_image.header)
                            nib.save(localisation_img, f'{sdir}/mri/localisation_{plane}_{batch_idx}.mgz')

                            
                            # permute the image into the out slice order
                            images = images.permute(1,3,2,0).to(out.device)  # the to-operation is implicit

                            # cut image to the image size
                            #images = images[pred_ii].cpu()
                            images = images.cpu() * 255
                            images = torch.round(images)
                            
                            print(torch.unique(images))
                            input_img = nib.MGHImage(images[:,:,:,0], orig_image.affine, orig_image.header)
                            nib.save(input_img, f'{sdir}/mri/img_{plane}_{batch_idx}.mgz')


                            #AC_PC = calculate_centers_of_comissures(loc_pred)
                            #with open(f'{sdir}/mri/AC_PC_{plane}.txt', 'a') as f:
                            #    f.write(f'{batch_idx},{AC_PC[0]},{AC_PC[1]}\n')
                            
                        

                    start_index = end_index

            except:
                logger.exception("Exception in batch {} of {} inference.".format(batch_idx, plane))
                raise
            else:
                logger.info("Inference on {} batches for {} successful".format(batch_idx+1, plane))
        
        if self.doing_localisation:
            return out, out_localisation

        if self.doing_inpainting:
            return out, localisation.squeeze()
        else:
            return out

    @torch.no_grad()
    def run(self, 
            init_pred: torch.Tensor, 
            img_filename, 
            orig_data, 
            orig_zoom,
            out: Optional[torch.Tensor] = None, 
            out_res = None, 
            lesion_mask = None, 
            batch_size: Optional[int] = None,
            out_localisation : Optional[torch.Tensor] = None,
            init_pred_localisation : Optional[torch.Tensor] = None,
            sdir : Optional[str] = None,
            pad : bool = False):
        """Run the loaded model on the data (T1) from orig_data and filename img_filename with scale factors orig_zoom."""
        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(img_filename, orig_data, orig_zoom, self.cfg, lesion_mask=lesion_mask,
                                                     transforms=transforms.Compose([ToTensorTest(include=['image','cutout_mask']), CutoutTumorMaskInference()]),
                                                     pad = pad)

        test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                      batch_size=self.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size)

        # Run evaluation
        start = time.time()

        out = self.eval(init_pred, test_data_loader, out=out, out_scale=out_res, out_localisation=out_localisation, init_pred_localisation=init_pred_localisation, sdir=sdir)
        time_delta = time.time() - start
        logger.info(f"{self.cfg.DATA.PLANE.capitalize()} inference on {img_filename} finished in "
                    f"{time_delta:0.4f} seconds")
        
        return out

