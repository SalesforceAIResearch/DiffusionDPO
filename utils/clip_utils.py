# just hps_utils w/o clip loading and no model_dict

import hpsv2
import os 
root_path = "/export/share/bwallace/hps/"
import requests
from clint.textui import progress
from PIL import Image
import torch

from .tokenizer_hps import HFTokenizer 
from .open_clip import create_model_and_transforms, get_tokenizer



# device = 'cuda'
HF_HUB_PREFIX = 'hf-hub:'


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.model, preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False, # not sure what this means but seems to work without
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )


        self.tokenizer = get_tokenizer('ViT-H-14')


    def score(self, img_path, prompt):
        if isinstance(img_path, list):
            result = []
            for one_img_path in img_path:
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    if isinstance(one_img_path, str):
                        image = self.preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                    elif isinstance(one_img_path, Image.Image):
                        image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    # Process the prompt
                    text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                    # Calculate the clip score
                    # NOTE: https://github.com/mlfoundations/open_clip/issues/484
                    if True: # with eval(f'torch.{self.device}.amp.autocast()'): # cuda or cpu
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        clip_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(clip_score[0])    
            return result
        elif isinstance(img_path, str):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the clip score
                if True: # with eval(f'torch.{self.device}.amp.autocast()'): # found not needed
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    clip_score = torch.diagonal(logits_per_image).cpu().numpy()
            return [clip_score[0]]

