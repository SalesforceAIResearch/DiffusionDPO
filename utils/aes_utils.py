# just hps_utils w/o clip loading and no model_dict

import os 
root_path = "./hps/"
import requests
from clint.textui import progress
from PIL import Image
import torch

from .tokenizer_hps import HFTokenizer 


# import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms

from os.path import join

# import clip
from transformers import AutoProcessor, AutoModel


# if you changed the MLP architecture during training, change it also here:
class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Selector():
    
    def __init__(self, device):
        self.device = device

        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        s = torch.load("aesthetics_model/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)

        self.model.to(device)
        self.model.eval()

        clip_model_name = "openai/clip-vit-large-patch14"
        self.model2 = AutoModel.from_pretrained(clip_model_name).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(clip_model_name)
        # self.model2, self.preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

    def score(self, images, prompt_not_used):
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.model2.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
            return self.model(image_embs).cpu().flatten().tolist()
