# -*- coding: utf-8 -*-
"""Download ISIC2018.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hATNECzcCb688vucXPGQs2AJOQahqfuo
"""

from extra.utils import (
  load_config,
  _print,
)
import json
import importlib
from datasets import (
  ISIC2018Dataset
)
from .models.unet import UNET


# # training
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
# !unzip -q ISIC2018_Task1-2_Training_Input.zip
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
# !unzip -q ISIC2018_Task1_Training_GroundTruth.zip

# # validation
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip
# !unzip -q ISIC2018_Task1-2_Validation_Input.zip
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip
# !unzip -q ISIC2018_Task1_Validation_GroundTruth.zip

# # test
# !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip
# !unzip -q ISIC2018_Task1-2_Test_Input.zip


# ------ load the config file ------
config = load_config("./configs/default.yaml")
_print("Config:", "info_underline")
print(json.dumps(config, indent=2))
print(20*"~-", "\n")



"""# Dataset"""

import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob



from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


"""# Train

## imports
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import numpy as np
import copy

import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""## make dataset"""

training_dataset = ISIC2018Dataset(**config['dataset']['training']['params'])
validation_dataset = ISIC2018Dataset(**config['dataset']['validation']['params'])

print(f"Length of trainig_dataset:\t{len(training_dataset)}")
print(f"Length of validation_dataset:\t{len(validation_dataset)}")

train_dataloader = DataLoader(training_dataset, **config['data_loader']['train'])
validation_dataloader = DataLoader(validation_dataset, **config['data_loader']['validation'])

"""## model"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Net = globals()[f"{config['model']['name']}"](**config['model']['params'])

criterion = nn.NLLLoss()
optimizer = optim.Adam(Net.parameters(), lr=config['training']['lr'])
# optimizer = optim.RMSprop(Net.parameters(), lr= float(config['lr']), weight_decay=1e-8, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

Net.to(device)

"""## Training"""

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
    for batch in train_dataloader:
        imgs = batch['x']
        msks = batch['y']
        steps += 1
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        preds = Net.forward(imgs)
        loss = criterion(preds, msks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            Net.eval()
            with torch.no_grad():
                for batch in validation_dataloader:
                    imgs = batch['x']
                    labels = batch['y']
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = Net.forward(imgs)
                    batch_loss = criterion(preds, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(preds)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(train_dataloader))
            test_losses.append(test_loss/len(validation_dataloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validation_dataloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validation_dataloader):.3f}")
            running_loss = 0
            Net.train()
torch.save(Net, 'basemodel.pth')





