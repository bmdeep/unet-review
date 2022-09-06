from extra.utils import (
  load_config,
  _print,
)
import json
import importlib
from datasets import (
  ISIC2018Dataset
)

from models.unet import BasicUnet


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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


# ------ load the config file ------
config = load_config("./configs/default.yaml")
_print("Config:", "info_underline")
print(json.dumps(config, indent=2))
print(20*"~-", "\n")

train_batch_size = config['training']['batch_size']
test_batch_size = config['testing']['batch_size']

# ------ load a config file ------
Dataset = globals()[config['dataset']['class_name']]
dataset = Dataset(**config['dataset']['params'])


# total images in set
print(dataset.len)

train_len = int(0.6 * dataset.len)
val_len   = dataset.len - train_len
train_set, val_set = Dataset.random_split(
                      dataset, 
                      lengths=[train_len, val_len]
                    )

# check lens of subset
len(train_set), len(val_set)

train_set    = CustomDataset("")
train_set    = torch.utils.data.TensorDataset(
                train_set, 
                train=True, 
                batch_size=train_batch_size
              )

train_loader = torch.utils.data.DataLoader(
                train_set, 
                batch_size=train_batch_size, 
                shuffle=True, 
                num_workers=1
              )

val_set      = torch.utils.data.DataLoader(Dataset, batch_size=4, sampler=val_set)
val_loader   = torch.utils.data.DataLoader(Dataset, batch_size=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.resnet50(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

model.fc = nn.Sequential(
                          nn.Linear(2048, 512),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(512, 10),
                          nn.LogSoftmax(dim=1)
                        )
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
# optimizer = optim.RMSprop(Net.parameters(), lr= float(config['lr']), weight_decay=1e-8, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
model.to(device)
if int(pretrained=True):
  model.load_state_dict(
    torch.load(
      config['Model']['SavePath'], 
      map_location='cpu'
    )['model_weights']
  )
