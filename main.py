import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ------------------- For the training set -------------------
