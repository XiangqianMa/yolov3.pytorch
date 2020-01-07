import torch
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt


class COCODataset(Dataset):
    def __init__(self, data_root):
