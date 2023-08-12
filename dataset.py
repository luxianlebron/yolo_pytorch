import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import pandas as pd
import matplotlib.pyplot as plt


VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')



class YoloDataset(data.Dataset):
    def __init__(self, file_path, transform) -> None:
        super(YoloDataset, self).__init__()
        self.file_path = file_path
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104) # RGB

        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append()
            

