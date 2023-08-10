import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


class YoloDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform) -> None:
        super(YoloDataset, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104) # RGB

        if isinstance(list_file, list):
            tmp_file = './tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file
        
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()
            