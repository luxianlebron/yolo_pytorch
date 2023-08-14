import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES_DICT = {c:idx for idx, c in enumerate(VOC_CLASSES)}

class YoloDataset(data.Dataset):
    img_size = 448
    def __init__(self, root_dir, train_val_test, transform) -> None:
        '''
            args:
                root_dir: dataset root dir
                train_val_test: in ['train', 'val', 'test']
                transform: torchvision
        '''
        super(YoloDataset, self).__init__()
        self.root_dir = root_dir
        self.train_val_test = train_val_test
        self.im_sets = os.path.join(root_dir, 'ImageSets', 'Main', train_val_test+'.txt')
        self.annotations = os.path.join(root_dir, 'Annotations')
        self.jpg_imgs = os.path.join(root_dir, 'JPEGImages')
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104) # RGB

        with open(self.im_sets) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            xml_file = os.path.join(self.annotations, splited[0]+'.xml').replace('\\', '/')
            labels, boxes = self._get_objects_info(xml_file)

            self.labels.append(labels)
            self.boxes.append(boxes)
        self.num_samples = len(self.labels)

    def __getitem__(self, idx):
        img_file = os.path.join(self.jpg_imgs, self.fnames[idx]+'.jpg').replace('\\', '/')
        img = cv2.imread(img_file)
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        h, w, _ = img.shape
        if h != self.img_size or w != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        # resize boxes of object after resizing image
        resized_boxes = []
        for box in boxes: # box: [xmin, ymin, xmax, ymax]
            xmin = float(box[0] * (self.img_size / w))
            ymin = float(box[1] * (self.img_size / h))
            xmax = float(box[2] * (self.img_size / w))
            ymax = float(box[3] * (self.img_size / h))
            resized_boxes.append([xmin, ymin, xmax, ymax])
        boxes = resized_boxes

        if self.train_val_test == 'train':
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        img = self.BGR2RGB(img)

        ''' debug '''
        # fig = plt.figure(figsize=(20, 10))
        # box_show = boxes[0]
        # pt1 = (int(box_show[0]), int(box_show[1])); pt2=(int(box_show[2]), int(box_show[3]))
        # cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        # plt.imshow(img)
        # plt.show()
        ''' debug '''

        # img = self.normalization(img)
        target = self.encoder(torch.Tensor(boxes), torch.Tensor(labels))

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[xmin, ymin, xmax, ymax], []]
        labels (tensor)  [...]
        return (tensor) 7x7x30 (7 x 7 x (5*2 + 20))
        '''
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, 30))
        grid_size = self.img_size/grid_num
        wh = (boxes[:, 2:] - boxes[:, :2]) / self.img_size # nomalization to the whole image
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2. # x y 
        for i in range(cxcy.size(0)):
            grid_ij = ((cxcy[i, :] / grid_size).ceil() - 1).type(torch.int32)
            grid_cxcy = grid_ij * (self.img_size / grid_num)
            delta_cxcy = (cxcy[i, :] - grid_cxcy) / grid_size
            target[grid_ij[1], grid_ij[0], 4] = 1 # c
            target[grid_ij[1], grid_ij[0], 9] = 1 # c
            target[grid_ij[1], grid_ij[0], int(labels[i])+9+1] = 1 # class label
            target[grid_ij[1], grid_ij[0], :2] = delta_cxcy
            target[grid_ij[1], grid_ij[0], 2:4] = wh[i]
            target[grid_ij[1], grid_ij[0], 5:7] = delta_cxcy
            target[grid_ij[1], grid_ij[0], 7:9] = wh[i]
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def 



    @staticmethod
    def _get_objects_info(xml_file):
        dom = minidom.parse(xml_file)
        anno = dom.documentElement
        objects = anno.getElementsByTagName('object')
        labels = []
        bndboxes = []
        for obj in objects:
            obj_name = obj.getElementsByTagName('name')[0].childNodes[0].data
            bndbox_dom = obj.getElementsByTagName('bndbox')[0]
            xmin = int(bndbox_dom.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = int(bndbox_dom.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = int(bndbox_dom.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = int(bndbox_dom.getElementsByTagName('ymax')[0].childNodes[0].data)
            bndbox = [xmin, ymin, xmax, ymax]

            labels.append(VOC_CLASSES_DICT[obj_name])
            bndboxes.append(bndbox)

        return labels, bndboxes



if __name__ == '__main__':
    root_dir = 'C:/Users/Administrator/Desktop/VOCdevkit/VOC2007'

    val_dataset = YoloDataset(root_dir, 'val', None)
    print(val_dataset[0])

    print('Done!')

