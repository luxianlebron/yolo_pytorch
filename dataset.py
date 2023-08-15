import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom
import random

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
            if not os.path.exists(xml_file):
                continue
            labels, boxes = self._get_objects_info(xml_file)

            self.labels.append(torch.Tensor(labels))
            self.boxes.append(torch.Tensor(boxes))
        self.num_samples = len(self.labels)

    def __getitem__(self, idx):
        img_file = os.path.join(self.jpg_imgs, self.fnames[idx]+'.jpg').replace('\\', '/')
        img = cv2.imread(img_file)
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        if self.train_val_test == 'train':
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)
            img = self.random_blur(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

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
            boxes = torch.Tensor(resized_boxes)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ''' debug '''
        # fig = plt.figure(figsize=(20, 10))
        # box_show = boxes[0]
        # pt1 = (int(box_show[0]), int(box_show[1]))
        # pt2=(int(box_show[2]), int(box_show[3]))
        # cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        # plt.imshow(img)
        # plt.show()
        ''' debug '''
        img = self.sub_mean(img, self.mean)
        target = self.encoder(boxes, labels)

        return torch.Tensor(img), target

    def __len__(self):
        return self.num_samples

    def sub_mean(self, rgb, mean):
        mean = np.array(mean)
        rgb = rgb - mean
        return rgb

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

    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            img_lr = np.fliplr(img).copy()
            h,w,_ = img_lr.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return img_lr, boxes
        return img, boxes

    def random_crop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height, height)
            w = random.uniform(0.6*width, width)
            x = random.uniform(0, width-w)
            y = random.uniform(0, height-h)
            x, y, h, w = int(x), int(y), int(h), int(w)
            
            center -= torch.Tensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0]>0) & (center[:, 0]<w)
            mask2 = (center[:, 1]>0) & (center[:, 1]<h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.Tensor([[x, y, x, y]]).expand_as(boxes_in)
    
            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h, x:x+w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def random_scale(self, img, boxes):
        # 固定高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height,width,c = img.shape
            img = cv2.resize(img, (int(width*scale), height))
            scale_tensor = torch.Tensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
        return img, boxes
    
    def random_bright(self, img, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            img = img * alpha + random.randrange(-delta, delta)
            img = img.clip(0, 255).astype(np.uint8)
        return img

    def random_brightness(self, bgr):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v*adjust
            v = v.clip(0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def random_blur(self, img):
        if random.random() < 0.5:
            img = cv2.blur(img, (5,5))
        return img
    
    def random_hue(self, bgr):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h*adjust
            h = h.clip(0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s*adjust
            s = s.clip(0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def random_shift(self, bgr, boxes, labels):
        # 平移变换
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height, width, c = bgr.shape
            after_shift_image = np.zeros_like(bgr).astype(bgr.dtype)
            after_shift_image[:, :, :] = (104, 117, 123)
            shift_x = random.uniform(-0.2*width, 0.2*width)
            shift_y = random.uniform(-0.2*height, 0.2*height)
            #原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shift_image[int(shift_y):, int(shift_x):, :] = bgr[:height-int(shift_y), :width-int(shift_x), :]
            elif shift_x>=0 and shift_y<0:
                after_shift_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shift_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shift_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]
            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shift_image, boxes_in, labels_in
        return bgr, boxes, labels

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

    val_dataset = YoloDataset(root_dir, 'train', None)
    print(val_dataset[0])

    print('Done!')

