import torch
from loss import YoloLoss

def NMS(bounding_boxes:torch.Tensor, confidence_threshold, iou_threshold):
    """
    bounding_boxes: (tensor) [[x1, y1, x2, y2, confidence1,
                               x1, y1, x2, y2, confidence2, classes_prob], ]
    """
    # 1. 初步筛选，选取 grid cell 预测出的两个框中 confidence 最大的那个
    boxes = []
    for box_ in bounding_boxes:
        if box_[4] < confidence_threshold or box_[9] < confidence_threshold:
            continue
        classes = box_[10:]
        class_prob_idx = torch.argmax(classes)
        class_prob = classes[class_prob_idx]
        box = torch.zeros(7)
        if box_[4] > box_[9]:
            box[:5] = box_[:5]
        else:
            box[:5] = box_[5:10]

        box[5:] = torch.Tensor([class_prob_idx, class_prob])
        boxes.append(box)

    # 2. NMS. 以confidence为标准对bounding boxes进行排序，取出confidence最大的box
    predicted_boxes = []
    while len(boxes) != 0:
        boxes = sorted(boxes, key=(lambda x : x[4]), reverse=True)
        choiced_box = boxes.pop(0)
        predicted_boxes.append(choiced_box)
        for i in range(len(boxes)):
            if YoloLoss.compute_iou(choiced_box[:4].unsqueeze(0), boxes[i][:4].unsqueeze(0)) > iou_threshold:
                boxes.pop(i)

    return predicted_boxes


if __name__ == '__main__':
    bounding_boxes = torch.randn((49, 30))
    predicted_boxes = NMS(bounding_boxes, 0.5, 0.5)
    print(predicted_boxes)
