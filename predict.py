import cv2
import torch
import matplotlib.pyplot as plt

from nms import NMS
from model import YoloModel
from dataset import VOC_CLASSES


model_path = r''
image_path = r'./horse.jpg'
confidence_threshold = 0.5
iou_threshold = 0.5

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (448, 448))
model_in = torch.Tensor(img).type(torch.float32).unsqueeze(0).permute(0, 3, 1, 2) / 255.

model = YoloModel(7, 2, 20)
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)
model.eval()
with torch.no_grad():
    pred = model(model_in)

pred_boxes = pred.squeeze(0)
# translate (cx, cy, dx, dy) to (xmin, ymin, xmax, ymax)
w1h1 = pred_boxes[:, :, 2:4] * 448.
for grid_i in range(7):
    for grid_j in range(7):
        grid_xy = torch.Tensor([grid_j*64, grid_i*64])
        # first bounding box
        cxcy = grid_xy + pred_boxes[grid_j, grid_i, :2]
        xminymin = cxcy - grid_xy
        xmaxymax = cxcy + grid_xy
        pred_boxes[grid_j, grid_i, :2] = xminymin
        pred_boxes[grid_j, grid_i, 2:4] = xmaxymax
        # second bounding box
        cxcy = grid_xy + pred_boxes[grid_j, grid_i, 5:7]
        xminymin = cxcy - grid_xy
        xmaxymax = cxcy + grid_xy
        pred_boxes[grid_j, grid_i, 5:7] = xminymin
        pred_boxes[grid_j, grid_i, 7:9] = xmaxymax

pred_boxes = pred_boxes.contiguous().view(49, 30)
predicted_boxes = NMS(pred_boxes, confidence_threshold, iou_threshold)

for box in predicted_boxes:
    pt1 = [int(box[0].item()), int(box[1].item())]
    pt2 = [int(box[2].item()), int(box[3].item())]
    cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=1)
    cv2.putText(img, VOC_CLASSES[int(box[6].item())], [int(box[0].item()), int(box[1].item())-5], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

figure = plt.figure(figsize=(10, 5))
plt.imshow(img)
plt.show()

