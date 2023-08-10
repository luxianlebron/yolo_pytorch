import torch
nn = torch.nn
F = torch.nn.functional

class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord=5, l_noobj=0.5) -> None:
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    @staticmethod
    def compute_iou(box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1, y1, x2, y2].
                                                                                [x1, y1] left top point
                                                                                [x2, y2] right bottom point
        Args:
            box1: (tensor) bounding boxes, shape [N, 4].
            box2: (tensor) bounding boxes, shape [M, 4].
        Return:
            (tensor), iou, shape [N, M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            box2[:, :2].unsqueeze(0).expand(N, M, 2))
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:].unsqueeze(0).expand(N, M, 2))

        wh = rb - lt
        wh[wh<0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
    
        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred, target, weights=None):
        '''
        pred: (N, S, S, B*5+20=30) [x, y, w, h, c]
        target: (N, S, S, 30)
        '''
        N = pred.size(0)
        device = target.device
        coo_mask = target[..., 4] > 0
        noo_mask = target[..., 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target)

        coo_pred = pred[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5) #[x1, y1, w1, h1, c1]
        class_pred = coo_pred[:, 10:]

        coo_target = target[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 1. compute not contain obj loss
        noo_pred = pred[noo_mask].contiguous().view(-1, 30)
        noo_target = target[noo_mask].contiguous().view(-1, 30)
        noo_pred_mask = torch.zeros(noo_pred.size()).type(torch.bool).to(device)
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        noobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum') # 只计算 c 的损失

        # 2. compute contain obj c loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        # 3.compute contain obj loss
        coo_response_mask = torch.zeros(box_target.size()).type(torch.bool).to(device)
        coo_not_response_mask = torch.zeros(box_target.size()).type(torch.bool).to(device)
        box_target_iou = torch.zeros(box_target.size()).to(device)
        for i in range(0, box_target.size(0), 2):
            box1 = box_pred[i:i+2]
            box1_xyxy = torch.FloatTensor(box1.size())
            box1_xyxy[:, :2] = box1[:, :2]/14. - 0.5*box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2]/14. + 0.5*box1[:, 2:4]

            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2]/14. - 0.5*box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2]/14. + 0.5*box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou, max_index = iou.max(0)

            coo_response_mask[i+max_index] = 1
            coo_not_response_mask[i+1-max_index] = 1

            box_target_iou[i+max_index, :4] = max_iou

        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2], reduction='sum') + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]), reduction='sum')

        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4], reduction='sum')

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*noobj_loss + class_loss)/N

