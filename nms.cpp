// NMS algorithm implemented in C

#include <math.h>

typedef struct Bbox
{
    int x1;
    int y1;
    int x2;
    int y2;
    int class;
    float score;
}Bbox;


/*
bounding_box: xmin, ymin, xmax, ymax, confidence, classes
*/
std::vector<Bbox> NMS(std::vector<Bbox>& bboxes, int iou_th)
{
    std::vector<Bbox> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int select_idx = 0;
    int num_bbox = static_cast<int>(bboxes.size());
    std::vector(int) mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        Bbox select_bbox = bboxes[select_idx];

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            float iou = compute_iou(select_bbox, bboxes[i]);

            if (iou > iou_th) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}


float compute_iou(Bbox box1, Bbox box2)
{
    int xmax = std::max<float>(box1.x1, box2.x1);
    int xmin = std::min<float>(box1.x2, box2.x2);
    int ymax = std::max<float>(box1.y1, box2.y1);
    int ymin = std::min<float>(box1.y2, box2.y2);

    if (xmin <= xmax || ymin <= ymax) // 如果没有重叠
        return 0.;

    float inter_area = (xmin - xmax) * (ymin - ymax); // intersection area
    float area1 = static_cast<float>(box1.x2 - box1.x1) * (box1.y2 - box.y1);
    float area2 = static_cast<float>(box1.x2 - box1.x1) * (box1.y2 - box.y1);
    float iou = (inter_area / (area1 + area2 - inter_area));
    return iou
}

