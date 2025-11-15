import torch 
import numpy as np

def from_nms_to_targets(nms_pred, device, imgsz = (640,640)):
    """Convert NMS predictions to target format.

    Args:
        nms_pred (list of torch.Tensor): List of NMS predictions for each image in the batch.
                                         Each tensor has shape (num_detections, 6) where each detection
                                         is represented as (x1, y1, x2, y2, conf, class). 
                                         [322.39029, 343.93793, 393.15659, 379.43707, 0.94106, 1.00000]

    Returns:
        torch.Tensor: Converted targets with shape (total_detections, 6) where each target is
                      represented as (image_index, class, x_center, y_center, width, height).
                              [1.50000e+01, 1.00000e+00, 2.92725e-01, 4.63379e-01, 3.36914e-02, 2.92969e-02]]) (normalized)

    """
    print("imgsz in from_nms_to_targets:", imgsz)
    fog_targets = []
    for img_idx, nms_det_each_img in enumerate(nms_pred): 
        # nms_det cho một ảnh, trong đó có nhiều labels khác nhau 
        for det_each_box in nms_det_each_img:
            x1, y1, x2, y2, conf, cls = det_each_box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            x_center, y_center, width, height = x_center / imgsz[1], y_center / imgsz[0], width / imgsz[1], height / imgsz[0]
            tensor = torch.tensor([img_idx, cls, x_center, y_center, width, height])
            fog_targets.append(tensor)
    if fog_targets:
        return torch.stack(fog_targets).to(device).type(nms_pred[0].dtype)
    else:
        return torch.zeros((0, 6), dtype=nms_pred[0].dtype).to(device)

