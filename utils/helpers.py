import torch 
import numpy as np
import random
from utils.general import xyxy2xywhn

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
    # print("imgsz in from_nms_to_targets:", imgsz)
    # fog_targets = []
    # for img_idx, nms_det_each_img in enumerate(nms_pred): 
    #     # nms_det cho một ảnh, trong đó có nhiều labels khác nhau 
    #     for det_each_box in nms_det_each_img:
    #         x1, y1, x2, y2, conf, cls = det_each_box
    #         x_center = (x1 + x2) / 2
    #         y_center = (y1 + y2) / 2
    #         width = x2 - x1
    #         height = y2 - y1
    #         x_center, y_center, width, height = x_center / imgsz[1], y_center / imgsz[0], width / imgsz[1], height / imgsz[0]
    #         tensor = torch.tensor([img_idx, cls, x_center, y_center, width, height])
    #         fog_targets.append(tensor)
    # if fog_targets:
    #     return torch.stack(fog_targets).to(device).type(nms_pred[0].dtype)
    # else:
    #     return torch.zeros((0, 6), dtype=nms_pred[0].dtype).to(device)
    
    pred_labels_out_batch = []
    for img_id in range(len(nms_pred)):
        labels_num = nms_pred[img_id].shape[0]  # pred_tf_nms prediction shape is (bs,n,6), per image [xyxy, conf, cls]
        if labels_num:
            labels_list = torch.cat((nms_pred[img_id][:, 5].unsqueeze(-1),
                nms_pred[img_id][:, 0:4]), dim=1)  # remove predicted conf, new format [cls x y x y]
            labels_list[:, 1:5] = xyxy2xywhn(labels_list[:, 1:5], w=imgsz[1], h=imgsz[0])  # xyxy to xywh normalized
            pred_labels_out = torch.cat(((torch.ones(labels_num)*img_id).unsqueeze(-1).to(device),
                labels_list), dim=1)  # pred_labels_out shape is (labels_num, 6), per label format [img_id cls x y x y]
        # else:
            # pred_labels_out = pred_tf_nms[img_id]  # in this condition, pred_tf_nms[img_id] tensor size is [0,6]
            '''[BUG] When training, nan can appear in batchnorm when all the values are the same, and thus std = 0'''
            # pred_labels_out = torch.from_numpy(np.array([[img_id,0,0,0,0,0]])).to(device)
            '''If no bboxes have been detected, we set a [0,0,w,h](xyxy) or [0.5,0.5,1,1](xywh) bounding-box for the image'''
            # pred_labels_out = torch.from_numpy(np.array([[img_id,0,0.5,0.5,1,1]])).to(device)
            pred_labels_out_batch.append(pred_labels_out)
    if len(pred_labels_out_batch) != 0:
        pred_labels = torch.cat(pred_labels_out_batch, dim=0)
    else:
        # pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, 1, 1]])).to(device)
        pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, random.uniform(0.2,0.8), random.uniform(0.2,0.8)]])).to(device)
    return pred_labels 

def update_teacher(student_model, teacher_model, alpha):
    """Update teacher model parameters using exponential moving average of student model parameters.

    Args:
        student_model (torch.nn.Module): The student model.
        teacher_model (torch.nn.Module): The teacher model to be updated.
        alpha (float): The EMA decay factor.

    Returns:
        None
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


def visualize_nms_for_an_img(det, img, save_dir="result.jpg"):
    """Visualize NMS detections on a single image and save it as 'result.jpg'.
    
    This function takes detection results from non-maximum suppression (NMS) and draws
    bounding boxes with confidence scores on the input image. The visualization is saved
    to disk as 'result.jpg'.
    
    Args:
        det (torch.Tensor): Detection tensor of shape (N, 6) where N is the number of detections.
                           Each detection contains [x1, y1, x2, y2, confidence, class_id].
                           Coordinates should be in pixel values (0-640 for 640x640 images).
                           Example: tensor([[120.5, 200.3, 350.2, 480.7, 0.89, 0.0],
                                           [50.1, 100.2, 180.9, 220.4, 0.75, 2.0]])
        img (torch.Tensor): Image tensor of shape (C, H, W) with values in range [0, 1].
                           Typically (3, 640, 640) for RGB images.
                           Will be converted to uint8 and transposed to (H, W, C) for visualization.
    
    Returns:
        None: Saves the visualized image to 'result.jpg' in the current directory.
    
    Note:
        - Bounding boxes are drawn in blue (BGR: 255, 0, 0)
        - Confidence scores are displayed in green text above each box
        - The function overwrites 'result.jpg' if it already exists
        - Input image is normalized [0, 1] and will be scaled to [0, 255]
    """
    img = img.cpu().numpy() * 255
    img = img.astype(np.uint8).transpose(1, 2, 0)  # H, W, C
    img = np.ascontiguousarray(img)

    import cv2
    for *xyxy, conf, cls in det:
        label = f'{conf:.2f}'
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                        (255, 0, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)

    cv2.imwrite(save_dir, img)
    print(f"Saved {save_dir}")