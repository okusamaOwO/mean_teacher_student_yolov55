import torch 
import numpy as np
import random
from utils.general import xyxy2xywhn
import csv
import matplotlib.pyplot as plt
import pandas as pd

def from_targets_to_nms(targets, device, imgsz=(640, 640)):
    """Convert YOLOv5 targets to NMS prediction format for a batch.

    This function is useful for visualizing ground truth labels in the same format
    as model predictions, allowing consistent visualization and debugging.

    Args:
        targets (torch.Tensor): Target tensor of shape (total_detections, 6) where each target
                               is represented as [image_index, class, x_center, y_center, width, height].
                               All coordinates are normalized to [0, 1].
                               Example: tensor([[0.0, 1.0, 0.293, 0.463, 0.034, 0.029],
                                               [0.0, 2.0, 0.521, 0.381, 0.112, 0.055],
                                               [1.0, 0.0, 0.645, 0.712, 0.089, 0.124]])
        device (torch.device): Device to place the output tensors on.
        imgsz (tuple): Image size as (height, width). Default: (640, 640)

    Returns:
        list of torch.Tensor: List of NMS-format predictions for each image in the batch.
                             Each tensor has shape (num_detections, 6) where each detection
                             is represented as [x1, y1, x2, y2, conf, class].
                             Coordinates are in pixel values (0-640 for 640x640 images).
                             Confidence is set to 1.0 for all ground truth boxes.
                             Example: [tensor([[120.5, 200.3, 350.2, 480.7, 1.0, 1.0],
                                              [280.1, 150.2, 390.9, 220.4, 1.0, 2.0]]),
                                      tensor([[350.3, 380.1, 450.6, 480.9, 1.0, 0.0]])]

    Example:
        >>> # Convert ground truth targets to NMS format for visualization
        >>> targets = torch.tensor([[0, 1, 0.5, 0.5, 0.2, 0.3],
        ...                        [1, 0, 0.3, 0.4, 0.1, 0.15]])
        >>> nms_format = from_targets_to_nms(targets, device='cuda', imgsz=(640, 640))
        >>> print(nms_format[0])  # First image detections
        tensor([[256., 192., 384., 288., 1.0, 1.0]])
        >>> print(nms_format[1])  # Second image detections
        tensor([[160., 208., 224., 272., 1.0, 0.0]])
    """
    # Ensure targets are on the correct device
    targets = targets.to(device)
    
    # Find the number of images in the batch
    batch_size = int(targets[:, 0].max().item()) + 1 if len(targets) > 0 else 0
    
    nms_pred_batch = []
    
    for img_id in range(batch_size):
        # Get all targets for this image
        img_targets = targets[targets[:, 0] == img_id]  # Filter by image index
        
        if len(img_targets) == 0:
            # No detections for this image - create empty tensor on correct device
            nms_pred_batch.append(torch.zeros((0, 6), device=device, dtype=targets.dtype))
            continue
        
        # Extract [class, x_center, y_center, width, height] (all normalized)
        cls = img_targets[:, 1].unsqueeze(-1)  # Shape: (N, 1)
        xywhn = img_targets[:, 2:6]  # Shape: (N, 4)
        
        # Convert normalized xywh to pixel xywh (stays on same device)
        xywh_px = xywhn.clone()
        xywh_px[:, 0] = xywhn[:, 0] * imgsz[1]  # x_center to pixels
        xywh_px[:, 1] = xywhn[:, 1] * imgsz[0]  # y_center to pixels
        xywh_px[:, 2] = xywhn[:, 2] * imgsz[1]  # width to pixels
        xywh_px[:, 3] = xywhn[:, 3] * imgsz[0]  # height to pixels
        
        # Convert xywh to xyxy (corner format) - stays on same device
        xyxy = torch.zeros_like(xywh_px)
        xyxy[:, 0] = xywh_px[:, 0] - xywh_px[:, 2] / 2  # x1 = x_center - width/2
        xyxy[:, 1] = xywh_px[:, 1] - xywh_px[:, 3] / 2  # y1 = y_center - height/2
        xyxy[:, 2] = xywh_px[:, 0] + xywh_px[:, 2] / 2  # x2 = x_center + width/2
        xyxy[:, 3] = xywh_px[:, 1] + xywh_px[:, 3] / 2  # y2 = y_center + height/2
        
        # Create confidence column (all 1.0 for ground truth) - explicitly set device
        conf = torch.ones((len(img_targets), 1), device=device, dtype=targets.dtype)
        
        # Concatenate to NMS format: [x1, y1, x2, y2, conf, cls]
        nms_pred = torch.cat([xyxy, conf, cls], dim=1)
        
        nms_pred_batch.append(nms_pred)
    
    return nms_pred_batch

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

def save_losses_to_csv(batch_num, supervised_loss, unsupervised_loss, total_loss, csv_path):
    """Save losses to CSV file for later analysis."""
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Batch', 'Supervised_Loss', 'Unsupervised_Loss', 'Total_Loss'])
        writer.writerow([batch_num, supervised_loss, unsupervised_loss, total_loss])

def plot_losses_from_csv(csv_path, save_path):
    """Plot losses from saved CSV file."""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All losses together
    axes[0, 0].plot(df['Batch'], df['Supervised_Loss'], label='Supervised', color='blue', alpha=0.7)
    axes[0, 0].plot(df['Batch'], df['Unsupervised_Loss'], label='Unsupervised', color='red', alpha=0.7)
    axes[0, 0].plot(df['Batch'], df['Total_Loss'], label='Total', color='green', linewidth=2)
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('All Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Supervised loss only
    axes[0, 1].plot(df['Batch'], df['Supervised_Loss'], color='blue')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Supervised Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Unsupervised loss only
    axes[1, 0].plot(df['Batch'], df['Unsupervised_Loss'], color='red')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Unsupervised Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Moving average (smoothed)
    window = 50
    df['Supervised_MA'] = df['Supervised_Loss'].rolling(window=window).mean()
    df['Unsupervised_MA'] = df['Unsupervised_Loss'].rolling(window=window).mean()
    df['Total_MA'] = df['Total_Loss'].rolling(window=window).mean()
    
    axes[1, 1].plot(df['Batch'], df['Supervised_MA'], label='Supervised (MA)', color='blue')
    axes[1, 1].plot(df['Batch'], df['Unsupervised_MA'], label='Unsupervised (MA)', color='red')
    axes[1, 1].plot(df['Batch'], df['Total_MA'], label='Total (MA)', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Loss (Moving Average)')
    axes[1, 1].set_title(f'Smoothed Losses (window={window})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")