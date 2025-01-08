import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Load your trained YOLOv11 model
model_path="C:\\Users\\Ahmet\\Desktop\\Python\\cv project YOLOV11\\runs\\detect\\YOLO v11 trhird trial 200 epochs\\weights\\best.pt"
model = YOLO(model_path)

# Evaluate the model on your validation dataset
results = model.val(data="dataset_val.yaml",workers=0)
print(results)

# Extract precision, recall, and mAP from the results
precision = results.results_dict['metrics/precision(B)']
recall = results.results_dict['metrics/recall(B)']
mAP50 = results.results_dict['metrics/mAP50(B)']
mAP50_95 = results.results_dict['metrics/mAP50-95(B)']

iou_threshold = 0.52

def ret_sum_center_point_cords(pnt):
    x_middle = (pnt[0]+pnt[2]) / 2
    y_middle = (pnt[1]+pnt[3]) / 2
    return x_middle+y_middle

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    #print(f"Ground Truth : {box1}")
    #print(f"Predicted : {box2}")
    #print()
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / float(union_area)

    return iou


# Function to read the ground truth boxes from the label files and scale them
def read_ground_truth_boxes(label_file, img_width, img_height):
    boxes = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
    return boxes

# Directory paths
val_images_dir = Path('C:/Users/Ahmet/Desktop/Python/cv project YOLOV11/test/images')
val_labels_dir = Path('C:/Users/Ahmet/Desktop/Python/cv project YOLOV11/test/labels')

iou_scores = []

# Iterate over all images and labels in the validation set
for img_file in val_images_dir.glob('*.png'):
    label_file = val_labels_dir / f"{img_file.stem}.txt"


    # Get predictions from the model
    #print(img_file)
    img = Image.open(img_file)
    img_width, img_height = img.size
    results = model(img)[0]
    pred_boxes = results.boxes.xyxy.cpu().numpy() # Extract predicted bounding boxes (x1, y1, x2, y2)
    pred_boxes = sorted(pred_boxes,key= lambda x: ret_sum_center_point_cords(x))

    # Get ground truth boxes
    #print(label_file)
    ground_truth_boxes = read_ground_truth_boxes(label_file, img_width, img_height)
    ground_truth_boxes = sorted(ground_truth_boxes,key= lambda x: ret_sum_center_point_cords(x))

    #print(f"pred_boxes : {pred_boxes}\n")
    #print(f"ground truth boxes : {ground_truth_boxes}\n\n")

    # Calculate IoU for each ground truth and predicted box pair
    for gt_box in ground_truth_boxes:
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box  , pred_box)
            iou_scores.append(iou)


#filter iou
iou_scores = [i for i in iou_scores if i>=iou_threshold]
# Calculate the mean IoU
mean_iou = np.mean(iou_scores)

# Print out the evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"mAP@0.5: {mAP50}")
print(f"mAP@0.5:0.95: {mAP50_95}")
print(f"Mean IoU: {mean_iou}")
print(f"Filtered IoU list length {len(iou_scores)}")
