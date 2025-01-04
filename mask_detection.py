import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from PIL import Image
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
import itertools
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

PATH = './mask_trained.pth'
TEST_PATH = './test_images/test0.jpeg'
METRICS = False
SAVING = True
ONLY_TESTING = True
VIDEO = False
batch_size = 2
epochs = 200
num_classes = 3
threshold = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['without_mask', 'with_mask', 'mask_weared_incorrect']

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def generate_default_boxes(device):
    feature_map_sizes = [38, 19, 10, 5, 3, 1]
    aspect_ratios = [1.0, 2.0, 0.5]
    scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]

    default_boxes = []
    for k, f in enumerate(feature_map_sizes):
        for i, j in itertools.product(range(f), repeat=2):
            for ar in aspect_ratios:
                cx = (i + 0.5) / f
                cy = (j + 0.5) / f
                s = scales[k]
                default_boxes.append([cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)])
                if ar == 1.0:
                    try:
                        default_boxes.append([cx, cy, np.sqrt(s * scales[k+1]), np.sqrt(s * scales[k+1])])
                    except IndexError:
                        pass

    default_boxes = torch.tensor(default_boxes, dtype=torch.float32).to(device)
    return default_boxes



def iou(box_a, box_b):
    inter_x1 = torch.max(box_a[..., 0], box_b[..., 0])
    inter_y1 = torch.max(box_a[..., 1], box_b[..., 1])
    inter_x2 = torch.min(box_a[..., 2], box_b[..., 2])
    inter_y2 = torch.min(box_a[..., 3], box_b[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box_a_area = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    box_b_area = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    union_area = box_a_area + box_b_area - inter_area
    return inter_area / union_area

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
class SSDHead(nn.Module):
    def __init__(self, num_classes):
        super(SSDHead, self).__init__()
        self.num_anchors = 6
        self.num_classes = num_classes

        self.cls_conv = nn.Conv2d(128, self.num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.reg_conv = nn.Conv2d(128, self.num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def set_default_boxes(self, default_boxes):
        self.default_boxes = default_boxes.to(next(self.parameters()).device)

    def forward(self, x):
        class_preds = self.cls_conv(x)  # Should output 18 (6 anchors * 3 classes)
        bbox_preds = self.reg_conv(x)   # Should output 24 (6 anchors * 4 coordinates)

        batch_size, _, height, width = class_preds.shape
        class_preds = class_preds.view(batch_size, self.num_anchors, self.num_classes, height, width)
        bbox_preds = bbox_preds.view(batch_size, self.num_anchors, 4, height, width)

        # Debug: Print the shape and sample values of class predictions
        #print(f"Class Predictions Shape: {class_preds.shape}")
        #print(f"Sample Class Predictions: {class_preds[0, 0, :, 0, 0].detach().cpu().numpy()}")

        return class_preds, bbox_preds



class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.backbone = CNNBackbone()
        self.head = SSDHead(num_classes)
        self.apply(initialize_weights)  

    def forward(self, x):
        features = self.backbone(x)
        class_pred, bbox_pred = self.head(features)
        return class_pred, bbox_pred


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir("./Images/")))  
        self.labels = list(sorted(os.listdir("./Annotations/")))  

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]

        img_path = os.path.join("./Images/", img_name)
        label_path = os.path.join("./Annotations/", label_name)
        
        img = Image.open(img_path).convert("RGB")

        target = self.generate_target(label_path, img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def generate_target(self, label_path, img):
        label_map = {"without_mask": 0, "with_mask": 1, "mask_weared_incorrect": 2}
        tree = ET.parse(label_path)
        root = tree.getroot()

        unique_labels = set()
        img_width, img_height = img.size

        for obj in root.iter('object'):
            label = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text) / img_width
            ymin = int(obj.find('bndbox/ymin').text) / img_height
            xmax = int(obj.find('bndbox/xmax').text) / img_width
            ymax = int(obj.find('bndbox/ymax').text) / img_height

            if label in label_map:
                unique_labels.add(label_map[label])
            else:
                print(f"Warning: Label '{label}' not in label_map, skipping.")

        # Print unique labels
        #print(f"Unique Labels: {unique_labels}")

        return {
            'boxes': torch.tensor([[xmin, ymin, xmax, ymax] for _ in unique_labels], dtype=torch.float32),
            'labels': torch.tensor(list(unique_labels), dtype=torch.int64)
        }



def collate_fn(batch):
    max_num_boxes = max(len(target['boxes']) for _, target in batch)
    padded_images, padded_targets = [], []

    for img, target in batch:
        num_boxes = len(target['boxes'])
        pad_boxes = torch.cat([target['boxes'], torch.zeros(max_num_boxes - num_boxes, 4)])
        pad_labels = torch.cat([target['labels'], torch.full((max_num_boxes - num_boxes,), -1, dtype=torch.int64)])
        padded_targets.append({'boxes': pad_boxes, 'labels': pad_labels})
        padded_images.append(img)

    return torch.stack(padded_images, dim=0), padded_targets


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])



def focal_loss(class_preds, class_targets, alpha=0.25, gamma=2.0):
    cross_entropy = F.cross_entropy(class_preds, class_targets, reduction='none')
    pt = torch.exp(-cross_entropy)
    focal_loss = alpha * (1 - pt) ** gamma * cross_entropy
    return focal_loss.mean()

def ssd_loss(class_preds, bbox_preds, targets, default_boxes, num_classes=3, negative_ratio=3):
    batch_size = class_preds.size(0)
    num_anchors = class_preds.size(1)
    height, width = class_preds.size(3), class_preds.size(4)

    class_targets = torch.zeros(batch_size, num_anchors, height, width, dtype=torch.long, device=class_preds.device)
    bbox_targets = torch.zeros(batch_size, num_anchors, 4, height, width, device=bbox_preds.device)

    for i in range(batch_size):
        boxes = targets[i]['boxes']
        labels = targets[i]['labels']

        if len(boxes) == 0:
            continue

        iou_scores = iou(boxes.unsqueeze(1).to(class_preds.device), default_boxes.unsqueeze(0).to(class_preds.device))
        best_box_idx = iou_scores.argmax(dim=0)

        for j, box in enumerate(boxes):
            default_box_idx = best_box_idx[j].item()
            dbox_y, dbox_x = divmod(default_box_idx, width)
            if labels[j] >= 0:  # Ensure valid labels
                class_targets[i, :, dbox_y, dbox_x] = labels[j]
                bbox_targets[i, :, :, dbox_y, dbox_x] = box.unsqueeze(0).expand(num_anchors, 4)

    class_preds_flat = class_preds.permute(0, 3, 4, 1, 2).reshape(-1, num_classes)
    class_targets_flat = class_targets.permute(0, 2, 3, 1).reshape(-1)
    bbox_preds_flat = bbox_preds.permute(0, 3, 4, 1, 2).reshape(-1, 4)
    bbox_targets_flat = bbox_targets.permute(0, 2, 3, 1, 4).reshape(-1, 4)

    valid_mask = (class_targets_flat >= 0) & (class_targets_flat < num_classes)

    # Debug: Print raw class predictions and targets
    #print(f"Raw Class Predictions: {class_preds_flat[valid_mask].detach().cpu().numpy()}")
    #print(f"Raw Class Targets: {class_targets_flat[valid_mask].detach().cpu().numpy()}")

    positive_mask = class_targets_flat > 0
    negative_mask = class_targets_flat == 0

    # Calculate classification loss using focal loss
    class_loss = focal_loss(class_preds_flat[valid_mask], class_targets_flat[valid_mask])
    bbox_loss = F.smooth_l1_loss(bbox_preds_flat[positive_mask], bbox_targets_flat[positive_mask])

    return class_loss + bbox_loss




def inspect_predictions(model, device, dataset, threshold=0.5):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_preds = []
    all_labels = []

    for img, target in dataset:
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            class_preds, bbox_preds = model(img_tensor)

        class_preds = class_preds.permute(0, 2, 3, 4, 1).reshape(-1, class_preds.shape[1])
        class_probs, class_labels_indices = class_preds.max(dim=-1)
        mask = class_probs > threshold

        filtered_class_labels_indices = class_labels_indices[mask]
        ground_truth_labels = target['labels']

        preds_len = len(filtered_class_labels_indices)
        labels_len = len(ground_truth_labels)

        if preds_len > labels_len:
            filtered_class_labels_indices = filtered_class_labels_indices[:labels_len]
        elif preds_len < labels_len:
            ground_truth_labels = ground_truth_labels[:preds_len]

        all_preds.extend(filtered_class_labels_indices.cpu().numpy().tolist())
        all_labels.extend(ground_truth_labels.cpu().numpy().tolist())

        # Print raw outputs for debugging
        print(f"Raw Class Predictions: {class_preds}")
        print(f"Filtered Class Labels Indices: {filtered_class_labels_indices}")
        print(f"Ground Truth Labels: {ground_truth_labels}")

    unique_preds = np.unique(all_preds)
    unique_labels = np.unique(all_labels)
    print(f"Unique predicted labels: {unique_preds}")
    print(f"Unique ground truth labels: {unique_labels}")

    return all_preds, all_labels



def get_predictions_and_ground_truths(model, device, dataset, threshold=0.5):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_preds = []
    all_labels = []

    for img, target in dataset:
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            class_preds, bbox_preds = model(img_tensor)

        class_preds = class_preds.permute(0, 2, 3, 4, 1).reshape(-1, class_preds.shape[1])
        class_probs, class_labels_indices = class_preds.max(dim=-1)
        mask = class_probs > threshold

        filtered_class_labels_indices = class_labels_indices[mask]
        ground_truth_labels = target['labels']

        preds_len = len(filtered_class_labels_indices)
        labels_len = len(ground_truth_labels)

        if preds_len > labels_len:
            filtered_class_labels_indices = filtered_class_labels_indices[:labels_len]
        elif preds_len < labels_len:
            ground_truth_labels = ground_truth_labels[:preds_len]

        all_preds.extend(filtered_class_labels_indices.cpu().numpy().tolist())
        all_labels.extend(ground_truth_labels.cpu().numpy().tolist())

    # Print unique labels to debug
    unique_preds = np.unique(all_preds)
    unique_labels = np.unique(all_labels)
    print(f"Unique predicted labels: {unique_preds}")
    print(f"Unique ground truth labels: {unique_labels}")

    return all_preds, all_labels



def calculate_metrics(preds, labels):
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=class_labels, zero_division=0))
    
    cm = confusion_matrix(labels, preds, labels=range(len(class_labels)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate(model, data_loader, default_boxes, num_classes=3, threshold=0.5):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            if not check_data_consistency(images, targets):
                continue

            images = images.to(device)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            class_preds, bbox_preds = model(images)

            loss = ssd_loss(class_preds, bbox_preds, targets, default_boxes)
            if torch.isnan(loss):
                continue

            val_loss += loss.item()

            class_preds = class_preds.permute(0, 2, 3, 4, 1).reshape(-1, class_preds.shape[1])
            class_probs, class_labels = class_preds.max(dim=-1)
            mask = class_probs > threshold

            filtered_class_labels = class_labels[mask]
            ground_truth_labels = torch.cat([target['labels'].reshape(-1) for target in targets])

            # Filter out invalid class labels
            filtered_class_labels = filtered_class_labels[filtered_class_labels < num_classes]

            if len(filtered_class_labels) > len(ground_truth_labels):
                filtered_class_labels = filtered_class_labels[:len(ground_truth_labels)]
            elif len(filtered_class_labels) < len(ground_truth_labels):
                ground_truth_labels = ground_truth_labels[:len(filtered_class_labels)]

            all_preds.extend(filtered_class_labels.cpu().numpy())
            all_labels.extend(ground_truth_labels.cpu().numpy())

    val_loss /= len(data_loader)

    return val_loss



def non_max_suppression(boxes, scores, threshold=0.5):
    keep = []
    indices = scores.argsort(descending=True)
    while indices.numel() > 0:
        i = indices[0].item()
        keep.append(i)
        if indices.numel() == 1:
            break
        iou_values = iou(boxes[i], boxes[indices[1:]])
        indices = indices[1:][iou_values <= threshold]
    return keep

def denormalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(x_min * img_width, img_width))
    y_min = max(0, min(y_min * img_height, img_height))
    x_max = max(0, min(x_max * img_width, img_width))
    y_max = max(0, min(y_max * img_height, img_height))
    return [x_min, y_min, x_max, y_max]

def test_model_on_image(model, device, image_path, threshold=0.5):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    COLORS = {
        'without_mask': 'red',
        'with_mask': 'green',
        'mask_weared_incorrect': 'orange'
    }

    image = plt.imread(image_path)
    image_rgb = image[:, :, :3]  # Remove alpha channel if present
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
    height, width = image.shape[:2]

    with torch.no_grad():
        class_preds, bbox_preds = model(image_tensor)

    class_preds = class_preds.permute(0, 2, 3, 4, 1).reshape(-1, class_preds.shape[1])
    bbox_preds = bbox_preds.permute(0, 2, 3, 4, 1).reshape(-1, 4)

    class_probs, class_labels_indices = class_preds.max(dim=-1)
    mask = class_probs > threshold

    filtered_class_probs = class_probs[mask]
    filtered_class_labels_indices = class_labels_indices[mask]
    filtered_bbox_preds = bbox_preds[mask.nonzero(as_tuple=True)[0]]

    filtered_bbox_preds = filtered_bbox_preds * torch.tensor([width, height, width, height], device=device)

    valid_indices = (filtered_bbox_preds[:, 2] > filtered_bbox_preds[:, 0]) & \
                    (filtered_bbox_preds[:, 3] > filtered_bbox_preds[:, 1])
    filtered_bbox_preds = filtered_bbox_preds[valid_indices]
    filtered_class_probs = filtered_class_probs[valid_indices]
    filtered_class_labels_indices = filtered_class_labels_indices[valid_indices]

    plt.imshow(image_rgb)

    for bbox, label_index, score in zip(filtered_bbox_preds, filtered_class_labels_indices, filtered_class_probs):
        if 0 <= label_index.item() < len(class_labels):
            label = class_labels[label_index.item()]
            color = COLORS.get(label, "blue")
            bbox = bbox.cpu().numpy()
            x1, y1, x2, y2 = denormalize_bbox(bbox, width, height)
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            label_text = f"{label}: {score.item():.2f}"
            plt.text(x1, y1 - 10, label_text, color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()



def visualize_predictions_webcam(model, device, threshold=0.5):
    model.eval()
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    COLORS = {
        'without_mask': (0, 0, 255),        # Red
        'with_mask': (0, 255, 0),           # Green
        'mask_weared_incorrect': (255, 165, 0)  # Orange
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to tensor and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        height, width = frame.shape[:2]

        with torch.no_grad():
            class_preds, bbox_preds = model(frame_tensor)

        # Reshape model outputs
        class_preds = class_preds.permute(0, 2, 3, 4, 1).reshape(-1, class_preds.shape[1])
        bbox_preds = bbox_preds.permute(0, 2, 3, 4, 1).reshape(-1, 4)

        class_probs, class_labels_indices = class_preds.max(dim=-1)
        mask = class_probs > threshold

        filtered_class_probs = class_probs[mask]
        filtered_class_labels_indices = class_labels_indices[mask]
        filtered_bbox_preds = bbox_preds[mask.nonzero(as_tuple=True)[0]]

        # Denormalize bounding boxes
        filtered_bbox_preds = filtered_bbox_preds * torch.tensor([width, height, width, height], device=device)

        valid_indices = (filtered_bbox_preds[:, 2] > filtered_bbox_preds[:, 0]) & \
                        (filtered_bbox_preds[:, 3] > filtered_bbox_preds[:, 1])
        filtered_bbox_preds = filtered_bbox_preds[valid_indices]
        filtered_class_probs = filtered_class_probs[valid_indices]
        filtered_class_labels_indices = filtered_class_labels_indices[valid_indices]

        for bbox, label_index, score in zip(filtered_bbox_preds, filtered_class_labels_indices, filtered_class_probs):
            if 0 <= label_index.item() < len(class_labels):
                label = class_labels[label_index.item()]
                if label not in COLORS:
                    continue  # Skip labels not in COLORS dictionary
                color = COLORS[label]
                bbox = bbox.cpu().numpy()
                x1, y1, x2, y2 = denormalize_bbox(bbox, width, height)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_text = f"{label}: {score.item():.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def debug_outputs(images, targets, class_preds, bbox_preds):
    print("Images:", images)
    print("Targets:", targets)
    print("Class predictions:", class_preds)
    print("Bounding box predictions:", bbox_preds)

def debug_model_predictions(class_preds, bbox_preds):
    print(f"Class predictions shape: {class_preds.shape}")
    print(f"Sample class predictions: {class_preds[0, :5]}")
    print(f"Bounding box predictions shape: {bbox_preds.shape}")
    print(f"Sample bounding box predictions: {bbox_preds[0, :5]}")

def check_data_consistency(images, targets):
    if torch.isnan(images).any() or torch.isinf(images).any():
        #print("Data contains NaN or infinity values.")
        return False
    for target in targets:
        if torch.isnan(target['boxes']).any() or torch.isinf(target['boxes']).any():
            #print("Target boxes contain NaN or infinity values.")
            return False
        if torch.isnan(target['labels']).any() or torch.isinf(target['labels']).any():
            #print("Target labels contain NaN or infinity values.")
            return False
    return True

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
 
def train(default_boxes):
    early_stopping = EarlyStopping(patience=10, delta=0.0005)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model.head.set_default_boxes(default_boxes) 
    train_losses = []  
    val_losses = []  
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        start_time = time.time()

        for images, targets in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            if not check_data_consistency(images, targets):
                #print("Skipping batch due to data consistency issues.")
                continue

            optimizer.zero_grad()
            class_preds, bbox_preds = model(images)
            #debug_outputs(images, targets, class_preds, bbox_preds)

            loss = ssd_loss(class_preds, bbox_preds, targets, default_boxes)
            if torch.isnan(loss):
                #print("NaN loss encountered, skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Append train loss

        avg_val_loss = evaluate(model, val_loader, default_boxes)  # Pass default_boxes here
        val_losses.append(avg_val_loss)  # Append validation loss

        # Step the scheduler
        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, "
              f"Time: {epoch_duration:.2f} seconds")

        # Check for early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    if SAVING:
        torch.save(model.state_dict(), PATH)
        print("Final model saved to './mask_trained.pth'")

    epochs_run = len(train_losses)

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Plot Training Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(range(1, epochs_run + 1), train_losses, label="Training Loss", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Validation Loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(range(1, epochs_run + 1), val_losses, label="Validation Loss", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and Legends
    plt.title('Training and Validation Loss Over Epochs')
    plt.tight_layout()
    plt.legend()
    plt.show()



def visualize_sample(image, target):
    image = image.permute(1, 2, 0)  

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    label_map = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'}
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin * image.shape[1], ymin * image.shape[0]), (xmax - xmin) * image.shape[1], (ymax - ymin) * image.shape[0], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin * image.shape[1], ymin * image.shape[0], label_map.get(label, 'unknown'), color='white', backgroundcolor='red', fontsize=8)
    
    plt.show()



# Function to inspect unique labels in the dataset
def inspect_labels(dataset, dataloader):
    unique_labels = set()
    for images, targets in dataloader:
        for target in targets:
            labels = target['labels'].unique().cpu().numpy()
            unique_labels.update(labels)
    return unique_labels

class SSDPipelineDebugger:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def validate_predictions(self, class_preds):
        """Ensure predicted class indices are within the valid range."""
        if torch.any((class_preds < 0) | (class_preds >= self.num_classes)):
            print("[ERROR] Invalid class predictions detected:", class_preds)

    def validate_labels(self, targets):
        """Ensure ground-truth labels are within the valid range."""
        invalid_labels = torch.unique(targets[(targets < 0) | (targets >= self.num_classes)])
        if invalid_labels.numel() > 0:
            print("[ERROR] Invalid ground-truth labels detected:", invalid_labels.tolist())

    def debug_raw_class_preds(self, class_preds):
        """Log raw class predictions for inspection."""
        print("[DEBUG] Raw class predictions (sample):", class_preds.view(-1)[:10].cpu().numpy())

    def debug_target_labels(self, targets):
        """Log ground-truth target labels."""
        print("[DEBUG] Unique target labels:", torch.unique(targets).cpu().numpy())

# Example Usage
num_classes = 3  # Valid labels are [0, 1, 2]




if __name__ == '__main__':
    dataset = MaskedDataset(transforms=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = SSD(num_classes).to(device)
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    if METRICS:
        inspect_predictions(model, device, val_dataset, threshold=threshold)
        preds, labels = get_predictions_and_ground_truths(model, device, val_dataset, threshold=threshold)
        calculate_metrics(preds, labels)

    default_boxes = generate_default_boxes(device)
    if not ONLY_TESTING:
        train(default_boxes=default_boxes)

    model.load_state_dict(torch.load(PATH, map_location=device))

    if VIDEO:
        visualize_predictions_webcam(model, device,threshold=threshold)
    else:
        test_model_on_image(model, device, TEST_PATH, threshold=threshold)

    debugger = SSDPipelineDebugger(num_classes)

num_anchors = 5
height, width = 38, 38

# Example class predictions from SSDHead
class_preds = torch.randn(batch_size, num_anchors * num_classes, height, width)
class_preds = class_preds.argmax(dim=1)  # Mock predicted classes

debugger.debug_raw_class_preds(class_preds)
debugger.validate_predictions(class_preds)

# Example ground-truth targets
targets = torch.randint(0, 5, (batch_size, num_anchors * height * width))  # Mock targets, intentionally wrong range

debugger.debug_target_labels(targets)
debugger.validate_labels(targets)

# Ensure proper training by clamping or masking invalid predictions
class_preds = torch.clamp(class_preds, min=0, max=num_classes - 1)
