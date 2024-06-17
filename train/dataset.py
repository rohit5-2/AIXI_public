# UPDATED to include:
# - albumentations for data augmentation
# - correct handling of bboxes and labels during transformations

# annotations accessed from CSV in the format:
# filename,xmin,ymin,xmax,ymax,class


import os
import torch
from torchvision.io import read_image
import pandas as pd
import numpy as np
from torchvision.transforms.v2 import functional as F


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transforms):
        self.transforms = transforms
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)

        image_column = 'filename'
        all_images = self.img_labels[image_column].unique()
        self.class_to_int = {"grain": 1}  

        # Filter the images to include only those present in the img_dir
        actual_image_files = set(os.listdir(img_dir))
        self.images = [img for img in all_images if img in actual_image_files]

        # Group bounding box data by image
        self.bbox_data = {img: {'boxes': [], 'labels': []} for img in self.images}
        for _, row in self.img_labels.iterrows():
            img = row[image_column]
            if img in self.images:  # Only include data for images present in img_dir
                box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                label = self.class_to_int[row['class']]
                self.bbox_data[img]['boxes'].append(box)
                self.bbox_data[img]['labels'].append(label)


    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)  # Reads image using torchvision
        img = F.convert_image_dtype(img, dtype=torch.float32)  # Normalize the image

        boxes = self.bbox_data[img_name]['boxes']
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = self.bbox_data[img_name]['labels']
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # area calculated as (ymax - ymin) * (xmax - xmin)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)  # make sure iscrowd length matches number of boxes

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": int(idx)  # keep image id as int otherwise PyTorch Faster-RCNN implementation will break
        }

        if self.transforms is not None:
            sample = {
                'image': np.array(img.permute(1, 2, 0)),  # Convert to numpy array in HWC format
                'bboxes': boxes.tolist(),  # Convert to list for albumentations
                'labels': labels.tolist()  # Convert labels to list
            }
            transformed = self.transforms(**sample)
            img = transformed['image']

            # Update boxes and labels if 'bboxes' key exists in transformed and it's not empty
            if 'bboxes' in transformed and len(transformed['bboxes']) > 0:
                valid_boxes = []
                valid_labels = []
                for box, label in zip(transformed['bboxes'], transformed['labels']):
                    x_min, y_min, x_max, y_max = box
                    # Check if the box has positive area - mostly redundant as manually removed incorrect boxes from CSV, but should keep.
                    if x_max > x_min and y_max > y_min:
                        valid_boxes.append(box)
                        valid_labels.append(label)
                target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32)
                target['labels'] = torch.tensor(valid_labels, dtype=torch.int64)
                target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
            else:
                # if no valid boxes, fill empty tensors
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.tensor([], dtype=torch.int64)
                target['area'] = torch.tensor([], dtype=torch.float32)
                
        # convert img back to a PyTorch tensor if not already - old error check but keeping for safety
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # convert HWC to CHW format for PyTorch

        return img, target 


    def get_item_by_filename(self, filename):
        if filename in self.bbox_data:
            idx = list(self.images).index(filename)
            return self.__getitem__(idx)
        else:
            raise ValueError(f"No entry with filename {filename}")

    def __len__(self):
        return len(self.images)