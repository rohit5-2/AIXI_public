
# checklist before run
# 1. Change the annotations file path
# 2. Change the data folder path
# 3. Change the tensorboard log path (runs/runX)
# 4. Change the model save path
# 5. Change / create the checkpoint save path

import os
import io
import re
import sys
import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F
from dataset import CustomDataset
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
from utils import parse_train_metrics
from utils import parse_eval_metrics
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2


annotation_file_path = 'path/to/annotations.csv' 
train_data_folder = 'train/example_data/train'
validation_data_folder = 'train/example_data/validation'
tensorboard_log_path = 'train/tensorboard_logs/run_1'
model_save_path = 'train/model_final_epoch.pth'
checkpoint_dir = 'train/checkpoints/'


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', box_detection_per_img=300)
    model.roi_heads.box_nms_thresh = 0.5 # 0.5 is default
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    if train:
        return A.Compose([
            A.OneOf([
                A.RandomCrop(width=512, height=512),
                A.LongestMaxSize(max_size=1024),
                A.LongestMaxSize(max_size=512)
                ], p=1), 
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.5,  brightness_by_max=False, always_apply=False),
            ToTensorV2() 
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def main():

    writer = SummaryWriter(tensorboard_log_path)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    annotations_file = annotation_file_path
    

    train_dataset_part1 = CustomDataset(annotations_file, train_data_folder, get_transform(train=False)) # easiest way to augment on top of existing data
    train_dataset_part2 = CustomDataset(annotations_file, train_data_folder, get_transform(train=True))
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_part1, train_dataset_part2])
    validation_dataset = CustomDataset(annotations_file, validation_data_folder, get_transform(train=False))

    print('Train dataset length:', len(train_dataset), '(incl. augmented data)')
    print('Validation dataset length:', len(validation_dataset))


    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=utils.collate_fn
    )

    data_loader_validation = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        collate_fn=utils.collate_fn
    )


    model = get_model_instance_segmentation(num_classes)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )


    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )


    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
        train_metrics_str = str(train_loss)
        train_metrics = parse_train_metrics(train_metrics_str) # function in utils.py to get train eval metrics

        for metric_name, metrics_value in train_metrics.items():
            writer.add_scalar(metric_name, metrics_value, epoch)

        lr_scheduler.step()
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        eval_metrics = evaluate(model, data_loader_validation, device=device)
        sys.stdout = old_stdout
        eval_metrics_str = mystdout.getvalue()
        eval_metrics = parse_eval_metrics(eval_metrics_str) # function in utils.py to get ouputted metrics from CocoEvaluator taken from text output in terminal
        for metric_name, metrics_value in eval_metrics.items():
            writer.add_scalar(metric_name, metrics_value, epoch)

        torch.save(model.state_dict(), f'{checkpoint_dir}model_epoch_{epoch}.pth')
        print(f'Checkpoint saved as model_epoch_{epoch}.pth')


    torch.save(model.state_dict(), model_save_path)
    print('Model saved as model_final_epoch.pth')
    writer.close()

if __name__ == '__main__':
    main()    