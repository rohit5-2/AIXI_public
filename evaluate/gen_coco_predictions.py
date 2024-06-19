# the following script generates a COCO predictions .json file to be used with the fork of review object detection metrics tool
# designed to be used on a test set which has associated ground truth coco annotation file ready

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import json
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.v2.functional import convert_image_dtype

# Model path and images folder
MODEL_PATH = 'path/to/trained/model.pth'
IMAGES_FOLDER = 'path/to/test/set/images/folder'
OUTPUT_FILE = 'predictions_coco_format_test_set.json'

# Load the trained model
def load_model(num_classes, model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", box_detection_per_img=300)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Dataset class for prediction
class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = list(sorted(os.listdir(img_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img_for_size = Image.open(img_path)
        width, height = img_for_size.size
        img = read_image(img_path)
        img = convert_image_dtype(img, dtype=torch.float32) 
        return img, self.imgs[idx], (width, height)

    def __len__(self):
        return len(self.imgs)

def format_coco_predictions(predictions):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "grain"}]  
    image_id = 1
    annotation_id = 1

    for prediction in predictions:
        images.append({
            "id": image_id,
            "file_name": prediction["file_name"],
            "width": prediction["width"],
            "height": prediction["height"],
        })
        
        for bbox, category_id, score in zip(prediction["bbox"], prediction["category_ids"], prediction["scores"]):
            bbox = [round(coord) for coord in bbox]  # Ensure bounding box coordinates are integers
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],  # width * height
                "iscrowd": 0,
                "segmentation": [],
                "score": score  
            })
            annotation_id += 1
        
        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    return coco_format

def generate_coco_predictions(model, dataset, device):
    raw_predictions = []
    model = model.to(device)
    for img, img_name, (width, height) in dataset:
        img = img.unsqueeze(0).to(device)
        predictions = model(img)
        predictions = predictions[0]

        image_predictions = {
            "image_id": img_name,
            "file_name": img_name,
            "width": width,
            "height": height,
            "bbox": [],
            "category_ids": [],
            "scores": []
        }
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i].detach().cpu().numpy().tolist()
            score = predictions['scores'][i].detach().cpu().item()
            category_id = predictions['labels'][i].detach().cpu().item()
            box_int = [round(box[0]), round(box[1]), round(box[2] - box[0]), round(box[3] - box[1])]
            image_predictions["bbox"].append(box_int)
            image_predictions["category_ids"].append(category_id)
            image_predictions["scores"].append(score)
        
        raw_predictions.append(image_predictions)

    coco_formatted_predictions = format_coco_predictions(raw_predictions)
    return coco_formatted_predictions


if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  
    
    model = load_model(num_classes, MODEL_PATH)
    model.to(device)
    
    prediction_dataset = PredictionDataset(IMAGES_FOLDER)
    coco_predictions = generate_coco_predictions(model, prediction_dataset, device)
    
    with open(OUTPUT_FILE, 'w') as file:
        json.dump(coco_predictions, file, indent=4)

    print("Predictions saved in COCO format")
