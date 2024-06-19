import os
import torch
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  
import pandas as pd

MODEL_PATH = 'path/to/model.pth'
IMAGES_FOLDER = 'path/to/images/folder/'
CSV_FILE = 'path/to/detections.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", box_detections_per_img=400)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

model = load_model()
print("Model loaded successfully")
images = sorted([img for img in os.listdir(IMAGES_FOLDER) if img.endswith('.jpg')])

# DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence'])

# Perform predictions and save to CSV
for filename in images:
    image_path = os.path.join(IMAGES_FOLDER, filename)
    image = read_image(image_path).to(device)
    image = image.float() / 255  # Normalize the image to 0-1
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)

    # Extract predictions
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        predictions_df.loc[len(predictions_df)] = [filename, xmin, ymin, xmax, ymax, label, score]

# Save predictions to CSV
predictions_df.to_csv(CSV_FILE, index=False)

print(f"Detections saved to {CSV_FILE}")