# all images belonging to the same video should be in a single folder with some form of sequential numbering in the filenames.

import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

IMAGES_FOLDER = 'inference/example_data/exp_009'
DETECTIONS_CSV = 'inference/detections_009.csv'
VIDEO_OUTPUT = 'inference/exp_009_detections_video.mp4'
collection_frame_rate = 6.67
confidence_threshold = 0.8  # Confidence threshold of detections

# Load bounding box data from CSV
bbox_data = pd.read_csv(DETECTIONS_CSV)
bbox_data['frame'] = bbox_data['filename'].factorize()[0]  # Assign frame numbers

# Initialize the plot
fig, ax = plt.subplots()

# Preload images to speed up animation
images = {}
for filename in bbox_data['filename'].unique():
    image_path = f"{IMAGES_FOLDER}/{filename}"
    image = cv2.imread(image_path)
    if image is not None:
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Loaded {len(images)} images")

total_frames = len(bbox_data['frame'].unique())

def update(frame_number):
    ax.clear()  # Clear the previous frame
    filename = bbox_data[bbox_data['frame'] == frame_number]['filename'].iloc[0]
    image = images.get(filename)
    
    if image is not None:
        # Display the image
        ax.imshow(image)

        # Draw each bbox for the current frame that meets the confidence threshold
        frame_data = bbox_data[(bbox_data['frame'] == frame_number) & (bbox_data['confidence'] >= confidence_threshold)]
        for _, row in frame_data.iterrows():
            rect = Rectangle((row['xmin'], row['ymin']), row['xmax'] - row['xmin'], row['ymax'] - row['ymin'], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        # Set the limits and title
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_title(f"Detections: {round(frame_number / collection_frame_rate, 1)}")

# Create the animation
ani = FuncAnimation(fig, update, frames=total_frames, repeat=False)

# Save the animation
ani.save(VIDEO_OUTPUT, writer='ffmpeg', fps=30)

print(f"Video saved to {VIDEO_OUTPUT}")
