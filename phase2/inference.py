from ultralytics import YOLO
import argparse
import os
import cv2
import torch
import csv

# Setup
HOME = os.getcwd()
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('model_file', type=str, help='model file name including directory name')
parser.add_argument('source_path', type=str, help='Path to the directory containing images for inference')
parser.add_argument('output_csv_file', type=str, help='output CSV file name including directory name')
args = parser.parse_args()

# Load the YOLO model
model_path = args.model_file
net = YOLO(model_path).to(DEVICE)

# Path to the directory containing images for inference
source_path = args.source_path

# Prepare the CSV file
csv_file_path = args.output_csv_file
with open(csv_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # csv_writer.writerow(['ImageId', 'PredictionString'])

    for image_name in os.listdir(source_path):
        image_path = os.path.join(source_path, image_name)
        im_h, im_w, _ = cv2.imread(image_path).shape
        
        # Initialize an empty prediction string
        prediction_string = ""

        # Extract labels and bounding boxes
        results = net([image_path], task='detect', imgsz=800, verbose=False, augment=True, half=True)
        
        boxes = results[0].boxes.xyxyn.cpu().numpy()  # 예측된 bounding box를 numpy array로 변환
        labels = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        # Construct the prediction string
        for label, box in zip(labels, boxes):
            xmin, ymin, xmax, ymax = box
            prediction_string += f"{int(label)+1} {round(xmin * im_w)} {round(ymin * im_h)} {round(xmax * im_w)} {round(ymax * im_h)} "

        # Write the row to the CSV file
        csv_writer.writerow([image_name, prediction_string.strip()])

print(f"Predictions saved to {csv_file_path}")