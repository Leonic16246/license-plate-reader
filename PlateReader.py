import torch
import cv2
import easyocr  
from models.experimental import attempt_load
from utils.general import non_max_suppression 
import time 
import sqlite3
import os

def insert_into_db(plate, path_to_image, confidence):
    try:
        # Connect to SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect('database/plates.db')  # Adjust path as needed
        cursor = conn.cursor()

        # Create a table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                path_to_image TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert data into the table
        cursor.execute('''
            INSERT INTO plates (plate_number, path_to_image, confidence)
            VALUES (?, ?, ?)
        ''', (plate, path_to_image, confidence))

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()
        print(f"Plate {plate} inserted into database.")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

# Function to save the plate image and return its path
def save_plate_image(plate_region, plate_number):
    # Create directory if it doesn't exist
    save_dir = '/home/pi64/plate_images/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the image with the plate number as part of the filename
    image_path = os.path.join(save_dir, f"{plate_number}_{time.time()}.jpg")
    cv2.imwrite(image_path, plate_region)
    return image_path

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Force CPU usage

device = torch.device("cpu")

# model = attempt_load('runs/train/yolov7_tiny_numberplatesVRPD2/weights/best.pt')
model = attempt_load('runs/tiny_best.pt')
model.eval()

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)  # Camera FPS

frame_skip = 96  # Perform YOLO inference every _th frame
frame_count = 0  # Initialize a frame counter

# Variables to store the last detected bounding box and plate number
last_detected_plate = None
last_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only run YOLO inference every `frame_skip` frames
    if frame_count % frame_skip == 0:
        # Convert the frame from BGR to RGB format and directly use it without resizing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float()  # Convert to tensor and move to the correct device
        img = img.permute(2, 0, 1).unsqueeze(0)  # Change dimension order and add batch dimension
        img /= 255.0  # Normalize the image to [0, 1] range as expected by the model

        # Measure YOLO inference time
        start_time_yolo = time.time()  # Start timer for YOLO
        # Perform inference on the selected frame
        with torch.no_grad():  # Disable gradients for inference
            pred = model(img)[0]  # Get the prediction
        # Apply non-max suppression to the predictions
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        end_time_yolo = time.time()  # End timer for YOLO
        yolo_inference_time = end_time_yolo - start_time_yolo
        print(f"YOLO Inference Time: {yolo_inference_time:.4f} seconds")

        # Process predictions
        if pred[0] is not None:
            for detection in pred[0]:
                # Extract bounding box and confidence
                x1, y1, x2, y2, conf, cls = detection[:6]

                # Shrink the bounding box slightly (reduce width and height by a percentage)
                shrink_factor = 0.05  # Shrink by 5%
                width_reduction = int((x2 - x1) * shrink_factor)
                height_reduction = int((y2 - y1) * shrink_factor)
                x1 += width_reduction
                y1 += height_reduction
                x2 -= width_reduction
                y2 -= height_reduction

                # Ensure the bounding box coordinates are within frame bounds
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Extract the region of interest (the detected license plate)
                plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

                # Check if plate_region is valid (non-empty)
                if plate_region.size > 0:
                    # Measure EasyOCR inference time
                    start_time_easyocr = time.time()  # Start timer for EasyOCR
                    # Use EasyOCR to read the number plate
                    result = reader.readtext(plate_region)
                    end_time_easyocr = time.time()  # End timer for EasyOCR
                    easyocr_inference_time = end_time_easyocr - start_time_easyocr
                    print(f"EasyOCR Inference Time: {easyocr_inference_time:.4f} seconds")

                    if len(result) > 0:
                        last_detected_plate = result[0][1]  # Store the detected text
                        last_bbox = (int(x1), int(y1), int(x2), int(y2))  # Store the bounding box
                        confidence = float(conf)  # Assuming 'conf' is the confidence score from YOLO detection
                        
                        # Save the plate image and get the image path
                        image_path = save_plate_image(plate_region, last_detected_plate)
                        
                        # Insert the detected plate into the SQLite database
                        insert_into_db(last_detected_plate, image_path, confidence)
                    else:
                        print("Invalid plate region (empty).")

    # If we have a previous detection, keep the bounding box and label on the frame
    if last_bbox is not None and last_detected_plate is not None:
        x1, y1, x2, y2 = last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_detected_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and plate number
    cv2.imshow('Webcam Feed', frame)

    frame_count += 1  # Increment the frame counter

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()