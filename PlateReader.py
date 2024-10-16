import torch
import cv2
import easyocr  
from models.experimental import attempt_load
from utils.general import non_max_suppression 
import time 
import sqlite3
import os
import threading

# Function to handle OCR and database insertion in a separate thread
def process_plate(plate_region, plate_coordinates, confidence):
    x1, y1, x2, y2 = plate_coordinates
    # Measure EasyOCR inference time
    start_time_easyocr = time.time()
    result = reader.readtext(plate_region)
    end_time_easyocr = time.time()
    easyocr_inference_time = end_time_easyocr - start_time_easyocr
    print(f"EasyOCR Inference Time: {easyocr_inference_time:.4f} seconds")

    if len(result) > 0:
        detected_plate = result[0][1]  # Extract detected plate number
        # Save the image and get the image path
        image_path = save_plate_image(plate_region, detected_plate)
        # Insert the plate number into the database
        insert_into_db(detected_plate, image_path, confidence)
    else:
        print("No valid plate detected.")

# Function to save the plate image and return just the filename
def save_plate_image(plate_region, plate_number):
    save_dir = '/home/pi64/plate_images/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"{plate_number}_{int(time.time())}.jpg"
    image_path = os.path.join(save_dir, filename)
    cv2.imwrite(image_path, plate_region)
    return image_path

# Function to insert the plate information into the SQLite database
def insert_into_db(plate, path_to_image, confidence):
    try:
        conn = sqlite3.connect('database/plates.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                path_to_image TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('INSERT INTO plates (plate_number, path_to_image, confidence) VALUES (?, ?, ?)', 
                       (plate, path_to_image, confidence))
        conn.commit()
        conn.close()
        print(f"Plate {plate} inserted into database.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

# Initialize the EasyOCR reader and the YOLO model
reader = easyocr.Reader(['en'], gpu=False)
device = torch.device("cpu")
model = attempt_load('runs/tiny_best.pt')
model.eval()

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)

frame_skip = 120
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam Feed', frame)

    if frame_count % frame_skip == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        img /= 255.0

        start_time_yolo = time.time()
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)
        end_time_yolo = time.time()
        print(f"YOLO Inference Time: {end_time_yolo - start_time_yolo:.4f} seconds")

        if pred[0] is not None:
            for detection in pred[0]:
                x1, y1, x2, y2, conf, cls = detection[:6]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                plate_region = frame[y1:y2, x1:x2]

                if plate_region.size > 0:
                    plate_coordinates = (x1, y1, x2, y2)
                    confidence = float(conf)

                    # Spawn a EasyOCR thread
                    threading.Thread(target=process_plate, args=(plate_region, plate_coordinates, confidence)).start()

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()