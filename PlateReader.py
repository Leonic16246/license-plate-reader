import torch
import cv2
import easyocr
import time
import sqlite3
import os
import threading
from models.experimental import attempt_load
from utils.general import non_max_suppression
from concurrent.futures import ThreadPoolExecutor

# Function to handle YOLO inference
def yolo_inference(frame, model):
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] YOLO: Searching for plate...")
    start_time_yolo = time.time()  # Start YOLO inference timer
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    img /= 255.0

    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    end_time_yolo = time.time()  # End YOLO inference timer
    yolo_inference_time = end_time_yolo - start_time_yolo
    print(f"[{thread_name}] YOLO Inference Time: {yolo_inference_time:.4f} seconds")

    return pred

# Function to save the plate image and return just the filename
def save_plate_image(plate_region, plate_number):
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Saving plate...")
    save_dir = '/home/pi64/plate_images/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{plate_number}_{int(time.time())}.jpg"
    image_path = os.path.join(save_dir, filename)
    cv2.imwrite(image_path, plate_region)
    return image_path

# Function to insert plate data into SQLite database
def insert_into_db(plate, path_to_image, confidence):
    thread_name = threading.current_thread().name
    try:
        conn = sqlite3.connect('/home/pi64/plates.db')
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
        print(f"[{thread_name}] Plate {plate} inserted into database.")
    except sqlite3.Error as e:
        print(f"[{thread_name}] An error occurred: {e}")

# Function to handle OCR and database insertion
def process_plate(plate_region, plate_coordinates, confidence):
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] OCR: Reading plate...")
    reader = easyocr.Reader(['en'], gpu=False)
    x1, y1, x2, y2 = plate_coordinates
    start_time_easyocr = time.time()
    result = reader.readtext(plate_region)
    easyocr_inference_time = time.time() - start_time_easyocr
    print(f"[{thread_name}] EasyOCR Inference Time: {easyocr_inference_time:.4f} seconds")

    if len(result) > 0:
        detected_plate = result[0][1]  # Extract detected plate number
        image_path = save_plate_image(plate_region, detected_plate)
        insert_into_db(detected_plate, image_path, confidence)
    else:
        print(f"[{thread_name}] No valid plate detected.")

def run_camera():
    # Load YOLO model without specifying the device (will automatically use GPU if available)
    model = attempt_load('runs/tiny_best.pt')
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_skip = 120
    frame_count = 0

    # Use a ThreadPoolExecutor to manage threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Always display the camera feed
            cv2.imshow('Webcam Feed', frame)

            if frame_count % frame_skip == 0:
                # Submit YOLO inference to a thread
                yolo_future = executor.submit(yolo_inference, frame, model)
                predictions = yolo_future.result()

                if predictions[0] is not None:
                    for detection in predictions[0]:
                        x1, y1, x2, y2, conf, cls = detection[:6]
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        plate_region = frame[y1:y2, x1:x2]

                        if plate_region.size > 0:
                            plate_coordinates = (x1, y1, x2, y2)
                            confidence = float(conf)
                            print(f"Plate found in frame.")
                            # Submit EasyOCR and database insertion to a thread
                            executor.submit(process_plate, plate_region, plate_coordinates, confidence)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()