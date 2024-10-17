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

class YOLOModel: # Yolo model for plate detection
    def __init__(self, model_path='runs/tiny_best.pt'):
        self.model = attempt_load(model_path)
        self.model.eval()

    def inference(self, frame): # Search for plates
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] YOLO: Searching for plate...")
        start_time_yolo = time.time()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        img /= 255.0

        with torch.no_grad():
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.60)

        end_time_yolo = time.time()
        print(f"[{thread_name}] YOLO Inference Time: {end_time_yolo - start_time_yolo:.4f} seconds")
        return pred


class PlateProcessor: # EasyOCR model to read detected plates
    def __init__(self, db_path='database/plates.db', save_dir='/home/pi64/plate_images/'):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.db_path = db_path
        self.save_dir = save_dir

    def process_plate(self, plate_region, plate_coordinates, confidence):
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] OCR: Reading plate...")
        x1, y1, x2, y2 = plate_coordinates

        start_time_easyocr = time.time()
        result = self.reader.readtext(plate_region)
        print(f"[{thread_name}] EasyOCR Inference Time: {time.time() - start_time_easyocr:.4f} seconds")

        if result:
            plate_number = result[0][1]
            image_path = self.save_plate_image(plate_region, plate_number)
            self.insert_into_db(plate_number, image_path, confidence)
        else:
            print(f"[{thread_name}] No valid plate detected.")

    def save_plate_image(self, plate_region, plate_number): # Save plate image to local storage
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        filename = f"{plate_number}_{int(time.time())}.jpg"
        image_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(image_path, plate_region)
        return image_path

    def insert_into_db(self, plate, path_to_image, confidence):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    path_to_image TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''') # Creates tables if there isnt one
            cursor.execute('INSERT INTO plates (plate_number, path_to_image, confidence) VALUES (?, ?, ?)', 
                           (plate, path_to_image, confidence))
            conn.commit()
            conn.close()
            print(f"Plate {plate} inserted into database.")
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")


class PlateReaderApp:
    def __init__(self, yolo_model, plate_processor, frame_skip=120): # Interval for yolo inference
        self.yolo_model = yolo_model
        self.plate_processor = plate_processor
        self.frame_skip = frame_skip

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0

        with ThreadPoolExecutor(max_workers=2) as executor: # Max no. of threads 
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow('Webcam Feed', frame)

                if frame_count % self.frame_skip == 0:
                    yolo_future = executor.submit(self.yolo_model.inference, frame)
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
                                executor.submit(self.plate_processor.process_plate, plate_region, plate_coordinates, confidence)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_model = YOLOModel('model/tiny_best.pt') # yolov7 model
    plate_processor = PlateProcessor()
    app = PlateReaderApp(yolo_model, plate_processor)
    app.run_camera()
