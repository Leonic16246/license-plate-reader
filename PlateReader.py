import torch
import cv2
import easyocr  
from models.experimental import attempt_load
from utils.general import non_max_suppression 
import time 

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Force CPU usage

device = torch.device("cpu")
# Load your YOLOv7 model with local weights
model = attempt_load('models/best.pt')
model.eval()

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24) # camera fps

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

                    # If any text was detected, update the last detected plate and bounding box
                    if len(result) > 0:
                        last_detected_plate = result[0][1]  # Store the detected text
                        last_bbox = (int(x1), int(y1), int(x2), int(y2))  # Store the bounding box
                        print(f"Detected Plate: {last_detected_plate}")  # Print the detected text
                else:
                    print("Invalid plate region (empty).")

    # If there is a previous detection, keep the bounding box and label on the frame
    if last_bbox is not None and last_detected_plate is not None:
        x1, y1, x2, y2 = last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_detected_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and plate number
    cv2.imshow('Webcam Feed', frame)

    frame_count += 1  # Increment frame counter

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()