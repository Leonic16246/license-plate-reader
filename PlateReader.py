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

                # Extract the region of interest (the detected license plate)
                plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

                # Measure EasyOCR inference time
                start_time_easyocr = time.time()  # Start timer for EasyOCR
                # Use EasyOCR to read the number plate
                result = reader.readtext(plate_region)
                end_time_easyocr = time.time()  # End timer for EasyOCR
                easyocr_inference_time = end_time_easyocr - start_time_easyocr
                print(f"EasyOCR Inference Time: {easyocr_inference_time:.4f} seconds")

                # If any text was detected, print it
                if len(result) > 0:
                    print(f"Detected Plate: {result[0][1]}")  # Print the detected text
        else:
            print("No predictions.")

    frame_count += 1  # Increment the frame counter

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()