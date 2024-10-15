import torch
import cv2

# Load the test model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-tiny.pt', source='github')

# Start capturing from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)
    
    # Render results on frame
    results.render()

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
