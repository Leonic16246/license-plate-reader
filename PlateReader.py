import cv2

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
