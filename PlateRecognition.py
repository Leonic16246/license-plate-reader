import cv2
import numpy as np
from openalpr import Alpr  # Import the Alpr class from the OpenALPR module


# test


# Initialize OpenALPR
country = "eu"  # Change this to your desired country, e.g., "us", "eu", etc.
config_file = "C:\\Users\\Leon\\Documents\\ense810_project\\openalpr\\config\\openalpr.conf"  # Path to your OpenALPR config file
runtime_dir = "C:\\Users\\Leon\\Documents\\ense810_project\\openalpr\\runtime_data"  # Path to your OpenALPR runtime data directory

alpr = Alpr(country, config_file, runtime_dir)
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    exit()

alpr.set_top_n(3)  # Set the number of top results to consider
alpr.set_default_region("md")  # Set the default region (e.g., "md" for Maryland)

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

    # Convert the frame to a format OpenALPR can process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the license plate
    results = alpr.recognize_ndarray(rgb_frame)

    # Process the results
    for plate in results['results']:
        for candidate in plate['candidates']:
            # Draw rectangle around the detected plate
            cv2.rectangle(frame, (plate['coordinates'][0]['x'], plate['coordinates'][0]['y']),
                          (plate['coordinates'][2]['x'], plate['coordinates'][2]['y']), (0, 255, 0), 2)
            # Put text of the best plate guess
            cv2.putText(frame, candidate['plate'], (plate['coordinates'][0]['x'], plate['coordinates'][0]['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()

# Unload the ALPR instance
alpr.unload()
