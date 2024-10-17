# Number Plate Detection – ENSE810 Group Project

**Group Members:**
- Samuel Meads (20113456)
- Leon Lee (20125718)
- Henry Hu (20114453)

## Project Overview
This project is a license plate detection system built on a **Raspberry Pi** with a connected camera to capture and log license plate information in real-time. The system detects number plates, differentiates them from other objects, and securely stores the collected data.

## User Requirements
- **Object Detection:** The system must detect whether an object is a number plate.
- **Logging Capability:** It should log detected number plate information, including text, date, and time.
- **Real-Time Processing:** The system should process video or image input without significant delays.

## System Requirements

### Functional Requirements
1. Accept video input from a connected camera and process it in real-time.
2. Accurately detect number plates from the camera feed.
3. Distinguish between number plates and other objects.
4. Log detected information, including time, date, image, and number plate as text.
5. Handle detection errors, such as false positives or unreadable plates.

### Non-Functional Requirements
1. Process each video frame within 200 milliseconds.
2. Achieve a detection accuracy of at least 95%.
3. Maintain an uptime of 99.9%.
4. Handle up to 5 simultaneous license plates without performance loss.
5. Encrypt logged data using AES-256.
6. Initial setup and configuration should take no more than 10 minutes.

## Technologies Used
- **Hardware:** Raspberry Pi, Camera Module.
- **Software:**
  - Raspberry Pi OS (Linux)
  - Python with **OpenCV** for image processing and **YOLOv7** for number plate detection.
  - **Roboflow** for dataset preparation.
  - **SQLite** for local data storage.
  - **LAMP stack** for web-based data management and access.
- **Multithreading:** Used in Python, especially for **OCR** processes, to handle concurrent plate detection and data storage.

## How It Works
1. The camera captures video input, processed by a Python program utilizing OpenCV and YOLOv7 for license plate detection.
2. Detected number plates are stored in an SQLite database, with encrypted data storage for security.
3. The LAMP stack provides a web interface to view and manage logged number plate data remotely.
4. **Multithreading** ensures smooth operation, allowing real-time processing and logging without blocking.

## Setup Instructions
1. Clone this repository to your Raspberry Pi.
2. Install yolov7 dependencies using `pip install -r requirements.txt`.
3. Install other depedencies such as EasyOCR, Threading, sqlite3, etc.
4. Set up Apache for php and move plates.php to /var/www/html/.
5. Run `python PlateReader.py` to start capturing and logging plates.
6. Access the web interface by navigating to the Raspberry Pi's IP address in your browser.

---

*This project is developed as part of ENSE810 – Embedded Systems coursework.*
