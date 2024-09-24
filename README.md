# Number Plate Detection – ENSE810 Group Project

**Group Members:**
- Samuel Meads (20113456)
- Leon Lee (20125718)
- Henry Hu (20114453)

## Project Overview
This project is a license plate detection system using a Raspberry Pi with a connected camera to capture and log license plate information in real-time. The system detects number plates, differentiates them from other objects, and stores the collected data securely.

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
  - Python with OpenCV and OpenALPR for number plate detection.
  - SQLite for local data storage.
  - LAMP stack for web-based data management.
- **Multithreading:** Used in Python to capture plates and store data concurrently without blocking.

## How It Works
1. The camera captures video input, processed by a Python program utilizing OpenCV and OpenALPR.
2. Detected number plates are stored in an SQLite database.
3. Data is managed and displayed via a LAMP stack, allowing for real-time access to captured information.

## Setup Instructions
1. Clone this repository to your Raspberry Pi.
2. Install dependencies using `pip install -r requirements.txt`.
3. Configure the camera and database settings in `config.py`.
4. Run `PlateReader.py` to start capturing and logging plates.
5. Access the web interface by navigating to the Raspberry Pi's IP address in your browser.

---

*This project is developed as part of ENSE810 – Embedded Systems coursework.*
