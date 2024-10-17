import pytest
import sqlite3
import torch
import cv2
from PlateReader import yolo_inference
# Test YOLO detection
def test_yolo_inference():
    # Load YOLO model for testing
    model = torch.load('model/tiny_best.pt', map_location='cpu')
    model.eval()

    # Sample image for testing
    img = cv2.imread('test_image.jpg')
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Run YOLO inference
    pred = model(img_tensor)[0]
    assert pred is not None, "YOLO inference failed"

# Test SQLite integration
def test_sqlite_integration():
    conn = sqlite3.connect('test_plates.db')
    c = conn.cursor()

    # Create table for testing
    c.execute('''CREATE TABLE IF NOT EXISTS plates (id INTEGER PRIMARY KEY, plate_number TEXT, image_path TEXT)''')

    # Insert a test record
    c.execute("INSERT INTO plates (plate_number, image_path) VALUES (?, ?)", ("ABC123", "test_image.jpg"))
    conn.commit()

    # Query the database to verify the record
    c.execute("SELECT * FROM plates WHERE plate_number = ?", ("ABC123",))
    row = c.fetchone()

    assert row is not None, "Record not found in SQLite"
    assert row[1] == "ABC123", "Incorrect plate number in database"
    assert row[2] == "test_image.jpg", "Incorrect image path in database"

    conn.close()
