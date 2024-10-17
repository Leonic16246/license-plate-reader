import pytest
from unittest.mock import patch, MagicMock
from PlateReader import process_plate

@patch('PlateReader.save_plate_image')
@patch('PlateReader.insert_into_db')
@patch('PlateReader.easyocr.Reader')
def test_process_plate(mock_easyocr, mock_insert_into_db, mock_save_plate_image):
    mock_ocr_instance = MagicMock()
    mock_easyocr.return_value = mock_ocr_instance
    mock_ocr_instance.readtext.return_value = [("plate_region", "ABC123")]
    
    plate_region = MagicMock()
    plate_coordinates = (1, 1, 100, 100)
    confidence = 0.95
    
    # Mock save_plate_image to return a valid image path
    mock_save_plate_image.return_value = "/path/to/image.jpg"
    
    process_plate(plate_region, plate_coordinates, confidence)
    
    mock_ocr_instance.readtext.assert_called_once_with(plate_region)
    mock_save_plate_image.assert_called_once_with(plate_region, "ABC123")
    mock_insert_into_db.assert_called_once_with("ABC123", "/path/to/image.jpg", 0.95)
