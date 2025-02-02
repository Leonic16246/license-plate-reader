import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from PlateReader import PlateProcessor

@patch('PlateReader.PlateProcessor.save_plate_image')
@patch('PlateReader.PlateProcessor.insert_into_db')
@patch('PlateReader.easyocr.Reader')  # Properly mock EasyOCR globally
def test_plate_processor_insert(mock_easyocr, mock_insert_into_db, mock_save_plate_image):
    mock_ocr_instance = MagicMock()
    mock_easyocr.return_value = mock_ocr_instance  # Set up the mock instance
    mock_ocr_instance.readtext.return_value = [['plate_region', 'XYZ123']]  # Mock OCR result
    
    plate_processor = PlateProcessor()  # Create an instance of the processor
    
    # Mock plate region, coordinates, and confidence
    plate_region = MagicMock()
    plate_coordinates = (0, 0, 200, 100)
    confidence = 0.9
    
    # Call process_plate and test
    plate_processor.process_plate(plate_region, plate_coordinates, confidence)
    
    # Check if the OCR, save image, and DB insertion methods were called correctly
    mock_ocr_instance.readtext.assert_called_once_with(plate_region)
    mock_save_plate_image.assert_called_once()
    mock_insert_into_db.assert_called_once()

