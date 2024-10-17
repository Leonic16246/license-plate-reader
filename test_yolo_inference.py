import pytest
import numpy as np
from unittest.mock import patch
from PlateReader import yolo_inference

@patch('PlateReader.attempt_load')  # Mock YOLO model loading
@patch('PlateReader.cv2.cvtColor')  # Mock image processing
@patch('PlateReader.non_max_suppression')  # Mock NMS
def test_yolo_inference(mock_nms, mock_cvtColor, mock_attempt_load):
    mock_model = mock_attempt_load.return_value
    mock_model.return_value = np.array([[[1, 1, 100, 100, 0.8, 0]]])  # Mock YOLO output
    
    mock_nms.return_value = [np.array([[1, 1, 100, 100, 0.8, 0]])]
    mock_cvtColor.return_value = np.zeros((640, 480, 3))  # Mock image
    
    frame = np.zeros((640, 480, 3))  # Mock frame
    predictions = yolo_inference(frame, mock_model)
    
    assert predictions is not None
    assert len(predictions[0]) > 0
