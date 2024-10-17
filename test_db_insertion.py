import pytest
from unittest.mock import patch
from PlateReader import insert_into_db

@patch('sqlite3.connect')  # Mock SQLite connection
def test_insert_into_db(mock_connect):
    mock_conn = mock_connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    
    insert_into_db('ABC123', '/path/to/image.jpg', 0.95)
    
    mock_cursor.execute.assert_called_with(
        'INSERT INTO plates (plate_number, path_to_image, confidence) VALUES (?, ?, ?)',
        ('ABC123', '/path/to/image.jpg', 0.95)
    )
    mock_conn.commit.assert_called_once()
