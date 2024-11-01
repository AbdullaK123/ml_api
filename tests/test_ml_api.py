import sys
import pytest
import tempfile
print(f"Python version: {sys.version}")
print(f"Pytest version: {pytest.__version__}")

print("Starting imports...")
from ml_api import __version__
from ml_api.inference_api import (
    find_corners_of_doc,
    perspective_transform,
    preprocess,
    get_ocr_results,
    apply_ocr_results
)
import base64
import cv2
import numpy as np
import os
print("Imports completed")

def setup_module(module):
    print("Setting up module")
    print(f"Discovered tests: {[name for name in dir(module) if name.startswith('test_')]}")

TEST_IMAGES_PATH = "/mnt/c/Users/abdul/OneDrive/Desktop/Test Images"
print(f"TEST_IMAGES_PATH: {TEST_IMAGES_PATH}")
print(f"Does path exist? {os.path.exists(TEST_IMAGES_PATH)}")

test_images = [os.path.join(TEST_IMAGES_PATH, img) for img in os.listdir(TEST_IMAGES_PATH)]
print(f"Number of test images found: {len(test_images)}")

test_img = cv2.imread(test_images[0])
print(f"Test image shape: {test_img.shape if test_img is not None else 'None'}")

@pytest.fixture(scope="module")
def image_fixture():
    print("Setting up image fixture")
    img = cv2.imread(test_images[0])
    if img is None:
        pytest.skip("Test image could not be loaded")
    return img

def test_find_corners_of_doc(image_fixture):
    print("Running test_find_corners_of_doc")
    corners = find_corners_of_doc(image_fixture)
    print(f"Corners found: {corners}")
    assert corners is not None
    assert corners.shape[-1] == 2  # Should have x,y coordinates
    assert len(corners) >= 4  # Should have at least 4 corners

def test_perspective_transform(image_fixture):
    print("Running test_perspective_transform")
    transformed = perspective_transform(image_fixture)
    print(f"Transformed image shape: {transformed.shape if transformed is not None else 'None'}")
    assert transformed is not None
    assert len(transformed.shape) == 3  # Should be a 3D array (height, width, channels)
    assert transformed.shape[2] == 3  # Should have 3 color channels

def test_preprocess(image_fixture):
    print("Running test_preprocess")
    _, buffer = cv2.imencode('.jpg', image_fixture)
    img_byte_string = base64.b64encode(buffer).decode('utf-8')
    transformed = preprocess(img_byte_string)
    print(f"Preprocessed image shape: {transformed.shape if transformed is not None else 'None'}")
    assert transformed is not None
    assert len(transformed.shape) == 2  # Should be a 2D array (grayscale)

def test_get_ocr_results(image_fixture):
    print("Running test_get_ocr_results")

    # Resize image for testing
    max_size = 1000
    h, w = image_fixture.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_fixture = cv2.resize(image_fixture, new_size, interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', image_fixture)
    img_byte_string = base64.b64encode(buffer).decode('utf-8')
    
    # Test OCR results
    lines, boxes = get_ocr_results(img_byte_string)
    
    print(f"Number of text lines found: {len(lines)}")
    print(f"Sample text: {lines[:3] if lines else 'None'}")
    
    assert lines is not None
    assert boxes is not None
    assert len(lines) == len(boxes)  # Should have same number of lines and boxes
    
    # Verify box coordinates format
    if boxes:
        assert len(boxes[0]) == 4  # Each box should have 4 points
        assert all(isinstance(coord, (list, tuple, np.ndarray)) for coord in boxes[0])

def test_apply_ocr_results(image_fixture):
    print("Running test_apply_ocr_results")

    # Resize image for testing
    max_size = 1000
    h, w = image_fixture.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_fixture = cv2.resize(image_fixture, new_size, interpolation=cv2.INTER_AREA)

    # Get OCR results
    _, buffer = cv2.imencode('.jpg', image_fixture)
    img_byte_string = base64.b64encode(buffer).decode('utf-8')
    lines, boxes = get_ocr_results(img_byte_string)
    
    # Apply boxes to image
    result_image = apply_ocr_results(image_fixture.copy(), lines, boxes)
    print(f"Result image shape: {result_image.shape if result_image is not None else 'None'}")
    
    assert result_image is not None
    assert result_image.shape == image_fixture.shape  # Should maintain original dimensions
    assert len(result_image.shape) == 3  # Should be a 3D array
    assert result_image.shape[2] == 3  # Should have 3 color channels
    
    # Check that image was modified (should be different from original)
    assert not np.array_equal(result_image, image_fixture)

def test_version():
    print("Running test_version")
    print(f"Version: {__version__}")
    assert __version__ == '0.1.0'

if __name__ == "__main__":
    print("Running tests...")
    pytest.main([__file__, "-v", "-s"])