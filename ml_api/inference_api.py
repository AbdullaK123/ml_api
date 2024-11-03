from fastapi import FastAPI, Request, HTTPException
import numpy as np
from PIL import Image
import cv2
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from pathlib import Path
from ml_api.model_manager import model_manager
import sys
import uvicorn
import gc

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to show all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('app.log')      # Also save to file
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Set all relevant loggers to DEBUG level
logging.getLogger('ml_api').setLevel(logging.DEBUG)
logging.getLogger('transformers').setLevel(logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)  # Keep PIL logs less verbose
logging.getLogger('matplotlib').setLevel(logging.INFO)  # Keep matplotlib logs less verbose

# Test the logger
logger.debug("Debug message - most detailed")
logger.info("Info message - general information")
logger.warning("Warning message - something to watch out for")
logger.error("Error message - something went wrong")


try:
    # Get models when needed
    logger.info("Loading models...")
    background_remover = model_manager.get_background_remover()
    ocr_model = model_manager.get_ocr_model()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise RuntimeError("Error loading models")


# Constants
MAX_IMAGE_SIZE = 1280  # Reduced for better memory usage
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
MIN_IMAGE_SIZE = 300
JPEG_QUALITY = 85
BATCH_SIZE = 1  # Process one image at a time


def optimize_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Optimize image for OCR processing"""
    # Convert to grayscale if it's color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary)

    return denoised

def smart_resize(image, target_size=MAX_IMAGE_SIZE):
    """Memory-efficient image resizing"""
    try:
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
            
        if width <= MIN_IMAGE_SIZE and height <= MIN_IMAGE_SIZE:
            return image
            
        if width > target_size or height > target_size:
            scale = target_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if isinstance(image, np.ndarray):
                resized = cv2.resize(image, (new_width, new_height), 
                                interpolation=cv2.INTER_AREA)
                ImageProcessor.cleanup_resources(image)
                return resized
            else:
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                image.close()  # Close original PIL Image
                return resized
        return image
    except Exception as e:
        logger.error(f"Resize failed: {e}")
        raise
    finally:
        gc.collect()

def validate_image(img_data: str) -> bool:
    """Validate image data"""
    try:
        # Check file size
        size = len(base64.b64decode(img_data))
        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size is {MAX_FILE_SIZE/(1024*1024)}MB"
            )
        return True
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    
def resize_image_if_needed(image):
    """Resize image if it's too large"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size
        
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        aspect_ratio = width / height
        if width > height:
            new_width = MAX_IMAGE_SIZE
            new_height = int(MAX_IMAGE_SIZE / aspect_ratio)
        else:
            new_height = MAX_IMAGE_SIZE
            new_width = int(MAX_IMAGE_SIZE * aspect_ratio)
            
        if isinstance(image, np.ndarray):
            return cv2.resize(image, (new_width, new_height))
        else:
            return image.resize((new_width, new_height), Image.LANCZOS)
    return image


# function to find corners of the page
def find_corners_of_doc(image):

    # if the image is a PIL image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # convert to gray scale, apply a bilateral filter, and adaptive thresholding to make the corners easier to detect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # find the largest contour
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # find the convex hull of the largest contour and return its approximating polygon as a numpy array
    hull = cv2.convexHull(largest_contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    return np.array(approx)


# function to do a perspective transform
def perspective_transform(image):
     # find corners of doc
    corners = find_corners_of_doc(image)

    # return original image if we have less than 4 corners
    if len(corners) < 4:
        return image

    # Ensure corners are in the correct format
    corners = np.array(corners, dtype=np.float32).squeeze()

    # If we have more than 4 corners, find the bounding rectangle
    if len(corners) > 4:
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)
    
    # Ensure we have exactly 4 corners
    if len(corners) != 4:
        print(f"Unexpected number of corners: {len(corners)}. Using original image.")
        return image

    # Reshape corners to (4, 2) if necessary
    corners = corners.reshape(4, 2)
    
    # get the dims of the output img
    width = max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[2] - corners[3])
    )
    height = max(
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2])
    )

    # define the corners of the transformed img
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype="float32"
    )

    # get the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)

    # make sure image is a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # apply the warp transformation with the matrix
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))

    # return the transformed img
    return warped


# function to preprocess input img
def preprocess(img_byte_string):
    """Preprocess the image with correct format handling"""
    try:
        # Decode base64 string to image
        decoded_data = base64.b64decode(img_byte_string)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Resize if needed
        img = resize_image_if_needed(img)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Remove background
        no_background = background_remover(pil_img)
        
        # Convert result to numpy array
        if isinstance(no_background, dict) and 'map' in no_background:
            processed_img = np.array(no_background['map'])
        else:
            processed_img = np.array(no_background)
        
        # Apply perspective transform
        transformed = perspective_transform(processed_img)
        
        # Convert to grayscale and apply thresholding
        if len(transformed.shape) == 3:
            gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
        else:
            gray = transformed
            
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresholded
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def get_ocr_results(img_byte_string):
    # preprocess the img
    preprocessed_img = preprocess(img_byte_string)

    # get the ocr text
    ocr_result = ocr_model.ocr(preprocessed_img, cls=True)

    # get bounding boxes and text
    lines = []
    box_coords = []

    for line in ocr_result[0]:
        lines.append(line[1][0])  # get the text
        x1, y1, x2, y2 = line[0]  # get the bounding box coordinates
        box_coords.append([x1, y1, x2, y2])

    return lines, box_coords

def apply_ocr_results(image, lines, box_coords):
    for line, box in zip(lines, box_coords):
        # get the coordinates of the bounding box
        x1, y1, x2, y2 = box
        # draw the bounding box on the image
        print(x1, y1, x2, y2)
        cv2.rectangle(image, (int(x1[0]), int(y1[1])), (int(x2[0]), int(y2[1])), (0, 255, 0), 2)
    return image

class ImageProcessor:
    @staticmethod
    def cleanup_resources(*arrays):
        """Clean up numpy arrays"""
        for arr in arrays:
            if arr is not None:
                del arr
        gc.collect()

    @staticmethod
    def decode_and_resize_image(img_data: str) -> np.ndarray:
        """Decode base64 and resize image with memory cleanup"""
        try:
            decoded_data = base64.b64decode(img_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Resize if needed
            resized = smart_resize(img)
            ImageProcessor.cleanup_resources(img)
            
            return resized
        finally:
            ImageProcessor.cleanup_resources(decoded_data, nparr)
            gc.collect()

    @staticmethod
    def process_image_for_ocr(image: np.ndarray) -> np.ndarray:
        """Process image for OCR with memory management"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ImageProcessor.cleanup_resources(image)
            
            # Convert to PIL for background removal
            pil_img = Image.fromarray(image_rgb)
            ImageProcessor.cleanup_resources(image_rgb)
            
            # Remove background
            no_background = model_manager.get_background_remover()(pil_img)
            ImageProcessor.cleanup_resources(pil_img)
            
            # Convert result to numpy array
            processed = np.array(no_background['map'] if isinstance(no_background, dict) else no_background)
            ImageProcessor.cleanup_resources(no_background)
            
            # Apply perspective transform
            transformed = perspective_transform(processed)
            ImageProcessor.cleanup_resources(processed)
            
            # Optimize for OCR
            optimized = optimize_image_for_ocr(transformed)
            ImageProcessor.cleanup_resources(transformed)
            
            return optimized
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
        finally:
            gc.collect()


# set up paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# template and static dirs
template_dir = project_root / "templates"
static_dir = project_root / "static"

# server logic
app = FastAPI(debug=True)

# add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# set up templates
templates = Jinja2Templates(directory=str(template_dir))

# set up root
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    
    logger.debug("Received response for /")

    try:
        response = templates.TemplateResponse("index.html", {"request": request})
        logger.debug("Template rendered successfully")
        return response
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return {"error": "An error occurred while rendering the template"}
    
# route that takes in an image from the client, preprocesses it, and then returns the image with the ocr results
@app.post("/ocr")
async def ocr(img: dict):
    """OCR endpoint with memory optimization"""
    logger.debug("OCR endpoint called")
    
    image = None
    processed_image = None
    result_image = None
    
    try:
        if not img or "img" not in img:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        validate_image(img["img"])
        
        try:
            # Process image in stages with memory cleanup
            image = ImageProcessor.decode_and_resize_image(img["img"])
            processed_image = ImageProcessor.process_image_for_ocr(image)
            ImageProcessor.cleanup_resources(image)
            image = None
            
            # Perform OCR
            ocr_result = model_manager.get_ocr_model().ocr(
                processed_image, 
                cls=True
            )
            
            if not ocr_result or not ocr_result[0]:
                return {"img": "", "text": ["No text found"]}
            
            # Draw results
            result_image = processed_image.copy()
            text_lines = []
            
            for line in ocr_result[0]:
                text = line[1][0]
                points = np.array(line[0]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
                text_lines.append(text)
            
            # Encode result
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', result_image, encode_param)
            img_str = base64.b64encode(buffer).decode()
            
            ImageProcessor.cleanup_resources(buffer)
            
            # Force cleanup
            model_manager.cleanup()
            gc.collect()
            
            return {
                "img": img_str,
                "text": text_lines
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Image processing failed")
            
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up all resources
        ImageProcessor.cleanup_resources(image, processed_image, result_image)
        gc.collect()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=5,
        timeout=120,
        limit_concurrency=20,
        backlog=100
    )