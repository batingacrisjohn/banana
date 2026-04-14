
"""
Flask backend for YOLO Banana Detection
Loads best.pt trained model and provides API endpoints for object detection
Serves the web app frontend directly
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Read HTML frontend
with open('index.html', 'r', encoding='utf-8') as f:
    HTML_CONTENT = f.read()

# Load YOLO model
MODEL_PATHS = [
    'last.pt',
    'best.pt',
    'banana_model.pt',
    'runs/detect/train/weights/best.pt',
    'yolov8n.pt',
    'yolov8m.pt'
]

# Detect banana or finger labels in the model class names
BANANA_SEARCH_TERMS = ('banana', 'saba', 'banana_bunch', 'finger')


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def find_banana_class_ids(names):
    return [int(k) for k, v in names.items() if any(term in str(v).lower() for term in BANANA_SEARCH_TERMS)]


MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError(
        'No YOLO model found. Place your trained model as best.pt, banana_model.pt, or put yolov8n.pt / yolov8m.pt in the folder.'
    )

model = YOLO(MODEL_PATH)
BANANA_CLASS_IDS = find_banana_class_ids(model.names)
BANANA_CLASS_NAMES = [model.names[i] for i in BANANA_CLASS_IDS]
print(f"✓ YOLO model loaded from {MODEL_PATH}")
print(f"✓ Model class names: {model.names}")
print(f"✓ Target class ids: {BANANA_CLASS_IDS} | names: {BANANA_CLASS_NAMES}")

# Configuration
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45
MAX_SIZE = 1280


def resize_image(image, max_size=MAX_SIZE):
    """Resize image if necessary while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image


def draw_boxes(image, results):
    """Draw bounding boxes and labels on image"""
    if results[0].boxes is None:
        return image
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names.get(int(cls_id), 'object')
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 8), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(data_url):
    """Convert base64 data URL to OpenCV image"""
    # Handle both data:image/jpeg;base64,... and raw base64
    if isinstance(data_url, str) and data_url.startswith('data:'):
        data_url = data_url.split(',')[1]
    
    image_data = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(image_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def file_to_image(file_storage):
    image = Image.open(file_storage.stream)
    image = image.convert('RGB')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def get_request_image():
    """Extract an image either from JSON base64 data or multipart file upload."""
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        if 'image' not in request.files:
            return None, 'Missing file field "image" in multipart request'

        file = request.files['image']
        if file.filename == '':
            return None, 'No file selected'
        if not allowed_file(file.filename):
            return None, 'Unsupported file type. Use JPG or PNG.'

        try:
            return file_to_image(file), None
        except Exception as e:
            return None, f'Could not decode uploaded file: {e}'

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return None, 'Missing image data'

    try:
        return base64_to_image(data['image']), None
    except Exception as e:
        return None, f'Could not decode base64 image: {e}'


@app.route('/', methods=['GET'])
def index():
    """Serve the web app frontend"""
    return HTML_CONTENT


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': MODEL_PATH,
        'model_file': MODEL_PATH,
        'model_loaded': True
    }), 200


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint
    Expects JSON with 'image' field containing base64 encoded image
    Returns detected objects with bounding boxes and confidence scores
    """
    try:
        image, error = get_request_image()
        if error:
            return jsonify({'error': error}), 400

        kg_per_piece = request.json.get('kg_per_piece', 2.5)

        original_h, original_w = image.shape[:2]
        
        # Resize if necessary
        image = resize_image(image, MAX_SIZE)
        
        # Run inference
        results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        print(f"Detection results: {len(results[0].boxes) if results[0].boxes is not None else 0} boxes found")  # Debug
        if results[0].boxes is not None:
            print(f"Confidences: {results[0].boxes.conf.cpu().numpy()}")  # Debug
        
        # Extract detections
        detections = []
        banana_warning = None
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Calculate scale factor
            scale_y = original_h / image.shape[0]
            scale_x = original_w / image.shape[1]
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                class_id = int(cls_id)
                class_name = model.names.get(class_id, 'unknown')
                if BANANA_CLASS_IDS and class_id not in BANANA_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = box
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': class_id,
                    'class_name': class_name
                })

        if not BANANA_CLASS_IDS:
            banana_warning = (
                'Loaded model does not contain a banana or finger class label. '
                'Please use a banana-trained model named best.pt or banana_model.pt.'
            )
        
        # Draw boxes on image
        image_with_boxes = draw_boxes(image, results)
        output_image = image_to_base64(image_with_boxes)
        
        # Count bananas (assuming class 0 is banana or use first class found)
        banana_count = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        estimated_weight = banana_count * kg_per_piece
        
        response = {
            'success': True,
            'detections': detections,
            'count': banana_count,
            'avg_confidence': float(avg_confidence),
            'estimated_weight': float(estimated_weight),
            'output_image': output_image,
            'image_size': {
                'width': original_w,
                'height': original_h
            },
            'banana_class_ids': BANANA_CLASS_IDS,
            'banana_class_names': BANANA_CLASS_NAMES,
        }
        if banana_warning:
            response['warning'] = banana_warning
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detect-stream', methods=['POST'])
def detect_stream():
    """
    Fast detection for streaming/live mode
    Returns minimal data for real-time performance
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        kg_per_piece = data.get('kg_per_piece', 2.5)

        # Decode image
        image = base64_to_image(data['image'])
        original_h, original_w = image.shape[:2]
        
        # Resize for faster processing
        image = resize_image(image, 640)
        
        # Run inference
        results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        # Extract detections (minimal data for speed)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            scale_y = original_h / image.shape[0]
            scale_x = original_w / image.shape[1]
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                class_id = int(cls_id)
                if BANANA_CLASS_IDS and class_id not in BANANA_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = box
                detections.append({
                    'x1': float(x1 * scale_x),
                    'y1': float(y1 * scale_y),
                    'x2': float(x2 * scale_x),
                    'y2': float(y2 * scale_y),
                    'conf': float(conf),
                    'class': class_id,
                    'class_name': model.names.get(class_id, 'unknown')
                })
        
        banana_count = len(detections)
        avg_confidence = np.mean([d['conf'] for d in detections]) if detections else 0
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': banana_count,
            'avg_confidence': float(avg_confidence),
            'estimated_weight': float(banana_count * kg_per_piece)
        }), 200
        
    except Exception as e:
        print(f"Error during stream detection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and class names"""
    return jsonify({
        'model': os.path.basename(MODEL_PATH),
        'model_file': MODEL_PATH,
        'classes': {int(k): v for k, v in model.names.items()},
        'banana_class_ids': BANANA_CLASS_IDS,
        'banana_class_names': BANANA_CLASS_NAMES,
        'conf_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'max_size': MAX_SIZE
    }), 200


if __name__ == '__main__':
    print("=" * 50)
    print("YOLO Banana Detection Backend")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {model.names}")
    print("\nStarting Flask server on http://0.0.0.0:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
