"""
Flask backend for MobileNetV3 Banana Detection
Properly loads full model checkpoint and provides accurate detection API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATHS = ['best_light.pth', 'best.pth', 'banana_model.pth', 'model.pth']
CLASS_NAMES = {0: 'banana', 1: 'finger'}
CONF_THRESHOLD = 0.70
IOU_THRESHOLD  = 0.30
MAX_SIZE       = 1280
WINDOW_SIZE    = 224
STRIDE         = 112   # 50% overlap — fewer windows, less duplication

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# MODEL LOADING — robust full-model loader
# ──────────────────────────────────────────────
def find_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_model(model_path, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    print(f"\n=== Loading model from: {model_path} ===")

    # ── Case 1: checkpoint IS the model object (torch.save(model, path))
    if not isinstance(checkpoint, dict):
        print("✓ Loaded as full model object")
        model = checkpoint
        model.to(device).eval()
        return model, device

    # ── Print checkpoint keys for debugging
    top_keys = list(checkpoint.keys())[:15]
    print(f"Checkpoint top-level keys: {top_keys}")

    # ── Case 2: wrapped dict with 'model' or 'model_state_dict'
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found key: model_state_dict")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found key: state_dict")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found key: model")
    else:
        # The checkpoint itself is the state_dict
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")

    # ── Detect architecture (large vs small) from key shapes
    use_large = True
    for k, v in state_dict.items():
        if 'features.0.0.weight' in k:
            use_large = (v.shape[1] == 3)  # both large/small have this
            break
    # Distinguish large vs small by a key that differs
    # MobileNetV3-large has 16 feature blocks, small has 13
    feature_indices = [int(k.split('.')[1]) for k in state_dict.keys()
                       if k.startswith('features.') and k.split('.')[1].isdigit()]
    max_feat = max(feature_indices) if feature_indices else 0
    use_large = max_feat >= 14

    print(f"Detected architecture: MobileNetV3-{'large' if use_large else 'small'}")
    print(f"Max feature block index: {max_feat}")

    # ── Normalize key prefixes (remove 'base.', 'module.', 'backbone.' etc.)
    def strip_prefix(sd, prefixes):
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            for pfx in prefixes:
                if new_k.startswith(pfx):
                    new_k = new_k[len(pfx):]
                    break
            new_sd[new_k] = v
        return new_sd

    state_dict = strip_prefix(state_dict, ['base.', 'module.', 'backbone.', 'model.', 'net.'])

    # ── Build model
    ModelClass = mobilenet_v3_large if use_large else mobilenet_v3_small
    model = ModelClass(pretrained=False, num_classes=num_classes)

    # ── Try strict load first (best case — weights match exactly)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Full model loaded (strict=True) — classifier head included!")
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        # ── Try non-strict (weights mismatch in some layers)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"⚠ Non-strict load: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
        # Warn if classifier head is missing — predictions will be random
        clf_missing = [k for k in missing if 'classifier' in k]
        if clf_missing:
            print("⚠ WARNING: Classifier head NOT loaded — predictions may be inaccurate!")
            print("  This means the .pth file only saved features, not the full model.")
            print("  Re-export your model using: torch.save(model.state_dict(), 'best_light.pth')")
        else:
            print("✓ Classifier head loaded successfully")

    model.to(device).eval()
    return model, device


MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError(
        'No model found. Place best_light.pth or best.pth in the same folder.'
    )

model, device = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# ── Quick sanity check
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    out = model(dummy)
    probs = F.softmax(out, dim=1)[0].cpu().tolist()
print(f"✓ Sanity check — output shape: {list(out.shape)}, sample probs: {[round(p,3) for p in probs]}")
print(f"✓ Device: {device}")
print(f"✓ Classes: {CLASS_NAMES}")
print(f"✓ Conf threshold: {CONF_THRESHOLD}")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def classify_window(bgr_window):
    """Run the model on a single BGR crop. Returns (class_id, confidence_dict)."""
    rgb  = cv2.cvtColor(bgr_window, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    tens = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tens)
        probs = F.softmax(out, dim=1)[0]
    return probs


def nms(detections, iou_threshold=IOU_THRESHOLD):
    """Non-maximum suppression for classifier-based sliding-window detections."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        x1b, y1b, x2b, y2b = best['bbox']
        ab = (x2b - x1b) * (y2b - y1b)
        remaining = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            xi1, yi1 = max(x1b, x1), max(y1b, y1)
            xi2, yi2 = min(x2b, x2), min(y2b, y2)
            if xi2 > xi1 and yi2 > yi1:
                inter = (xi2 - xi1) * (yi2 - yi1)
                union = ab + (x2 - x1) * (y2 - y1) - inter
                iou   = inter / union if union > 0 else 0
                if iou > iou_threshold:
                    continue  # suppress
            remaining.append(d)
        detections = remaining
    return keep


def sliding_window_detect(image, conf_threshold=CONF_THRESHOLD):
    """
    Sliding window detection using the classifier.
    Uses STRIDE = WINDOW_SIZE // 2 to reduce duplicate detections.
    Returns only banana-class detections above conf_threshold.
    """
    h, w = image.shape[:2]
    detections = []

    # If image fits in one window, classify whole image
    if h <= WINDOW_SIZE and w <= WINDOW_SIZE:
        probs = classify_window(image)
        banana_conf = probs[0].item()
        if banana_conf > conf_threshold:
            detections.append({
                'bbox': [0, 0, w, h],
                'confidence': float(banana_conf),
                'class': 0,
                'class_name': 'banana'
            })
        return detections

    # Multi-scale sliding window
    scales = [1.0, 0.75]
    for scale in scales:
        sw = int(w * scale)
        sh = int(h * scale)
        if sw < WINDOW_SIZE or sh < WINDOW_SIZE:
            continue
        scaled = cv2.resize(image, (sw, sh)) if scale != 1.0 else image

        for y in range(0, sh - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, sw - WINDOW_SIZE + 1, STRIDE):
                crop  = scaled[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
                probs = classify_window(crop)
                banana_conf = probs[0].item()
                if banana_conf > conf_threshold:
                    # Map coords back to original image space
                    ox = int(x / scale)
                    oy = int(y / scale)
                    os = int(WINDOW_SIZE / scale)
                    detections.append({
                        'bbox': [float(ox), float(oy), float(ox + os), float(oy + os)],
                        'confidence': float(banana_conf),
                        'class': 0,
                        'class_name': 'banana'
                    })

    if detections:
        detections = nms(detections, iou_threshold=IOU_THRESHOLD)
    return detections


def resize_image(image, max_size=MAX_SIZE):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale   = max_size / max(h, w)
        image   = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def draw_detections(image, detections):
    """Draw bounding boxes on image."""
    colors = [
        (245, 196, 0), (45, 186, 88), (0, 200, 255),
        (255, 100, 0), (200, 0, 255), (255, 60, 107)
    ]
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf  = det['confidence']
        color = colors[i % len(colors)]

        # Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label background
        label     = f"banana {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(image, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(image, label, (x1 + 3, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Summary banner
    count = len(detections)
    banner = f"Bananas detected: {count}"
    if detections:
        avg_c = np.mean([d['confidence'] for d in detections])
        banner += f"  |  avg conf: {avg_c:.0%}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, banner, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 196, 0), 2)
    return image


def img_to_b64(image):
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')


def decode_image(data):
    """Accept base64 data URL or multipart file."""
    if isinstance(data, str):
        if data.startswith('data:'):
            data = data.split(',', 1)[1]
        raw = base64.b64decode(data)
    else:
        raw = data
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': os.path.basename(MODEL_PATH),
        'device': str(device),
        'conf_threshold': CONF_THRESHOLD,
        'classes': CLASS_NAMES
    })


@app.route('/model-info')
def model_info():
    return jsonify({
        'model': os.path.basename(MODEL_PATH),
        'type': 'MobileNetV3',
        'classes': CLASS_NAMES,
        'conf_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'stride': STRIDE,
        'window_size': WINDOW_SIZE,
        'device': str(device)
    })


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Accept both JSON (base64) and multipart
        if request.content_type and 'multipart' in request.content_type:
            if 'image' not in request.files:
                return jsonify({'error': 'Missing image field'}), 400
            file  = request.files['image']
            raw   = file.read()
            image = decode_image(raw)
            kg_per_piece = float(request.form.get('kg_per_piece', 0.20))
        else:
            data = request.get_json(silent=True) or {}
            if 'image' not in data:
                return jsonify({'error': 'Missing image data'}), 400
            image = decode_image(data['image'])
            kg_per_piece = float(data.get('kg_per_piece', 0.20))

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        orig_h, orig_w = image.shape[:2]
        image = resize_image(image)

        detections = sliding_window_detect(image, conf_threshold=CONF_THRESHOLD)

        banana_count   = len(detections)
        avg_confidence = float(np.mean([d['confidence'] for d in detections])) if detections else 0.0
        estimated_weight = round(banana_count * kg_per_piece, 2)

        # Draw and encode result image
        image_out = draw_detections(image.copy(), detections)
        output_b64 = img_to_b64(image_out)

        return jsonify({
            'success': True,
            'count': banana_count,
            'detections': detections,
            'avg_confidence': avg_confidence,
            'estimated_weight': estimated_weight,
            'output_image': output_b64,
            'image_size': {'width': orig_w, 'height': orig_h},
            'conf_threshold': CONF_THRESHOLD
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/detect-stream', methods=['POST'])
def detect_stream():
    """Lightweight endpoint for live/streaming mode."""
    try:
        data = request.get_json(silent=True) or {}
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        image = decode_image(data['image'])
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        kg_per_piece = float(data.get('kg_per_piece', 0.20))

        # Shrink for speed in stream mode
        image = resize_image(image, 640)

        detections = sliding_window_detect(image, conf_threshold=CONF_THRESHOLD)

        count          = len(detections)
        avg_confidence = float(np.mean([d['confidence'] for d in detections])) if detections else 0.0

        minimal = [{
            'x1': d['bbox'][0], 'y1': d['bbox'][1],
            'x2': d['bbox'][2], 'y2': d['bbox'][3],
            'conf': d['confidence']
        } for d in detections]

        return jsonify({
            'success': True,
            'count': count,
            'detections': minimal,
            'avg_confidence': avg_confidence,
            'estimated_weight': round(count * kg_per_piece, 2)
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  MobileNetV3 Banana Detection API")
    print("=" * 55)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {device}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Thresh : {CONF_THRESHOLD}")
    print(f"  Stride : {STRIDE}px (window: {WINDOW_SIZE}px)")
    print(f"  Server : http://0.0.0.0:5000")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
