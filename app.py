from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torchvision.ops as ops
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# === Konfigurasi ===
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'TajwidModelCNNSSD.pth'
CLASS_NAMES = ['background', 'ikhfa', 'idgham', 'idzhar', 'iklab']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Model ===
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.head.classification_head.num_classes = len(CLASS_NAMES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model = model.to(device)
model.eval()

# === Transformasi Inference ===
def get_infer_transform():
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

# === Fungsi Deteksi dan Return Image dengan Bounding Box ===
def detect_and_draw(image_path, threshold=0.5, iou_thresh=0.3):
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size
    img_resized = original_img.resize((224, 224))
    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    all_boxes = outputs['boxes']
    all_labels = outputs['labels']
    all_scores = outputs['scores']

    final_boxes, final_labels, final_scores = [], [], []
    for class_idx in range(1, len(CLASS_NAMES)):
        cls_mask = all_labels == class_idx
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        if cls_boxes.size(0) == 0:
            continue

        keep = ops.nms(cls_boxes, cls_scores, iou_thresh)
        for idx in keep:
            if cls_scores[idx] >= threshold:
                final_boxes.append(cls_boxes[idx])
                final_labels.append(class_idx)
                final_scores.append(cls_scores[idx])

    # Tidak ada deteksi
    if not final_boxes:
        return None

    # Plot hasil
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(original_img)
    scale_x = orig_w / 224
    scale_y = orig_h / 224

    for box, label, score in zip(final_boxes, final_labels, final_scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        x1 *= scale_x; x2 *= scale_x
        y1 *= scale_y; y2 *= scale_y

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor='red', linewidth=2))
        ax.text(x1, y1, f"{CLASS_NAMES[label]} ({score*100:.1f}%)",
                fontsize=25, color='red')

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# === Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result_image = detect_and_draw(filepath)

    if result_image is None:
        # Tidak terdeteksi apapun
        return jsonify({'message': 'Tajwid nun mati tidak terdeteksi'}), 204

    return send_file(result_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
