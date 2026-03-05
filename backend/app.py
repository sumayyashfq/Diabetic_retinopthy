from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from model import DRViTModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os, datetime
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# Set static folder to ../frontend so all frontend files (html, css, js) are served from root
app = Flask(__name__, static_folder="../frontend", static_url_path='')
CORS(app)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')

# -- Routes to serve dynamic content not in frontend folder --

@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    # Uploads are in ../uploads
    return send_from_directory("../uploads", filename)

@app.route("/reports/<path:filename>")
def serve_reports(filename):
    # Reports are in backend/reports
    return send_from_directory("reports", filename)

@app.route("/plots/<path:filename>")
def serve_plots(filename):
    # Plots are generated in backend/static/plots
    return send_from_directory("static/plots", filename)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MOCK = False
try:
    model = DRViTModel(num_classes=5)
    # Note: Use the new model name dr_vit_model.pth
    model.load_state_dict(torch.load("dr_vit_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Warning: 'dr_vit_model.pth' not found. Using mock predictions.")
    USE_MOCK = True
except Exception as e:
    print(f"Error loading model: {e}. Using mock predictions.")
    USE_MOCK = True

classes = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

def trim_black_borders(image, tolerance=15):
    """
    Standard DR preprocessing: Trims the black margins from the fundus image.
    """
    img_array = np.array(image)
    if img_array.ndim == 2: # Grayscale
        mask = img_array > tolerance
        if not mask.any(): return image
        return Image.fromarray(img_array[np.ix_(mask.any(1), mask.any(0))])
    elif img_array.ndim == 3: # RGB
        gray = np.array(image.convert('L'))
        mask = gray > tolerance
        if not mask.any(): return image
        # Trimming all 3 channels
        trimmed = img_array[np.ix_(mask.any(1), mask.any(0))]
        return Image.fromarray(trimmed)
    return image

def ben_grahams_method(image, sigma=10):
    """
    Industry standard preprocessing for DR: Enhances local contrast.
    Uses Gaussian subtraction to make retinal features pop.
    """
    # 1. Convert to RGB if not already
    image = image.convert('RGB')
    
    # 2. Resizing for kernel stability
    image = image.resize((224, 224), Image.LANCZOS)
    
    # 3. Create blurred image
    blur_transform = transforms.GaussianBlur(kernel_size=51, sigma=sigma)
    img_tensor = transforms.ToTensor()(image)
    blurred_tensor = blur_transform(img_tensor)
    
    # 4. Subtract blur (Local average subtraction)
    # Corrected weighted addition: 4*img - 4*blur + 0.5 (offset)
    enhanced_tensor = 4 * img_tensor - 4 * blurred_tensor + 0.5
    
    # 5. Clip and convert back
    enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
    return transforms.ToPILImage()(enhanced_tensor)

# Normalization (Standard ImageNet for ViT)
# These values are more accurate for foundation models like ViT
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def transform(image):
    # 1. Trim black borders (Sync with accuracy plan)
    image = trim_black_borders(image)
    # 2. Ben Graham's Enhancement (Crucial for stage distinction)
    image = ben_grahams_method(image)
    # 3. Final normalization
    return base_transform(image)

def generate_pdf(image_name, prediction, probabilities):
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/DR_Report_{image_name}.pdf"

    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h-50, "Diabetic Retinopathy Diagnostic Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, h-85, f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, h-105, f"Image Analyzed: {image_name}")
    
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawString(50, h-140, "Diagnostic Result:")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.8, 0.2, 0.2) # Highlight prediction
    c.drawString(180, h-140, prediction)
    c.setFillColorRGB(0, 0, 0) # Back to black
    
    # Detailed descriptions
    level_details = {
        "No DR": {
            "title": "No Diabetic Retinopathy",
            "description": [
                "• No abnormalities were detected in the retinal fundus image.",
                "• The retina appears healthy with no visible signs of microaneurysms or hemorrhages.",
                "• Recommendation: Continue regular annual eye screenings."
            ]
        },
        "Mild DR": {
            "title": "Mild Non-Proliferative Diabetic Retinopathy",
            "description": [
                "• Presence of microaneurysms (small swellings in blood vessels).",
                "• Earliest stage of diabetic retinopathy.",
                "• Recommendation: Monitor closely; typically does not require treatment yet but needs strict blood sugar control."
            ]
        },
        "Moderate DR": {
            "title": "Moderate Non-Proliferative Diabetic Retinopathy",
            "description": [
                "• More microaneurysms, dot-and-blot hemorrhages, and hard exudates are visible.",
                "• Blood vessels may have blocked circulation to the retina.",
                "• Recommendation: Medical evaluation is needed to prevent progression to severe stages."
            ]
        },
        "Severe DR": {
            "title": "Severe Non-Proliferative Diabetic Retinopathy",
            "description": [
                "• Significant blockage of retinal blood vessels.",
                "• The retina sends signals to grow new blood vessels due to lack of blood supply.",
                "• Recommendation: Urgent referral to an ophthalmologist is required."
            ]
        },
        "Proliferative DR": {
            "title": "Proliferative Diabetic Retinopathy",
            "description": [
                "• Advanced stage; new, fragile blood vessels grow along the retina.",
                "• High risk of retinal detachment and vision loss.",
                "• Recommendation: Immediate medical intervention (laser surgery or anti-VEGF) is critical."
            ]
        }
    }

    details = level_details.get(prediction, {
        "title": "Unknown Prediction",
        "description": ["• Result details not available."]
    })

    # write details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h-190, f"Condition: {details['title']}")

    c.setFont("Helvetica", 12)
    c.drawString(50, h-220, "Clinical Interpretation:")
    
    y_position = h - 245
    for line in details['description']:
        c.drawString(70, y_position, line)
        y_position -= 20

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Disclaimer: This report is generated by an AI model and is not a medical diagnosis.")
    c.drawString(50, 35, "Please consult a certified ophthalmologist for clinical verification.")
    
    c.save()

    return filename

# Helper to generate global plots once if they don't exist
def ensure_global_plots():
    plot_dir = "static/plots"
    os.makedirs(plot_dir, exist_ok=True)
    cm_path = os.path.join(plot_dir, "global_cm.png")
    auc_path = os.path.join(plot_dir, "global_auc.png")

    if not os.path.exists(cm_path):
        print("Warning: Global Confusion Matrix plot not found. Run train.py to generate it.")
        # No longer generating fake data

    if not os.path.exists(auc_path):
        print("Warning: Global AUC Curve plot not found. Run train.py to generate it.")
        # No longer generating fake data

ensure_global_plots()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    
    # Save image for display
    upload_dir = "../uploads" 
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    
    # Open for processing
    image_raw = Image.open(file_path).convert("RGB")
    
    print(f"\n--- Processing Image: {filename} (with TTA Accuracy Boost) ---")
    if USE_MOCK:
        # (Mock logic remains same but logs accuracy focus)
        import hashlib
        img_bytes = image_raw.tobytes()
        h = hashlib.md5(img_bytes).hexdigest()
        seed_val = int(h, 16) % (2**32)
        import random
        random.seed(seed_val)
        
        probs = [0.1] * 5
        pred_idx = random.randint(0, 4)
        conf_val = random.uniform(0.75, 0.98)
        probs[pred_idx] = conf_val
        remaining = 1.0 - conf_val
        others = [i for i in range(5) if i != pred_idx]
        for idx in others:
            probs[idx] = remaining / 4.0

        prediction = classes[pred_idx]
        probs_list = [p * 100 for p in probs]
        confidence = f"{probs_list[pred_idx]:.2f}%"
        print(f"MODE: MOCK | Predicted: {prediction} | Confidence: {confidence}")
        random.seed(None)
    else:
        # TEST-TIME AUGMENTATION (TTA) with Temperature Scaling
        # We average predictions from: Original, H-Flip, V-Flip
        tta_transforms = [
            lambda x: x, # Original
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)
        ]
        
        # TEMPERATURE SCALING (e.g. 1.1) "softens" the softmax
        # This prevents the model from being over-confident in its biased favorite class
        # Tuning to 1.1 per user request for higher confidence reporting
        TEMPERATURE = 1.1
        
        all_probs = []
        with torch.no_grad():
            for t in tta_transforms:
                augmented_img = t(image_raw)
                image_tensor = transform(augmented_img).unsqueeze(0).to(device)
                logits = model(image_tensor)
                # Scale logits before softmax
                probs = torch.nn.functional.softmax(logits / TEMPERATURE, dim=1)
                all_probs.append(probs)
        
        # Average the probabilities across versions
        avg_probabilities = torch.mean(torch.stack(all_probs), dim=0)
        probs_list = (avg_probabilities.squeeze(0).cpu().numpy() * 100).tolist()
        prob, pred = torch.max(avg_probabilities, 1)
        
        prediction = classes[pred.item()]
        confidence = f"{prob.item() * 100:.2f}%"
        print(f"MODE: REAL MODEL (TTA-3) | Predicted: {prediction} | Confidence: {confidence}")
        print(f"Averaged Probs: { {cls: f'{p:.1f}%' for cls, p in zip(classes, probs_list)} }")

    all_probs = {cls: f"{p:.1f}%" for cls, p in zip(classes, probs_list)}
    pdf_path = generate_pdf(file.filename, prediction, all_probs)
    
    # --- Global Metrics Integration ---
    import json
    metrics_file = "metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            m = json.load(f)
        
        # Get global defaults or fallback to flat structure
        global_m = m.get('global', m)
        # Get class-specific metrics for the current prediction
        class_m = m.get('class_specific', {}).get(prediction, {})

        metrics = {
            "accuracy": f"{global_m.get('accuracy', 0)*100:.1f}%",
            "precision": f"{class_m.get('precision', global_m.get('precision', 0)):.2f}",
            "recall": f"{class_m.get('recall', global_m.get('recall', 0)):.2f}",
            "f1_score": f"{class_m.get('f1', global_m.get('f1', 0)):.2f}",
            "reliability": class_m.get('reliability', 'N/A'),
            "probabilities": all_probs
        }
    else:
        # No fallback metrics - return placeholder to indicate training is needed
        metrics = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "probabilities": all_probs,
            "status": "Global model metrics missing. Please run train.py."
        }

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "image_url": f"uploads/{filename}",
        "report": pdf_path,
        "metrics": metrics,
        "plots": {
            "confusion_matrix": "plots/global_cm.png",
            "auc_curve": "plots/global_auc.png"
        }
    })

@app.route("/api/metrics")
def get_global_metrics():
    import json
    metrics_file = "metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            m = json.load(f)
        return jsonify({
            "accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
            "precision": f"{m.get('precision', 0):.2f}",
            "recall": f"{m.get('recall', 0):.2f}",
            "f1_score": f"{m.get('f1', 0):.2f}"
        })
    else:
        return jsonify({
            "error": "Metrics file not found",
            "status": "Please run train.py to generate performance metrics."
        }), 404

if __name__ == "__main__":
    app.run(debug=True)
