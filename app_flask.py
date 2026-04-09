from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# ==========================================
# FLASK APP - SEGMENTATION LÉSIONS CUTANÉES
# ==========================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Créer dossier uploads
os.makedirs('uploads', exist_ok=True)

# Charger le modèle
MODEL_PATH = 'melanoma_model.h5'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modèle chargé:", MODEL_PATH)
    else:
        print("⚠️  Modèle non trouvé:", MODEL_PATH)

# Charger le modèle au démarrage
load_model()

def preprocess_image(image_path):
    """
    Prétraitement identique au training
    """
    # Lecture
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Image non valide")
    
    # Resize + Normalisation
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    
    return img_normalized, img_resized

def image_to_base64(img_array):
    """Convertir image numpy vers base64"""
    if len(img_array.shape) == 2:  # Grayscale
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    else:  # RGB
        img_pil = Image.fromarray(cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Page d'accueil"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nom de fichier vide'}), 400
        
        if model is None:
            return jsonify({'success': False, 'error': 'Modèle non chargé'}), 500
        
        # Sauvegarder temporairement
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prétraitement
        img_normalized, img_original = preprocess_image(filepath)
        
        # Prédiction
        input_tensor = np.expand_dims(img_normalized, axis=0)
        prediction = model.predict(input_tensor, verbose=0)[0]
        prediction_binary = (prediction > 0.5).astype(np.uint8)
        
        # Calcul métriques
        lesion_area = np.sum(prediction_binary) / (128 * 128) * 100
        confidence = np.max(prediction) * 100
        
        # Évaluation risque
        if lesion_area < 5:
            risk = "Faible"
            risk_color = "#10b981"  # Vert
        elif lesion_area < 15:
            risk = "Modéré"
            risk_color = "#f59e0b"  # Orange
        else:
            risk = "Élevé"
            risk_color = "#ef4444"  # Rouge
        
        # Convertir images en base64
        original_b64 = image_to_base64(img_normalized)
        mask_b64 = image_to_base64(prediction_binary.squeeze())
        
        # Nettoyer fichier temporaire
        os.remove(filepath)
        
        # Retourner résultats
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'segmentation_mask': mask_b64,
            'metrics': {
                'confidence': f"{confidence:.1f}",
                'surface': f"{lesion_area:.1f}",
                'risk': risk,
                'risk_color': risk_color
            }
        })
    
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 SERVEUR FLASK - SEGMENTATION IA")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("🔬 Modèle: melanoma_model.h5")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)