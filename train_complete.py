import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# PROJET: SEGMENTATION LÉSIONS CUTANÉES
# Architecture: U-Net
# Dataset: ISIC 2018
# Version Finale Optimale
# ==========================================

print("="*60)
print("🏥 SEGMENTATION LÉSIONS CUTANÉES - U-NET")
print("="*60)

# ==========================================
# 1. CONFIGURATION
# ==========================================

base_path = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_path, "Images_128")
mask_dir = os.path.join(base_path, "Masks_128")

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Lecture images avec support Unicode (paths arabes, etc.)"""
    if not os.path.exists(path): 
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================

print("\n📂 Chargement des données...")

X, y = [], []
img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

for img_file in img_files:
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_id = os.path.splitext(img_file)[0]
        
        # Recherche du mask correspondant
        mask_file = next((m for m in mask_files if m.startswith(img_id)), None)
        
        if mask_file:
            # Lecture RGB
            img = imread_unicode(os.path.join(img_dir, img_file), cv2.IMREAD_COLOR)
            mask = imread_unicode(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                X.append(img / 255.0)  # Normalisation [0-1]
                y.append(mask / 255.0)
        
        if len(X) >= 1000:  # Utiliser 1000 images
            break

X = np.array(X)  # (N, 128, 128, 3) RGB
y = np.expand_dims(np.array(y), axis=-1)  # (N, 128, 128, 1)

if len(X) == 0:
    print("❌ ERREUR: Aucune paire trouvée!")
    exit()

print(f"✅ {len(X)} paires chargées")
print(f"   Images: {X.shape}")
print(f"   Masks: {y.shape}")

# ==========================================
# 3. SPLIT TRAIN/VALIDATION
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Répartition:")
print(f"   Training: {len(X_train)} images")
print(f"   Validation: {len(X_val)} images")

# ==========================================
# 4. ARCHITECTURE U-NET
# ==========================================

print(f"\n🏗️ Construction U-Net...")

def build_unet(input_shape=(128, 128, 3)):
    """
    Architecture U-Net pour segmentation médicale
    
    Structure:
    - Encoder: Extraction des caractéristiques
    - Bridge: Couche intermédiaire
    - Decoder: Reconstruction du masque
    - Skip connections: Préservation des détails
    """
    
    inputs = layers.Input(input_shape)
    
    # ===== ENCODER (Contracting Path) =====
    # Block 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # ===== BRIDGE =====
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(b)
    
    # ===== DECODER (Expansive Path) =====
    # Block 1
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.concatenate([u1, c2])  # Skip connection
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    
    # Block 2
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c1])  # Skip connection
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    
    # ===== OUTPUT LAYER =====
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)
    
    model = models.Model(inputs, outputs)
    
    return model

# Création du modèle
model = build_unet()

# Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle U-Net créé!")

# ==========================================
# 5. ENTRAÎNEMENT
# ==========================================

print("\n" + "="*60)
print("🚀 DÉBUT ENTRAÎNEMENT")
print("="*60)
print("⏱️  Ceci peut prendre 20-30 minutes...")
print("💡 Conseil: Laissez tourner sans interruption\n")

# Callbacks pour améliorer l'entraînement
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# ENTRAÎNEMENT - 30 EPOCHS
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Entraînement terminé!")

# ==========================================
# 6. SAUVEGARDE DU MODÈLE
# ==========================================

model_path = 'melanoma_model.h5'
model.save(model_path)
print(f"\n💾 Modèle sauvegardé: {model_path}")

# ==========================================
# 7. VISUALISATION DES RÉSULTATS
# ==========================================

print(f"\n📊 Génération des graphiques...")

# Graphiques Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Évolution de la Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("✅ Graphiques sauvegardés: training_curves.png")

# Note: plt.show() commenté pour éviter le blocage du script
# Les graphiques sont disponibles dans training_curves.png

# ==========================================
# 8. TEST PRÉDICTION
# ==========================================

print("\n🔬 Test de prédiction...")

# Prendre une image de test
test_img = X_val[0:1]
test_mask = y_val[0:1]
pred = model.predict(test_img, verbose=0)
pred_bin = (pred > 0.5).astype(np.uint8)

# Affichage
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Image RGB", fontsize=12, fontweight='bold')
plt.imshow(test_img[0])
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Vrai Mask", fontsize=12, fontweight='bold')
plt.imshow(test_mask[0].squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Prédiction IA", fontsize=12, fontweight='bold')
plt.imshow(pred_bin[0].squeeze(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('test_prediction.png', dpi=150, bbox_inches='tight')
print("✅ Test sauvegardé: test_prediction.png")

# ==========================================
# 9. CALCUL DICE SCORE
# ==========================================

print("\n📐 Calcul Dice Score...")

def dice_score(y_true, y_pred):
    """
    Calcul du Dice Score (métrique de segmentation)
    Formule: 2 * |A ∩ B| / (|A| + |B|)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

# Calcul sur échantillon de validation
dice_scores = []
for i in range(min(10, len(X_val))):
    pred = model.predict(X_val[i:i+1], verbose=0)
    pred_bin = (pred > 0.5).astype(np.float32)
    dice = dice_score(y_val[i], pred_bin[0])
    dice_scores.append(dice)

mean_dice = np.mean(dice_scores)

print(f"\n🎯 Dice Score Moyen: {mean_dice*100:.2f}%")

# ==========================================
# 10. RÉSUMÉ FINAL
# ==========================================

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n" + "="*60)
print("📋 RÉSUMÉ DU PROJET")
print("="*60)
print(f"📊 Données:")
print(f"   - Images d'entraînement: {len(X_train)}")
print(f"   - Images de validation: {len(X_val)}")
print(f"\n🎯 Performance finale:")
print(f"   - Training Accuracy: {final_train_acc*100:.2f}%")
print(f"   - Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"   - Training Loss: {final_train_loss:.4f}")
print(f"   - Validation Loss: {final_val_loss:.4f}")
print(f"   - Dice Score: {mean_dice*100:.2f}%")
print(f"\n💾 Fichiers générés:")
print(f"   - {model_path}")
print(f"   - training_curves.png")
print(f"   - test_prediction.png")
print(f"\n Prochaine étape:")
print(f"   Exécutez: python test_final_avec_dice.py")
print("="*60)
print("\n PROJET TERMINÉ!")
print("\n Note: Les graphiques sont sauvegardés en PNG")
print("   Vous pouvez les utiliser directement dans votre rapport")