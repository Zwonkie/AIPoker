import cv2
import numpy as np
import os
import glob
import pickle
from PIL import Image, ImageDraw, ImageFont
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

binaries_dir = r'c:\REPO\Antigravity\AIPoker\core\models\binaries'
os.makedirs(binaries_dir, exist_ok=True)
model_path = os.path.join(binaries_dir, 'digit_classifier.pkl')

templates_dir = r'c:\REPO\Antigravity\AIPoker\digit_templates_16x16'

X = []
y = []

# 1. Load real templates using strictly INTER_NEAREST binarization
print("Loading real templates...")
real_count = 0
for path in glob.glob(os.path.join(templates_dir, '*.png')):
    filename = os.path.basename(path)
    char = filename[0]
    if char.isdigit():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)
            flat = (resized > 127).astype(np.float32).flatten()
            X.append(flat)
            y.append(int(char))
            real_count += 1
print(f"Loaded {real_count} real digit templates.")

# 2. Generate synthetic augmented samples
print("Generating synthetic digits...")
fonts = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\trebucbd.ttf",
]
available_fonts = [f for f in fonts if os.path.exists(f)]

def augment_image(img_bin):
    augmented = [img_bin]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(img_bin, M, (16, 16), borderValue=0)
            augmented.append(shifted)
    for angle in [-6, -3, 3, 6]:
        M = cv2.getRotationMatrix2D((8, 8), angle, 1.0)
        rotated = cv2.warpAffine(img_bin, M, (16, 16), borderValue=0)
        augmented.append(rotated)
    return augmented

synthetic_count = 0
for digit in range(10):
    char_str = str(digit)
    for font_path in available_fonts:
        for font_size in range(10, 22, 2):
            try:
                pil_img = Image.new('L', (40, 40), 0)
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype(font_path, font_size)
                draw.text((10, 10), char_str, font=font, fill=255)
                
                cv_img = np.array(pil_img)
                coords = cv2.findNonZero(cv_img)
                if coords is not None:
                    bx, by, bw, bh = cv2.boundingRect(coords)
                    cropped = cv_img[by:by+bh, bx:bx+bw]
                    # Binarize crop with Otsu
                    _, binary_crop = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    # Resize to 16x16 with INTER_NEAREST
                    resized = cv2.resize(binary_crop, (16, 16), interpolation=cv2.INTER_NEAREST)
                    
                    variants = augment_image(resized)
                    for var in variants:
                        X.append((var > 127).astype(np.float32).flatten())
                        y.append(digit)
                        synthetic_count += 1
            except:
                continue

print(f"Generated {synthetic_count} synthetic samples.")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training MLP Neural Network...")
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

print(f"Train Accuracy: {clf.score(X_train, y_train):.2%}")
print(f"Test Accuracy:  {clf.score(X_test, y_test):.2%}")

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"SUCCESS: Saved trained classifier to {model_path}")
