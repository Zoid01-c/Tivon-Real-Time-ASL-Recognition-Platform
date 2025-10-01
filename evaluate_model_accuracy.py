import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score

# ====== Paths ======
model_path = r"E:\CPP\Grok\Models\american-sign-language-tensorflow2-american-sign-language-v1"  # Path to the Kaggle ASL model (.h5 or SavedModel format)
test_folder_path = r"E:\CPP\Grok\asl_alphabet_test\asl_alphabet_test"  # Path to test images

# ====== Class Labels ======
class_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "space", "nothing"
]

# ====== Load the Kaggle ASL Model ======
print("Loading model...")
model = tf.keras.models.load_model(model_path)  # Change to .h5 or SavedModel path
print("Model loaded successfully.\n")

# ====== Helper to Extract Label from Filename ======
def extract_label_from_filename(filename):
    """
    Extracts label prefix from filename like 'A12.jpg' or 'space_01.jpg'
    """
    base = os.path.splitext(filename)[0]
    for label in class_labels:
        if base.startswith(label):
            return label
    return None  # Unknown or invalid label

# ====== Evaluation Function ======
def evaluate_model_on_flat_folder(test_folder):
    y_true = []
    y_pred = []
    total_images = 0

    for fname in os.listdir(test_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(test_folder, fname)
        true_label = extract_label_from_filename(fname)

        if true_label is None:
            print(f"⚠️  Skipping unknown label file: {fname}")
            continue

        try:
            # 💡 Change here: resizing to 224x224 as expected by the model
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array, verbose=0)
            pred_index = np.argmax(pred[0])
            pred_label = class_labels[pred_index]

            y_true.append(true_label)
            y_pred.append(pred_label)

            total_images += 1

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    if total_images > 0:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✅ Processed {total_images} images")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    else:
        print("🚫 No images were processed. Please check the folder and file format.")

# ====== Run Evaluation ======
evaluate_model_on_flat_folder(test_folder_path)
