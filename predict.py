import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json

# --- Settings ---
IMG_SIZE = 224
MODEL_PATH = '/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/flower_classifier.h5'
LABEL_MAP = '/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/label_map.json'
TEST_IMAGE = '/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/test_images/orange_dahlia.jpg'
TOP_K = 5

# --- Load the trained model ---
loaded_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'KerasLayer': hub.KerasLayer}
)

# --- Load the label mapping ---
with open(LABEL_MAP, 'r') as f:
    class_names = json.load(f)

# --- Image processing ---
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image.numpy()

# --- Prediction function ---
def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    img_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(img_batch)
    probs, classes = tf.math.top_k(predictions, k=top_k)
    classes = [str(c) for c in classes.numpy()[0]]
    flower_names = [class_names[c] for c in classes]
    return probs.numpy()[0], flower_names

# --- Run prediction ---
probs, flowers = predict(TEST_IMAGE, loaded_model, TOP_K)

# --- Show results ---
print("Top probabilities:", probs)
print("Top flowers:", flowers)
