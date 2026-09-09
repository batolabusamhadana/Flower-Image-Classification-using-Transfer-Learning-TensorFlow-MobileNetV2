# Flower Image Classification using Transfer Learning

A deep learning image classification project built with TensorFlow and TensorFlow Hub that identifies 102 different flower species using transfer learning with MobileNetV2.

## Project Overview

This project uses a pre-trained MobileNetV2 model as a feature extractor and trains a custom neural network classifier to recognize flower species from images.

The model is trained on the Oxford 102 Flowers Dataset and achieves over **77% test accuracy** while significantly reducing training time through transfer learning.

## Objectives

* Explore and preprocess an image classification dataset.
* Build an efficient TensorFlow data pipeline.
* Apply transfer learning using a pre-trained CNN.
* Train and evaluate a custom classifier.
* Save and load trained models.
* Perform inference on unseen flower images.
* Visualize model predictions and confidence scores.

---

## Dataset

**Oxford 102 Flowers Dataset**

* 102 flower categories
* 8,189 images
* Images vary in size, orientation, and background

Dataset Split:

| Set        | Samples |
| ---------- | ------: |
| Training   |   1,020 |
| Validation |   1,020 |
| Testing    |   6,149 |

---

## Technologies Used

* Python
* TensorFlow
* TensorFlow Hub
* Keras
* NumPy
* Matplotlib
* PIL (Pillow)

---

## Data Preprocessing

The preprocessing pipeline includes:

* Image resizing to **224 × 224**
* Pixel normalization to the range **[0,1]**
* Dataset batching
* Data shuffling
* Prefetching for performance optimization

```python
IMG_SIZE = 224
BATCH_SIZE = 32
```

---

## Model Architecture

### Feature Extractor

* MobileNetV2 (TensorFlow Hub)
* Pre-trained on ImageNet
* Frozen during training

### Classifier

```text
MobileNetV2 Feature Extractor
            ↓
Dense (512, ReLU)
            ↓
Dropout (0.3)
            ↓
Dense (102, Softmax)
```

---

## Training Configuration

| Parameter      | Value                           |
| -------------- | ------------------------------- |
| Optimizer      | Adam                            |
| Loss Function  | Sparse Categorical Crossentropy |
| Batch Size     | 32                              |
| Epochs         | 10                              |
| Output Classes | 102                             |

---

## Results

### Test Performance

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 77.54% |
| Loss     | 0.8630 |

```text
Test Accuracy: 0.7754
Test Loss: 0.8630
```

---

## Inference Pipeline

The project includes custom functions for:

### Image Processing

* Load image with PIL
* Resize to 224×224
* Normalize pixel values
* Convert to NumPy array

### Prediction

```python
probs, classes = predict(image_path, model, top_k=5)
```

Outputs:

* Top K predicted classes
* Prediction probabilities
* Human-readable flower names

---

## Visualization

The project visualizes:

### Training Metrics

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

### Prediction Results

* Input image
* Top 5 predicted flower classes
* Confidence score bar chart

---

## Model Saving

The trained model is saved using Keras:

```python
model.save("flower_classifier.h5")
```

and can later be loaded for inference:

```python
loaded_model = tf.keras.models.load_model(
    "flower_classifier.h5",
    custom_objects={"KerasLayer": hub.KerasLayer}
)
```

---

## Future Improvements

* Data augmentation
* Fine-tuning MobileNetV2 layers
* Learning rate scheduling
* Early stopping
* Deployment as a web application
* Conversion to TensorFlow Lite for mobile devices

---

## Key Concepts Demonstrated

* Deep Learning
* Computer Vision
* Transfer Learning
* TensorFlow Pipelines
* Model Evaluation
* Multi-Class Classification
* Neural Networks
* Image Processing
* Keras Model Deployment

---

## Author

**Batol Abu Samhadana**

Computer Engineering Student at Birzeit University

Interested in Artificial Intelligence, Machine Learning, Computer Vision, Automation, and Data Science.
