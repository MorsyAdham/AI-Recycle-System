# AI-Recycle-System

## **📌 Overview**
This project enables **automatic detection and classification** of recyclable materials (plastic, aluminum, glass, paper, etc.) using **YOLOv8** for object detection and **EfficientNet** for classification. The system also includes a **self-learning mechanism** that collects new images for retraining and logs misclassified items.

---

## **📌 1. Setup Instructions**

### **🔹 Prerequisites**
Ensure you have Python installed and install the required libraries:
```bash
pip install torch tensorflow opencv-python ultralytics matplotlib albumentations scikit-learn
```

### **🔹 Cloning the Repository**
```bash
https://github.com/MorsyAdham/AI-Recycle-System.git
cd recycle_ai
```

---

## **📌 2. Imported Libraries**

### **🔹 General Libraries**
- `os`, `time`, `json`, `shutil`, `subprocess`, `cv2` (OpenCV), `numpy`
| Library | Purpose |
|---------|---------|
| `os` | Handles file and directory operations |
| `time` | Manages timestamps for saving images |
| `json` | Handles structured logging (misclassifications) |
| `cv2` | (OpenCV) Handles image processing and video feed |
| `shutil` | For file operations (copying, moving data) |
| `subprocess` | Executes shell commands (e.g., YOLO training) |
| `numpy` | For array operations and preprocessing |

### **🔹 Deep Learning Libraries**
- `torch`, `tensorflow`, `ultralytics`
- `tensorflow.keras.applications`, `tensorflow.keras.preprocessing.image`, `tensorflow.keras.layers`, `tensorflow.keras.models`
| Library | Purpose |
|---------|---------|
| `torch` | PyTorch backend for YOLOv8 |
| `tensorflow` | Keras backend for EfficientNet |
| `ultralytics` | YOLOv8 model for object detection |
| `tensorflow.keras.applications` | Pre-trained models (EfficientNetB0) |
| `tensorflow.keras.preprocessing.image` | Image processing utilities |
| `tensorflow.keras.layers` | Neural network layers |
| `tensorflow.keras.models` | Model building utilities |

### **🔹 Data Handling & Visualization**
- `matplotlib.pyplot`, `albumentations`, `sklearn.utils.class_weight`
| Library | Purpose |
|---------|---------|
| `matplotlib.pyplot` | Visualizing dataset distribution & training |
| `albumentations` | Data augmentation for better model generalization |
| `sklearn.utils.class_weight` | Handling class imbalance |
---

## **📌 3. Class & Function Descriptions**
The code consists of **five major classes**:

1. `DatasetHandler` → Manages dataset, visualization & augmentation  
2. `YOLOModel` → Handles YOLOv8 object detection model  
3. `EfficientNetModel` → Manages EfficientNet classification model  
4. `SelfLearningSystem` → Enables self-improving dataset collection  
5. `Deployment` → Runs the AI model on a **laptop** or **edge device**  

### **🔹 `DatasetHandler` (Dataset Management)**
| Method | Description |
|--------|-------------|
| `visualize_dataset_distribution()` | Displays a **bar chart** of dataset class distribution |
| `augment_data(save_path)` | Performs **data augmentation** |

#### **📌 Example Usage**
```python
dataset = DatasetHandler("datasets/recyclables")
dataset.visualize_dataset_distribution()
dataset.augment_data("datasets/augmented")
```

---

### **🔹 `YOLOModel` (Object Detection)**
| Method | Description |
|--------|-------------|
| `train(dataset_yaml, epochs, imgsz)` | Trains YOLOv8 on a dataset |
| `detect(image)` | Runs object detection on an image |

#### **📌 Example Usage**
```python
yolo = YOLOModel()
yolo.train("datasets/taco.yaml", epochs=50)
detections = yolo.detect("test_image.jpg")
```

---

### **🔹 `EfficientNetModel` (Classification)**
| Method | Description |
|--------|-------------|
| `train(dataset_path, epochs)` | Trains EfficientNet classifier on dataset |
| `visualize_training(history)` | Plots **training loss & accuracy** |

#### **📌 Example Usage**
```python
classifier = EfficientNetModel(num_classes=4)
classifier.train("datasets/classification")
```

---

### **🔹 `SelfLearningSystem` (Self-Improvement)**
| Method | Description |
|--------|-------------|
| `save_detected_image(frame, prediction)` | Saves detected images for future retraining |
| `log_misclassification(image_path, true_label, predicted_label)` | Logs misclassified items |

#### **📌 Example Usage**
```python
self_learning = SelfLearningSystem()
self_learning.save_detected_image(image, "plastic")
self_learning.log_misclassification("image.jpg", "plastic", "glass")
```

---

### **🔹 `Deployment` (Model Deployment)**
| Method | Description |
|--------|-------------|
| `run()` | Runs real-time detection using a webcam |

#### **📌 Example Usage (Laptop)**
```python
deploy = Deployment(model_type="laptop")
deploy.run()
```

#### **📌 Example Usage (Edge Device)**
```python
deploy = Deployment(model_type="edge")
deploy.run()
```

---

## **📌 4. How to Train & Deploy the Model**

### **🔹 Training YOLOv8**
```python
from recycle_ai import YOLOModel

yolo = YOLOModel()
yolo.train("datasets/taco.yaml", epochs=50)
```

### **🔹 Training EfficientNet Classifier**
```python
from recycle_ai import EfficientNetModel

classifier = EfficientNetModel(num_classes=4)
classifier.train("datasets/classification")
```

### **🔹 Running Inference on Laptop**
```python
from recycle_ai import Deployment

deploy = Deployment(model_type="laptop")
deploy.run()
```

### **🔹 Running Inference on Edge Device**
```python
deploy = Deployment(model_type="edge")
deploy.run()
```

---

## **📌 5. Advanced Features**

### **🔹 Self-Learning: Improve Model Over Time**
- The system **saves newly detected images** for future training.
- It **logs misclassifications** for dataset improvement.

```python
self_learning = SelfLearningSystem()
self_learning.save_detected_image(frame, "plastic")
self_learning.log_misclassification("image.jpg", "plastic", "glass")
```

### **🔹 Data Augmentation for Better Performance**
```python
dataset = DatasetHandler("datasets/recyclables")
dataset.augment_data("datasets/augmented")
```

---

## **📌 6. Summary**
✅ **YOLOv8 for Object Detection** (Plastic, Aluminum, Paper, Glass)  
✅ **EfficientNet for Classification** (Material Type)  
✅ **Self-Improving AI** (Collects new data & learns over time)  
✅ **Deployment on Laptop & Edge Devices**  
✅ **Dataset Handling, Augmentation, and Visualization**  

---


