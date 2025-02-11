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
git clone https://github.com/your-repo/recycle_ai.git
cd recycle_ai
```

---

## **📌 2. Imported Libraries**

### **🔹 General Libraries**
- `os`, `time`, `json`, `shutil`, `subprocess`, `cv2` (OpenCV), `numpy`

### **🔹 Deep Learning Libraries**
- `torch`, `tensorflow`, `ultralytics`
- `tensorflow.keras.applications`, `tensorflow.keras.preprocessing.image`, `tensorflow.keras.layers`, `tensorflow.keras.models`

### **🔹 Data Handling & Visualization**
- `matplotlib.pyplot`, `albumentations`, `sklearn.utils.class_weight`

---

## **📌 3. Classes & Functions**

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

## **📌 4. Flutter Mobile App**
The project includes a **Flutter mobile app** that allows users to monitor and interact with the sorting bin remotely. The app connects via **WiFi/Bluetooth** to receive real-time detection results and send control commands.

### **🔹 Features**
- 📸 **Live camera feed** with detection results.
- 📊 **Sorting statistics & history**.
- 🔄 **Manual override** to open/close the bin.
- 🌍 **Remote monitoring & control**.

### **🔹 Setup Instructions**
1. Install Flutter: [Flutter Setup Guide](https://flutter.dev/docs/get-started/install)
2. Clone the mobile app repository:
```bash
git clone https://github.com/your-repo/recycle_ai_app.git
cd recycle_ai_app
flutter pub get
```
3. Run the app:
```bash
flutter run
```

### **🔹 Example Usage**
```dart
// Connect to the sorting bin via WiFi
SortingBinController binController = SortingBinController(ip: "192.168.1.10");
binController.getLiveFeed();
```

---

## **📌 5. How to Train & Deploy the Model**

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

## **📌 6. Advanced Features**

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

## **📌 7. Summary**
✅ **YOLOv8 for Object Detection** (Plastic, Aluminum, Paper, Glass)  
✅ **EfficientNet for Classification** (Material Type)  
✅ **Self-Improving AI** (Collects new data & learns over time)  
✅ **Deployment on Laptop & Edge Devices**  
✅ **Flutter Mobile App for Monitoring & Control**  
✅ **Dataset Handling, Augmentation, and Visualization**  

---

## **📌 8. Next Steps**
1. **Train the model on TACO dataset** (or custom dataset)  
2. **Deploy on an edge device (e.g., Raspberry Pi, Jetson Nano)**  
3. **Fine-tune with real-world collected data**  

This README provides **everything needed to train, test, and deploy the AI recycling system**. 🚀

