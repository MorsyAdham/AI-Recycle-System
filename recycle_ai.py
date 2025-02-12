import os
import time
import json
import cv2
import shutil
import subprocess
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
from ultralytics import YOLO
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from collections import Counter


class DatasetHandler:
    """
    Handles dataset loading, augmentation, and balancing.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def visualize_dataset_distribution(self):
        """
        Visualizes dataset class distribution.
        """
        labels = []
        for class_folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, class_folder)
            if os.path.isdir(folder_path):
                labels.extend([class_folder] * len(os.listdir(folder_path)))

        class_counts = Counter(labels)
        plt.bar(class_counts.keys(), class_counts.values(), color="blue")
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Images")
        plt.title("Dataset Distribution")
        plt.xticks(rotation=45)
        plt.show()

    def augment_data(self, save_path="augmented_dataset"):
        """
        Applies data augmentation to balance the dataset.
        """
        os.makedirs(save_path, exist_ok=True)
        augmentor = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ])

        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            save_class_path = os.path.join(save_path, class_folder)
            os.makedirs(save_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                augmented = augmentor(image=image)['image']
                aug_img_name = f"aug_{img_name}"
                cv2.imwrite(os.path.join(save_class_path, aug_img_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        print("Data Augmentation Completed!")


class YOLOModel:
    """
    Manages YOLOv8 object detection model.
    """

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def train(self, dataset_yaml, epochs=50, imgsz=640):
        """
        Trains the YOLOv8 model.
        """
        self.model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz)

    def detect(self, image):
        """
        Runs detection on a given image.
        """
        results = self.model(image)
        return results


class EfficientNetModel:
    """
    Manages EfficientNet classification model.
    """

    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        output = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, dataset_path, epochs=20):
        """
        Trains the classification model.
        """
        datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
        train_generator = datagen.flow_from_directory(dataset_path, target_size=(224, 224))

        history = self.model.fit(train_generator, epochs=epochs)
        self.visualize_training(history)

        # Save model in `.keras` format
        self.model.save("models/efficientnet.keras")

    def visualize_training(self, history):
        """
        Plots training loss and accuracy.
        """
        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Accuracy", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.show()


class Deployment:
    """
    Handles deployment for laptop testing & edge device.
    """

    def __init__(self, model_type="laptop"):
        self.model_type = model_type
        self.yolo = YOLOModel("models/best.pt")
        self.classifier = tf.keras.models.load_model("models/efficientnet.keras")

    def run(self):
        """
        Runs inference based on the selected deployment type.
        """
        cap = cv2.VideoCapture(0)  # Use camera

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.detect(frame)
            for result in detections:
                if result.boxes.xyxy:
                    x1, y1, x2, y2, conf, cls = result.boxes.xyxy[0]
                    detected_material = result.names[int(cls)]

                    # Classification
                    frame_resized = cv2.resize(frame, (224, 224)) / 255.0
                    material_type = self.classifier.predict(np.expand_dims(frame_resized, axis=0))
                    print(f"Detected: {detected_material}, Classified as: {np.argmax(material_type)}")

            cv2.imshow("Sorting Bin", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
