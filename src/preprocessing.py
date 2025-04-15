import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from input_handler import *

def normalize_image(image_np):
    print("Normalizing image...")  # In ra thông báo
    image_np = image_np.astype("float32")
    image_np = (image_np - 127.5) / 127.5
    return image_np

def preprocess_image(image_path, target_size=(160, 160)):
    print(f"Processing image: {image_path}")  # In ra đường dẫn ảnh hiện tại
    image_np = load_image_from_path(image_path)
    if image_np is not None:
        image_resized = resize_image(image_np, target_size)
        if image_resized is not None:
            image_normalized = normalize_image(image_resized)
            print(f"Image {image_path} processed successfully.")
            return image_normalized
    print(f"Failed to process image: {image_path}")  # Thông báo nếu không thành công
    return None

def process_gender_dataset_from_folders(base_folder_path, target_size=(160, 160)):
    images = []
    labels = []
    label_encoder = LabelEncoder()
    
    print("Starting dataset processing...")
    
    for folder in ["train", "test"]:
        folder_path = os.path.join(base_folder_path, folder)
        for gender in ["women", "men"]:
            gender_folder_path = os.path.join(folder_path, gender)
            for file_name in os.listdir(gender_folder_path):
                if file_name.endswith(".jpg"):
                    image_path = os.path.join(gender_folder_path, file_name)
                    image_processed = preprocess_image(image_path, target_size)
                    if image_processed is not None:
                        images.append(image_processed)
                        labels.append(gender)
    
    labels_encoded = label_encoder.fit_transform(labels)
    
    print("Dataset processing completed.")
    return np.array(images), np.array(labels_encoded), label_encoder

base_folder_path = "assets/images"
images, labels, label_encoder = process_gender_dataset_from_folders(base_folder_path)

print(f"Processed {len(images)} images")
print(f"Unique labels: {label_encoder.classes_}")
