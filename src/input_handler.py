from PIL import Image
import numpy as np
import io

def load_image_from_path(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)

def load_image_from_bytes(byte_data):
    image = Image.open(io.BytesIO(byte_data)).convert("RGB")
    return np.array(image)

def resize_image(image_np, target_size=(224, 224)):
    image = Image.fromarray(image_np)
    image = image.resize(target_size)
    return np.array(image)
