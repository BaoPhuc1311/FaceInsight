import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN
from keras_facenet import FaceNet
from preprocessing import preprocess_image

detector = MTCNN()
embedder = FaceNet()

def extract_face_features(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        faces = detector.detect_faces(image)
        if len(faces) > 0:
            x1, y1, width, height = faces[0]['box']
            x1, y1 = max(0, x1), max(0, y1)
            face = image[y1:y1+height, x1:x1+width]
            embeddings = embedder.embeddings([face])
            if embeddings is not None and len(embeddings) > 0:
                return embeddings[0]
    return None

def train_gender_model_using_facenet(base_folder_path):
    images = []
    labels = []
    label_encoder = LabelEncoder()

    for folder in ["train", "test"]:
        folder_path = os.path.join(base_folder_path, folder)
        for gender in ["women", "men"]:
            gender_folder_path = os.path.join(folder_path, gender)
            for file_name in os.listdir(gender_folder_path):
                if file_name.endswith(".jpg"):
                    image_path = os.path.join(gender_folder_path, file_name)
                    features = extract_face_features(image_path)
                    if features is not None:
                        images.append(features)
                        labels.append(gender)

    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(np.array(images), labels_encoded, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return svm_model, label_encoder

base_folder_path = "assets/images"
svm_model, label_encoder = train_gender_model_using_facenet(base_folder_path)

joblib.dump(svm_model, 'gender_classification_svm_model.pkl')
