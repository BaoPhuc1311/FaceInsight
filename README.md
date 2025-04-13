# 👁️ Face Insight

## 🧠 Introduction  
**Face Insight** is a computer vision project that focuses on facial recognition and analysis. It aims to detect human faces from images, extract facial features, and provide analytical insights such as age, gender, and emotion estimation. The project serves as a foundational tool for building advanced human-centric applications such as security systems, smart attendance, personalized AI, and behavioral analytics.

## 🎯 Objectives  
- Load and preprocess images from various input sources  
- Detect and align faces accurately using reliable detection models  
- Extract meaningful facial embeddings for recognition  
- Analyze facial features to estimate attributes such as age, gender, and emotion  
- Design a modular pipeline, ready for integration into a Streamlit-based UI  

## 📦 Requirements  
- Python 3.x  
- OpenCV, Numpy, PIL, MTCNN, FaceNet, DeepFace
- Streamlit

## 📊 Data  
This project uses the following dataset:  
- [Gender Detection and Classification Image Dataset – Kaggle](https://www.kaggle.com/datasets/trainingdatapro/gender-detection-and-classification-image-dataset)

## 🛠️ Installation & Usage  

1. **Clone the repository and set up a virtual environment**:
    ```bash
    git clone https://github.com/BaoPhuc1311/FaceInsight.git
    cd FaceInsight
    python -m venv venv
    .\venv\Scripts\activate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🤝 Contributing  
Feel free to fork the repository, submit issues, or open pull requests.  
Contributions related to new datasets, analysis methods, or visualizations are highly welcome!

## 📄 License  
This project is licensed under the [MIT License](LICENSE).
