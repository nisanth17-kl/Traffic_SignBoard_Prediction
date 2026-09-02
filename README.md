# 🚦 Traffic Sign Board Detection with Data Augmentation

A deep learning-based traffic sign recognition system that identifies traffic signs from images using a Convolutional Neural Network (CNN).

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and applies image data augmentation techniques to improve the model's ability to recognize different traffic signs.

## 📌 Features

- 🚦 Detects and classifies traffic signs
- 🧠 CNN-based deep learning model
- 🔄 Image data augmentation
- 📊 Model accuracy and performance evaluation
- 📷 Upload an image for prediction
- 🎥 Capture an image using a camera
- 📈 Displays prediction confidence
- ⚠️ Displays risk level
- 🛡️ Provides suggested action for the detected sign
- 🌐 Flask-based web application
- 🔝 Shows Top-3 predictions

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask
- Flask-CORS
- HTML
- CSS
- JavaScript

## 📂 Project Structure
```text
Traffic_SignBoard_Prediction/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── results/
    ├── Accuracy_Validation.png
    └── Confusion_Matrix.png


## 📊 Results

### Training Accuracy and Validation Accuracy

![Accuracy Graph](results/Accuracy_Validation.png)

### Confusion Matrix

![Confusion Matrix](results/Confusion_Matrix.png)
