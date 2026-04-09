# MNIST CNN Image Classifier

This project builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## 🚀 Features
- CNN-based digit classification
- Achieves ~98–99% accuracy
- Supports custom image prediction
- Clean and modular code

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Run
```bash
python main.py
```

## 🖼️ Test Custom Image
Place your image inside `sample_images/` and update filename in `main.py`.

## 📊 Results
- Training Accuracy: ~99%
- Test Accuracy: ~98–99%

## 📁 Structure
```
mnist-cnn-classifier/
│── main.py
│── requirements.txt
│── README.md
│── sample_images/
```

## 📌 Notes
- Images must be 28x28 grayscale
- White background preferred (auto-inverted if needed)
