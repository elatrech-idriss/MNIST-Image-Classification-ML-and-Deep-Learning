# MNIST Image Classification with Machine Learning and Deep Learning

## Project Overview
This project explores image classification on the **MNIST dataset** using both
traditional machine learning techniques and deep learning models.

The objective is to understand the differences in performance, training behavior,
and representation power between linear models, fully connected neural networks,
and convolutional neural networks (CNNs).

---

## Dataset
- **MNIST** handwritten digits dataset
- 60,000 training images
- 10,000 test images
- Image size: 28×28 (grayscale)
- 10 classes (digits 0–9)

---

## Implemented Approaches

### 1. Machine Learning Baseline
- Data flattening and normalization
- **SGDClassifier** with logistic loss
- Evaluation of performance variability using different random seeds

### 2. Fully Connected Neural Network
- Multi-layer perceptron (MLP)
- ReLU activations
- Softmax output layer
- Trained using **Adam optimizer**

### 3. Convolutional Neural Network (CNN)
- Convolution + MaxPooling layers
- Dense classification head
- Improved spatial feature extraction
- Achieves higher accuracy than fully connected models

---

## Key Concepts Covered
- Data preprocessing and normalization
- Train / validation / test splits
- Overfitting vs underfitting
- Model comparison
- Deep learning architectures
- CNN feature extraction

---

## Technologies Used
- Python
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## Results
The CNN model significantly outperforms classical machine learning approaches,
highlighting the importance of spatial structure in image classification tasks.

---

## Author
**Idriss Elatrech**  
Engineering student – Data Science & Machine Learning  
