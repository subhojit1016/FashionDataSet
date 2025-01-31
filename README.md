# Summary of the Project: Image Classification using Fashion-MNIST Dataset

This project involves building and training a deep learning model using PyTorch to classify images from the Fashion-MNIST dataset, achieving high accuracy through Convolutional Neural Networks (CNNs), data augmentation, and optimization techniques.

# Data Preparation & Preprocessing
1. Used the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images.
2. Each image is 28x28 grayscale, representing 10 different fashion categories (e.g., T-shirt, Sneaker, Coat, etc.).
3. Applied data augmentation using:
  a. Gaussian Blur
  b. Normalization [-1, 1]
  c. Conversion to Tensor
4. Data split:
  a. 80% Training (48,000 samples)
  b. 20% Validation (12,000 samples)
  c. A custom function split_indices() was used to create train-validation splits.
# Model Architecture
1. A CNN-based model (ImageClassifierNet) was built with:

2. Two convolutional layers:
  a. Conv2D(1 → 4 channels), followed by ReLU activation and MaxPooling
  b. Conv2D(4 → 16 channels), followed by ReLU activation and MaxPooling
  c. Batch Normalization and Dropout (10%) to prevent overfitting.
3. Fully Connected (FC) Layers:
  a. Linear(7x7x16 → 100)
  b. Linear(100 → 10) (output layer)
  c. ReLU activations were used in hidden layers.
# Model Training
1. Used Stochastic Gradient Descent (SGD) with momentum=0.9 as the optimizer.
2. Cross-Entropy Loss function was used to compute classification error.
3. Training was performed for 200 epochs, with:
  a. Gradual decrease in training loss from 1.24 to 0.14.
  b. Validation accuracy improvement from 62% to ~90%.
# Model Evaluation
1. The model was evaluated on a separate test set.
2. Achieved Test Accuracy = 91.82%.
3. Loss and accuracy trends were plotted to ensure smooth training.
