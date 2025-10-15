🧠 Brain Tumor Detection using CNN

This project builds a Convolutional Neural Network (CNN) model using TensorFlow and Keras to automatically detect brain tumors from MRI images.

The model is trained on a Brain MRI dataset sourced from Kaggle, which can be found here
.

📁 About the Dataset

The dataset includes two folders:

yes/ → 155 MRI images containing brain tumors

no/ → 98 MRI images without tumors

In total, there are 253 MRI scans used for model training and validation.

🚀 Getting Started

⚠️ Sometimes GitHub doesn’t render Jupyter notebooks properly.
You can view them easily using nbviewer
.

🧩 Data Augmentation
Why Augmentation Was Used

Since the dataset is quite small and imbalanced, data augmentation was performed to generate more diverse examples. This helps the CNN generalize better and prevents overfitting.

Dataset Summary
Stage	Tumorous Images	Non-Tumorous Images	Total
Before Augmentation	155	98	253
After Augmentation	1085	980	2065

All augmented samples are stored inside the ‘augmented data’ directory, including the original 253 images.

🧼 Data Preprocessing

Each MRI image underwent the following preprocessing steps:

Brain Region Cropping – Extract only the brain portion from the image to eliminate background noise.

Resizing – Resize all images to (240, 240, 3) to maintain consistent input dimensions.

Normalization – Scale pixel values between 0 and 1 for better convergence during training.

🔀 Data Splitting

The processed dataset was divided into:

70% → Training set

15% → Validation set

15% → Test set

🧠 Model Architecture

The model follows a simple CNN architecture optimized for small datasets and limited compute power.

Layers Overview

ZeroPadding2D with pool size (2, 2)

Conv2D layer (32 filters, kernel size (7,7), stride 1)

BatchNormalization

ReLU Activation

MaxPooling2D with pool size (4,4)

MaxPooling2D (again)

Flatten Layer

Dense Layer with 1 output neuron (Sigmoid activation for binary classification)

⚙️ Why This Architecture?

Initially, pre-trained models like VGG16 and ResNet50 were tested, but due to limited data and hardware (Intel i7 6th Gen, 8GB RAM), they led to overfitting and long training times.
Hence, a lightweight CNN was built and trained from scratch — achieving competitive results.

📊 Model Training

The model was trained for 24 epochs with binary cross-entropy loss and Adam optimizer.

Training Visualizations:




The best validation accuracy was achieved on epoch 23.

🧾 Results
Metric	Validation Set	Test Set
Accuracy	91%	89%
F1 Score	0.91	0.88

✅ Final Test Accuracy: 88.7%
✅ Final F1 Score: 0.88

These metrics demonstrate that the model performs well even with limited and balanced data.

💾 Model Files

All model weights and checkpoints are stored in the models/ folder.

The best model is saved as:

cnn-parameters-improvement-23-0.91.model


You can easily reload it using:

from tensorflow.keras.models import load_model
best_model = load_model('models/cnn-parameters-improvement-23-0.91.model')

📂 Repository Structure
├── data/
│   ├── yes/
│   ├── no/
│   └── augmented data/
├── models/
│   └── cnn-parameters-improvement-23-0.91.model
├── notebooks/
│   └── preprocessing_and_training.ipynb
├── Loss.PNG
├── Accuracy.PNG
└── README.md

🤝 Contributions

Feel free to open issues or submit pull requests for further improvements such as:

Using advanced architectures like EfficientNet or MobileNet

Hyperparameter tuning

Visualizing Grad-CAM heatmaps

✨ Acknowledgments

Special thanks to the original dataset contributors and TensorFlow community for tools and resources that made this project possible.