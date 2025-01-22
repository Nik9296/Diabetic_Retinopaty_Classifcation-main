# Diabetic Retinopathy Classification

This repository contains a project for classifying retinal images into five stages of diabetic retinopathy using a Convolutional Neural Network (CNN) model. The five stages include `No_DR`, `Mild`, `Moderate`, `Severe`, and `Proliferate_DR`. The model achieves a validation accuracy of 92%.

## Project Overview

Diabetic Retinopathy is a leading cause of blindness and early detection is crucial for effective treatment. This project focuses on developing a CNN-based approach to automatically classify retinal images into the appropriate stage of diabetic retinopathy, aiding in the diagnosis process.

### Key Features:
- **Multiclass Classification:** The model classifies images into one of the five stages of diabetic retinopathy.
- **High Accuracy:** Achieved accuracy of 92%.
- **Image Preprocessing:** Applied various preprocessing techniques to enhance image quality for better model performance.
- **Deep Learning:** Utilized TensorFlow and Keras for building, training, and evaluating the CNN model.
- **End-to-End Pipeline:** Implemented a complete pipeline from data preparation to model deployment using Python.

## Dataset

The dataset consists of labeled retinal images classified into five categories representing different stages of diabetic retinopathy:
- No_DR
- Mild
- Moderate
- Severe
- Proliferate_DR

## Model Architecture

The CNN model architecture includes:
- **Convolutional Layers:** To extract features from the input images.
- **Max Pooling Layers:** To downsample the feature maps.
- **Batch Normalization:** To stabilize and accelerate the training process.
- **Fully Connected Layers:** To perform the final classification.
- **Softmax Activation:** To output probabilities for each class.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AniketLokhande801/Diabetic_Retinopathy_classification.git
    cd Diabetic_Retinopathy_classification
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered.

## Usage

1. **Training the Model:**

    To train the model, run the following command:
    ```bash
    python train.py
    ```

2. **Testing the Model:**

    To test the model on a new image, use the `app.py` script:
    ```bash
    streamlit run app.py
    ```

## Results

- **Validation Accuracy:** 92%
- **Sample Predictions:** https://github.com/AniketLokhande801/Diabetic_Retinopaty_classifcation/blob/main/results.png

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements.

## Contact

For any inquiries, please contact Aniket Somnath Lokhande at aniketlokhande3654@gmail.com.
