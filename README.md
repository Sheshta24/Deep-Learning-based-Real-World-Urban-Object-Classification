# Deep-Learning-based-Real-World-Urban-Object-Classification


# Project Overview

This project involves developing and evaluating deep learning models for object classification in urban driving scenarios. Using a dataset collected by the RAMI group, which includes real-world images from urban driving scenes in London, the project aims to classify objects such as cars, pedestrians, and bicycles. The goal is to train and test models to achieve high recognition accuracy, providing insights into the performance of both a custom Convolutional Neural Network (CNN) and a transfer learning model.

# Objectives

Data Processing:
Extract and label sub-objects (cars, pedestrians, and bicycles) from original images.
Resize images to a uniform size for training and store them in a format suitable for deep learning.
Split data into training, validation, and test sets.
Model Development:
Build a custom CNN model from scratch.
Build a transfer learning model using an existing pre-trained model.
Experiments and Training:
Train both models with validation processes.
Evaluate and compare model performances in terms of validation and testing accuracy.
Analysis and Visualization:
Plot the training process, including loss and accuracy over epochs.
Generate Grad-CAM-based saliency maps to visualize which parts of the images the models focus on during classification.
Perform hyperparameter tuning, data augmentation, and explore various network architectures to enhance model accuracy.

# Dataset

The dataset contains 821 images of urban scenes in London, annotated in Pascal VOC format for object detection. The dataset is provided by the RAMI group and includes monocular camera information.

# Requirements

Python 3.x
TensorFlow or PyTorch for building and training deep learning models
OpenCV for image processing
Pandas, NumPy, Matplotlib for data handling and visualization

# Steps to Run the Project

Clone this repository.
Install required libraries using pip install -r requirements.txt.
Follow the notebook or scripts provided to preprocess the dataset, train the models, and evaluate performance.
