# Gender Classification using VGG

This project implements Gender Classification using the VGG Neural Network Architecture. 
The model is trained on the CelebA dataset, which contains images labeled with gender information, to predict the gender of a person from an input image.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction

Gender classification is a common task in computer vision, with applications ranging from demographic analysis to targeted advertising. This project utilizes the VGG architecture, specifically VGG-16 and VGG-19, which have shown strong performance in image classification tasks. The CelebA dataset, a widely used benchmark for face-related tasks, is employed for training and evaluation.

## Features

- Gender Classification from input images using VGG-16 and VGG-19
- Hyperparameter tuning using Hyperband, Random Search, and Bayesian Optimization
- Customizable VGG-based Neural Network model
- Flexible for further training or fine-tuning on custom datasets

## Requirements

- Python 3.9
- Tensorflow and Keras
- Keras Tuner
- Scikit-Learn
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Usage

- Download the CelebA dataset and preprocess it according to your requirements.
- Choose between VGG-16 and VGG-19 for your gender classification task.
- Train the model using hyperparameter tuning techniques such as Hyperband, Random Search, or Bayesian Optimization.
- Evaluate the model performance and fine-tune as necessary.
- Use the trained model for gender classification on new images.