## Introduction

In 2019, we experienced a pandemic that no one could predict or was equipped to handle. COVID-19 had symptoms like harmless flu but quickly spread around the globe and changed our lives forever. By the time politicians and researchers were busy figuring out a permanent solution to save as many lives as possible, a precautionary measure was suggested and face masks became the new norm to safeguard ourselves. Unfortunately, due to misinformation and several other reasons getting people to follow face mask restrictions was more difficult than expected. People didn’t respect this measure and continued to get infected as well as spread infection.

Face detection is one of the most commonly discussed topics when it comes to the power of deep learning. Most electronic devices such as phones, billing systems in some countries, etc. already use the person’s face for identification purposes such as smartphone’s face unlock (Apple) and password protection feature.
At the moment we are writing this document, we’re not anyone in this pandemic situation, but we want our world to be more prepared for the next one. The goal of this project is to help reduce future virus spread by using deep Learning neural networks as a solution to identify people who don’t wear or don’t properly wear their face mask.

## Problem Description:

Indiscipline and Non-compliance with COVID-19-related public health measures were one of the causes of the duration of this pandemic. While the doctors and research specialists were doing their best to find a solution for COVID-19, people were spreading the disease by not wearing their masks and not respecting measures taken by the governments.
We wanted to help reduce the spread of future illness by building a model which can detect the face mask on an individual’s face and determine if the person is wearing it correctly/incorrectly or not wearing it at all.

## Description of Data:

During requirement gathering, we identified two potential datasets that could help us build the model. The first Dataset was gathered from kaggle and only two classes (WithMask and WIthoutMask). The other dataset identified was based on a dataset called MaskedFace-Net which recorded human faces with correctly and incorrectly worn masks. This dataset contained three classes representing images of individuals wearing masks, without masks and masks worn incorrectly. Additionally, the dataset met our requirements of having images taken from different angles as well as diversity.

The train dataset has 4000 images with masks and 4100 images without masks, and 3500 with incorrect orientation of mask, whereas the test dataset contains 1300 images with masks and 1400 without masks and 1100 worn incorrectly. Here is the link to the Datasets: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset, https://github.com/cabani/MaskedFace-Net

## Data Transformation:

Having loaded the necessary libraries and fetched the data into the working environment, we augmented the images by rotating, zooming, and flipping the images at various angles.

## Modeling

Model 1 - Basic CNN: To commence the project in its simplest form, we started with a simple basic CNN model with the following hyperparameters and achieved a good accuracy percentage.
Activation = “tanh”,”dropout”,”Batch Normalization”, epoch: 15, batch size = 32
Optimizer: adam, Loss: categorical_crossentropy
Accuracy Achieved: 97.84%

Though the accuracy we achieved was close to 98%, the model was not detecting the facemask very efficiently. For example, if the mask is of different color or a hand on the face, the model would have difficulty in distinguishing it and predicting it incorrectly. So to move forward, we planned to try an advanced model that is more practical and not just relying on accuracy.

Model 2 - MobilNet: As the basic CNN had few challenges in detecting the facemask, we planned to try MobilNet CNN model below hyperparameters:
Activation = “relu”, Optimizer = “SGD”, loss = “binary_crossentropy”, epochs = 20 Accuracy Yield: 87.90%

Despite decent accuracy the model was not performing well with several scenarios and there was lag in identifying the mask on a person's face in real time. Therefore, we decided to move to another model.