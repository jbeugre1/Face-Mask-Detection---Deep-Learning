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

![alt text](https://github.com/jbeugre1/Face-Mask-Detection---Deep-Learning/blob/main/img/2.png)
Figure 1: Building block for MobileNet


Model 3 - MobilNet V2

While testing MobileNet version 1 using a webcam we observed performance issues in generating accurate and fast predictions. Despite lower accuracy Mobilenet was working considerably better than our Native CNN model we wanted to experiment with MobileNet version 2. MobileNet v2 works slightly differently as it keeps the number of incoming channels to approximately the same amount instead of doubling it as V1 while also adding an expansion layer to boost feature learning and a residual connection similar to Resnet for the flow of gradients. The V2 version had 17 such blocks instead of the 13 in V1 but ensured the relatively smaller size of tensors for faster and accurate predictions.

Model 3 - MobilNetV2: The model was trained on the following hyperparameters:
Activation = “relu”, Optimizer = “Adam”, loss = “binary_crossentropy”, epochs = 20, learning rate =
1e-4, batch size = 32 Accuracy Yield: 98.90%

![alt text](https://github.com/jbeugre1/Face-Mask-Detection---Deep-Learning/blob/main/img/3.png)
Figure 2: Building block for MobileNetV2

## Methodology

Given the aim of the project was a fast and accurate model we wanted to build the model to learn about scenarios it might experience during a live test. To make it possible, we had to find a dataset large enough to improve the model performance up to a certain point. As diversity was a key objective, we had to make sure that everybody regardless of their gender, skin tone and position can be identified by our models.

In order to enrich the image dataset, we decided to apply a data augmentation to every image to ensure our model can learn variations in the input image through data transformations. To do this, we used the function ImageDataGenerator from the Keras to apply the data augmentation on our image. This function helps us create new images by flipping our datasets horizontally and changing their size, either in one direction or a combination.

With this augmented dataset, we started loading our different dataset to start training our models, and since our data came inside folders with different type of picture, we had to use another Keras’ function called image_dataset_from_directory to load those datasets inside our code. We then tested different convolutional neural networks and selected only those with test accuracy above 90% for our real-time detection.

![alt text](https://github.com/jbeugre1/Face-Mask-Detection---Deep-Learning/blob/main/img/4.png)
Figure 3: Real-time detection methodology

We used OpenCV to capture Video from a Camera and since video can be represented as a series of images, our model will predict a label in each frame of the video captured by the webcam. The first step in our real-time classification model was to find a model capable of detecting a single face. We tested two models. The Haar cascade and the Caffe model.

The Haar cascade model relies on edge and line detection and returns the position and size of a face. Unfortunately, this method has too many flaws such as the fact that it could only detect the front face and it wasn’t accurate and was very slow when it was trying to detect a moving target.
 
So, we had to go with a deep neural network module and Caffe models and those two models put together could solve all the problems of the Haar cascade model. This model could detect faces even when an object hides a part of those faces.
The remaining part was loading our convolutional neural network models, crop the face detected by the deep neural network module and Caffe models and predict the label of the pictures with our models.

## Result
After testing different models, we choose to compare our basic CNN model and the MobilNet V2 on the real time face detection, because both of them have a very high accuracy close to 99%.

We know the models wouldn’t be perfect, but we were surprised by the different results that we got for those models. Despite having a high accuracy on the test dataset, the basic CNN model performance in the real time face classification is not stable at all. The production highly depends on the positioning of the face and most of the time return the wrong label. On the contrary, the MobilNet V2 had a very good performance and accuracy.

The table below will provide feedback on the various tests we have conducted.

![alt text](https://github.com/jbeugre1/Face-Mask-Detection---Deep-Learning/blob/main/img/1.png)


## Conclusion

In conclusion, we can say that creating a real-time face mask classification was a delicate process and requires a lot of preparation. The selection of the model was very important because the high-test accuracy of a model didn’t guarantee a good performance when we associate it with the face detection application.
From the result obtained, we realized that we need more complex neural network architecture to solve some problems that affect the performance of our models such as the lighting condition, the diversity of masks and users. Additionally, augmenting the data by applying additional transformations to replicate the scenarios during the training process can also be helpful.
We think that with an enhancement of this model, this project could be implemented to truly solve real life problems in case another pandemic appears.
References:
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset https://github.com/cabani/MaskedFace-Net https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08