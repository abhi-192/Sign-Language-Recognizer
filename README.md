# Sign-Language-Recognizer

This project is all about creating a Sign
Language Recognizer which can help hearing-impaired individuals in
better communication of their ideas.

# Introduction

Our project aims to provide a solution to hearing-impaired community for
easy communication of their ideas and thoughts. The main problem with
sign language is that it is not so popular and very few people know it,
and so there is an undeniable lack of communication between
hearing-impaired people and people who don’t need any hearing aid. This
thesis lists the detailed implementation of a sign language recognizer
which can be used for easy communication. When viewed through a
systematic standpoint , we can identify that there are four major steps
in implementing such a sign language recognizer. First is finding a
dataset which can be used for proper training . Next comes selecting a
proper model for training . Then follows the testing phase, and
validating what the model predicts is correct or not. And finally the
last steps constitutes the analysis of results what our model predicts
and realization of whether our model is upto our standards or not, and
what specific test cases or patterns our model fails upon, if it fails.


# Proposed Work

## Data Collection

Proper selection of data is the most important in training any model, a
wrong dataset or impurities present in dataset can lead to wrong
predictions. The main problem with our problem is lack of a proper
dataset which can be used. While many datasets do exist for this
problem, but each one of them lacks some features, for example, the ASL
Alphabet dataset available on Kaggle, is sufficiently large, consisting
of more than 87000 images distributed over 29 classes, with 3000 images
for each, is still not upto standards due to its lack of clarity in
images. Most of the images are confusing and captured in dim light which
seems to create problem in distinguishing even for humans.

## Data Preprocessing

The important factor that we must not ignore is that black and white
images with clear boundaries are much better than colored images for
training our model. That’s why we have firstly converted the colored RGB
image into GRAY image. Upon further exploration and experimentation we
found that gaussian blur in combination with adaptive thresholding is
perfect for edge enhancement, this makes border distinct and sharp. This
effect will be of great impact when we train our model. We have used
OpenCV library for all image processing tasks.

## Model used

Our choice of model is inspired by [Pigou](https://link.springer.com/chapter/10.1007/978-3-319-16178-5_40), where the author uses a CNN
and achieves an accuracy of 91.7 % using a 2D kernel. However, an
accuracy of over 95% has been achieved by same techniques with a 3D
kernel. [Simming He](https://ieeexplore.ieee.org/document/8950864/) proposed a system using R-CNN, but the accuracy
achieved was only 89%. Another researcher, [Rekha, J](https://ieeexplore.ieee.org/document/6169079) used YCbCr skin
model in combination with multi class SVM, and achieved an accuracy of
86.4%. By far in terms of its simplicity, and accuracy achieved CNN
seems to be the best choice, so far.

### Neural Network

Neural Network is a series of connected nodes called as neurons which
are inspired from neurons present in human brain , these nodes work in
collaboration to establish and predict complex relationship between
data. These nodes do series processing and try to replicate the neural
network of human brain. Each of these node gets some input, performs
some mathematical function and forwards it to next neuron, and this goes
on in well defined layers, so that complex relationships can easily be
predicted using model.

### Convolutional Neural Network

CNN(Convolutional Neural Network) is a type of neural network which is
specifically used for images. CNN consists of multilayer perceptrons,
that is, each neuron in one layer is connected to all neurons in the
next layer. The "full connectivity" of these networks make them prone to
overfitting data. The central idea in image recognition is that we
combine a neighbouring group of pixel values to get an idea of what
feature it contains, then this value is passed to next layer and then
next layer does the same.

#### Convolution Layer

Much like its name, it does convolution , a convolution is a linear
operation that involves the multiplication of a set of weights with the
input.

#### Pooling Layer

Pooling Layer is added after each convolution layer for the purpose of
reducing number of channels. This way we can have a summary of features
without making the number of variables too large.

#### Dense Layer

This layer is part of last few layers and put simply, is a feed forward
neural network, and is also known as Hidden layer. This layer multiply
weights , add bias and apply activation function and filter number of
variables.

#### Dropout Layer

This layer is also implemented although it is not compulsory, its main
purpose is to drop a number of data points so that the model avoids
overfitting.

#### Flatten Layer

This layer is responsible for flattening the multidimensional data that
we have finally arrived at.

#### Output Layer

This layer is a final layer in CNN and is responsible for transforming
the output into a number of classes.  
  
Our proposed architecture uses three convolution layer, with kernel size
of (3,3) each followed by a pooling layer with kernel size of (2,2). The
activation function used is ReLU. The result through these layers is
passed to Flattening layer, which then passes it to a series of four
fully connected layer, the first two with 128 units and then the next
one with 64 units and the last one with 27 units. We also propose one
dropout layer each for first two dense layer with a dropout probability
of 0.30. We will use Adam Optimizer and the "categorical\_crossentropy"
loss function that is used in the classification task. This loss is a
very good measure of how distinguishable two discrete probability
distributions are from each other. We have used Keras and Tensorflow to
implement our CNN.

## Data Augmentation

Data augmentation is a technique to generate more training data using
previous well defined data, by using various transformations on original
data. In our dataset there are limited augmentation techniques which we
can use. We cannot rotate, flip or crop the images. However we use
techniques such as translation and rescaling for generating more data.

## Testing and Validation

We use 75% of our dataset for training purpose and remaining 25% for
testing and validation.

![image](img2%20-%20model%20metrics.png)

# Experimental Setup and Results Analysis

We have used following hardware setup for training and testing
purposes.  
  
Processor: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz 1.99 GHz  
Installed RAM: 8.00 GB (7.89 GB usable)  
System type: 64-bit operating system, x64-based processor  
Operating System: Windows 11 Pro Version 21H2  
  
The result obtained show an accuracy of 70.74% on training data with a
loss of 0.9679 and an accuracy of 98.62% on testing and validation data
with a loss of 0.0585.

# Conclusion and Future Work

This seems to be a fair result considering the fact that an overall
accuracy of over 95% has already been achieved by various researchers in
sign language recognizer.  
  
For future work, we could try to implement an object detection model for
recognizing our hand and then use our model to predict the correct
symbol. This way there will be no limitation of fixed ROI. Our sign
language recognizer performs a simple task of recognizing correct
gestures, we may implement a next word suggestor, or even integrate a
word to speech converter with it for future ideas.
