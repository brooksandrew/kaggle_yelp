### Overview

This project explores the DeepLearning4J implementation within Scala to classify images for the [Yelp Photo Classification problem on Kaggle].
It is a [multi-label classification problem] where each entity can belong to multiple classes.  The goal of this project was to experiment 
with a data science problem in Scala utilizing a deep learning library.  It could be possible to train meaningful CNNs with this approach.
However, the CNNs produced in the results folder of this repo are by no means novel, as I currently
don't have the patience or resources (GPU, EC2) to train these models at scale.  My intentions are pedantic, so this is more of an experiment/tutorial than
a shot at the non-monetary Kaggle grand prize.

### What the project does

1. Reads images from .jpg into a matrix representation in Scala  
2. Process images for convolutional neural network  
  a. square image  
  b. resize every image to the same dimensions  
  c. apply grayscale filter to image  
3. Train 9 convolutional neural networks (CNNs) on training data for each class.
4. Saves CNN config and parameters
5. Applies a simple aggregate (average) function to assign classes to each business.  Each business has multiple images associated with it, each with its own vector of probabilities for each of the 9 classes)
6. Scores test data
7. Compiles predictions into submission CSV file for Kaggle

### How to run 

`/src/main/scala/modeling/main.scala` is the code to run the project end-to-end.  Training CNNs is very time consuming, so it is
likely that you will not often run the project in its entireity very often.

#### Using Sbt (tested on Mac Terminal)
1. Clone project
2. > Sbt 
3. > run

### Getting into the code





[Multi-Label classification problem]: https://en.wikipedia.org/wiki/Multi-label_classification
[Yelp Photo Classification problem on Kaggle]: https://www.kaggle.com/c/yelp-restaurant-photo-classification
