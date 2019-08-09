# SVM---Handwritten-digit-recognition

Business Understanding

A classic problem in the field of pattern recognition is that of handwritten digit recognition.
Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

 
Objective
You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits based on the pixel values given as features.

 

Results Expected

    1. Write all your code in one well-commented R file; briefly, mention the insights and observations from the analysis.

Downloads:
For this problem, we use the MNIST data which is a large database of handwritten digits where we have pixel values of each digit along with its label. 

 

Important Note
It would take a lot of time for training the model on the full MNIST data (~ 60k observations), so you are advised to use not more than 5, 000 training observations. Please make sure you subset the data such that the class balance is maintained, i.e. do stratified sampling. 
