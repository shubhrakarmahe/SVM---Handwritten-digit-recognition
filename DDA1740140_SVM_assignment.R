
# ---------------------------- SVM Assignment - Digit Recognizer --------------------

# Problem Statement -

# Develop a model using Support Vector Machine which should correctly classify the
# handwritten digits based on the pixel values given as features.

# -----------------------------------------------------------------------------------

# Load Neccessary libraries

# install.packages("caret",dependencies = TRUE)
# install.packages("e1071",dependencies = TRUE)
# install.packages("kernlab",dependencies = TRUE)
# install.packages("ggplot2",dependencies = TRUE)
# install.packages("gridExtra",dependencies = TRUE)
# install.packages("splitstackshape",dependencies = TRUE)
# install.packages("data.table",dependencies = TRUE)

library(kernlab)
library(caret)
library(dplyr)
library(data.table)
library(ggplot2)
library(splitstackshape)
library(gridExtra)
library(e1071)

# Import Training & Test Data

train <- fread("mnist_train.csv",
               header = FALSE)
test <- fread("mnist_test.csv",
              header = FALSE)

# -------------------------------Preliminary Analysis -------------------------------

# 60000 rows and 785 columns
dim(train)
# 10000 rows and 785 columns
dim(test)

# Print few rows 
head(train)
tail(train)

head(test)
tail(test)

# Structure & Summary
# Observation - All columns are numeric.
str(train)
str(test)

summary(train)
summary(test)

# Duplicate Rows - none
uniqueN(train)
uniqueN(test)

# Missing values - none
sum(!complete.cases(train))
sum(!complete.cases(test))

# ---------------------------Data Understanding ------------------------------------
# The MNIST database of handwritten digits has a training set of 60,000 examples, 
# and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
# The 784 columns apart from the label consist of  28*28 matrix describing the scanned 
# image of the digits
# The digits have been size-normalized and centered in a fixed-size image.

# ---------------------------Data Preparation -------------------------------------

# There are no headers,Let's rename column 1 as label.
names(test)[1] <- "label"
names(train)[1] <- "label"

# Convert label column to factor
train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

table(train$label)

table(test$label)

# To train SVM for large training dataset, computation time will be high.
# Let's do a strapified sampling of apprx 5000 records out of 60000.
set.seed(1)
train_sample <- stratified(train, "label", 0.0825)

# 4982 rows & 785 columns
dim(train_sample)
head(train_sample)
  
table(train_sample$label)

# --------------------------Visual Analysis - Strapified Sampling ---------------------
plot_train <- ggplot(train, aes(x = label, y = (..count..)/sum(..count..))) + 
  geom_bar(fill = 'dodgerblue') +
  labs( y = "Frequency in %", title = "Train_60K") +
  scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot_test <- ggplot(test, aes(x = label, y = (..count..)/sum(..count..))) + 
  geom_bar(fill = 'dodgerblue') +
  labs( y = "Frequency in %", title = "Test_10K") +
  scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot_train_sample <- ggplot(train_sample, aes(x = label, y = (..count..)/sum(..count..))) + 
  geom_bar(fill = 'dodgerblue') +
  labs( y = "Frequency in %", title = "Train_Sample_5K") +
  scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

grid.arrange(plot_train, plot_test, plot_train_sample, nrow = 3)

# It is quite evident that our sample dataset have proper distribution of all classes.

# Scale all numeric columns 
# max value is 255,it will be used to scale all numeric values.
max(train_sample[ ,2:ncol(train_sample)]) 

train_sample[ , 2:ncol(train_sample)] <- train_sample[ , 2:ncol(train_sample)]/255

test[ , 2:ncol(test)] <- test[ , 2:ncol(test)]/255

# ------------------------ SVM Model Building -----------------------------------

# ----------------------Linear Kernel --------------------------------------
# Linear kernel using default parameters

model_linear_1 <- ksvm(label ~ ., 
                       data = train_sample, 
                       scaled = FALSE, 
                       kernel = "vanilladot", 
                       C = 1)

print(model_linear_1) 

eval_linear_1 <- predict(object = model_linear_1, 
                         newdata = test)

confusionMatrix(data = eval_linear_1, 
                reference = test$label) 

# Overall accuracy is 91.11%
# Overall Senstivity >= 83%
# Overall Specificity >= 98%
# --------------------------------------------------------------------------

#  5 cross validation to optimise C value

grid_linear <- expand.grid(C= c(0.001, 0.1 ,1 ,10 ,100)) 

fit.linear <- train(label ~ ., 
                    data = train_sample, 
                    metric = "Accuracy", 
                    method = "svmLinear",
                    scale = FALSE,
                    tuneGrid = grid_linear, 
                    trControl = trainControl(method = "cv", number = 5))

# print results of 5 cross validation
print(fit.linear) 
plot(fit.linear)

# Best accuracy of 91.9%  at C = 0.1

eval_linear_cv <- predict(object = fit.linear, 
                          newdata = test)

confusionMatrix(data = eval_linear_cv, 
                reference = test$label)

# Overall accuracy is 92.36%
# Overall Senstivity >= 87%
# Overall Specificity >= 98%

# Let's see the outcome of Polynomial Kernel.

#--------------------------------------------- Polynomial Kernel ----------------------------------------------#

# Polynomial kernel with degree 2 & default (scale & offset)
model_poly_1 <- ksvm(label ~ ., 
                     data = train_sample, 
                     kernel = "polydot", 
                     scaled = FALSE, 
                     C = 1,
                     kpar = list(degree = 2, scale = 1, offset = 1))
print(model_poly_1)

eval_poly_1 <- predict(object = model_poly_1, 
                       newdata = test)

confusionMatrix(data = eval_poly_1, 
                reference = test$label)

# Overall accuracy is 95%
# Overall Senstivity >= 92%
# Overall Specificity >= 99%
# Evaluation Metric values are far better than Linear Kernel. It means our dataset is non-linear.

# ---------------------------------------------------------------------------

# To optimise hyperparameters we will use Grid Search with 2 cross validation

grid_poly = expand.grid(C= c(0.01, 0.1, 1, 10), 
                        degree = c(1, 2, 3, 4, 5), 
                        scale = c(-100, -10, -1, 1, 10, 100))

fit.poly <- train(label ~ .,
                  data = train_sample, 
                  metric = "Accuracy", 
                  method = "svmPoly",
                  scale = FALSE,
                  tuneGrid = grid_poly,
                  trControl = trainControl(method = "cv", number = 2))

# printing results of cross validation
print(fit.poly) 
plot(fit.poly)

# Highest accuracy obtained for degree = 2, scale = -100 and C = 0.01.
eval_poly_cv <- predict(object = fit.poly, 
                        newdata = test)

confusionMatrix(data = eval_poly_cv, 
                reference = test$label)

# Overall Accuracy  95.41
# Overall Sensitivity >= 92%
# Overall Specificity >= 99%

# Let's check SVM for Radial Kernel
#------------------------------ Radial Kernel -------------------------------------------

# Radial kernel using default parameters

model_rbf_1 <- ksvm(label ~ ., 
                    data = train_sample, 
                    scaled = FALSE,
                    kernel = "rbfdot",
                    C = 1, 
                    kpar = "automatic")
print(model_rbf_1) 

eval_rbf_1 <- predict(object = model_rbf_1, 
                      newdata = test)

confusionMatrix(data = eval_rbf_1,
                reference = test$label) 

# Overall Accuracy 94.8 %
# Overall Senstivity >= 92%
# Overall Specificity >= 99%

# --------------------------------------------------------------

# TO optimise C and sigma,we will use  2-cross validation

grid_rbf = expand.grid(C= c(0.01, 0.1, 1, 5, 10), 
                       sigma = c(0.001, 0.01, 0.1, 1, 5)) 

fit.rbf_1 <- train(label ~ ., 
                 data = train_sample,
                 metric = "Accuracy", 
                 method = "svmRadial",
                 scale = FALSE,
                 tuneGrid = grid_rbf,
                 trControl = trainControl(method = "cv", number = 2))

print(fit.rbf_1) 
plot(fit.rbf_1)

# Best sigma value is 0.01 & C = 10

# Let's optimise C further with 5 cross validation holding sigma as 0.01
grid_rbf_2 <- expand.grid(C= c(0.01,0.1,1, 2, 3, 4, 5, 6 ,7, 8, 9, 10,15,20), 
                        sigma = 0.01)

fit.rbf_2 <- train(label ~ ., 
                  data = train_sample, 
                  metric = "Accuracy", 
                  method = "svmRadial",
                  scale = FALSE,
                  tuneGrid = grid_rbf_2,
                  trControl = trainControl(method = "cv", number = 5))

# print results of cross validation
print(fit.rbf_2) 
plot(fit.rbf_2)

# Accuracy is highest at C = 10 and sigma = 0.01

eval_rbf_cv <- predict(object = fit.rbf_2, 
                       newdata = test)

confusionMatrix(data = eval_rbf_cv, 
                reference = test$label)

# Overall Accuracy  95.66%
# Overall Sensitivites >= 93%
# Overall Specificities > 99%

# -----------------------------------------Conclusion -------------------------------------
# Final model
final_model = fit.rbf_2

# SVM using RBF kernel (C = 10, sigma = 0.01) gave highest accuracy in predicting digits.

# Model performance on test data set of 10000 instances 
# Overall Accuracy  95.66%
# Overall Sensitivites >= 93%
# Overall Specificities > 99%

# ---------------------------------------------------------------------------------------



