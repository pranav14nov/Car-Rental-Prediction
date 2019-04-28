#Load Library and Data Set
setwd("F:/PRANAV\Project/Car Rental Prediction")
getwd()
data = read.csv("train_cab.csv", header = TRUE)
data1 = read.csv("test.csv", header = TRUE)
library(dplyr)
library(randomForest)
library(caret)
library(e1071)
head(data)
str(data)
summary(data)
data = select (data,c(passenger_count,fare_amount,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

data$fare_amount = as.numeric(data$fare_amount)

#Missing Value Analysis
unique(data$passenger_count)
unique(data$fare_amount)
unique(data$pickup_longitude)
unique(data$pickup_latitude)
unique(data$dropoff_longitude)
unique(data$dropoff_longitude)
unique(data$pickup_datetime)

data <- data[complete.cases(data$fare_amount), ]
data <- data[complete.cases(data$passenger_count), ]
data <- data[complete.cases(data$pickup_longitude), ]
data <- data[complete.cases(data$pickup_latitude), ]
data <- data[complete.cases(data$dropoff_longitude), ]
data <- data[complete.cases(data$pickup_datetime), ]


#Split data into test and train
set.seed(500)
train <- sample(nrow(data), 0.7*nrow(data), replace = FALSE)
TrainSet <- data[train,]
ValidSet <- data[-train,]
summary(TrainSet)
summary(ValidSet)

#Applying Model to Train data
model <- randomForest(fare_amount ~ ., data = TrainSet, importance = TRUE)
model
plot(model)
model2 <- randomForest(fare_amount ~ ., data = TrainSet, ntree = 100, mtry = 4, importance = TRUE)
model2
plot(model2)
#--------------------

#Predicting Values
Model_lm1=lm(fare_amount~.,data=data)
summary(Model_lm1)
View(data)


#Applying Model on Test Data
prediction = predict (Model_lm1, data1)
View(data1)
output = cbind(data1, prediction)
View(output)
