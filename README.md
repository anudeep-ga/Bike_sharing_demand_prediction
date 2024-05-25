
# Bike Sharing Demand Prediction

## Problem Statement

The primary objective of this project is to develop and deploy a machine learning model for demand
prediction in the context of a bike rental company. Specifically, the aim is to create a predictive model
that can accurately forecast the total count of rental bikes (comprising both casual and registered
users) based on a range of relevant features, including date, time, weather conditions, and various
contextual factors. By achieving this goal, the project seeks to address the following key objectives:

1. Demand Forecasting: 

Develop a machine learning model that can provide accurate predictions
of bike rental demand for different date and time scenarios.

2. Factor Analysis: 

Identify and understand the factors that influence bike rental demand, such
as seasonal variations, weather conditions, and day of the week.

3. Operational Optimization: 

Enable the bike rental company to make informed decisions
regarding fleet management, staffing, and marketing strategies to enhance operational efficiency.

4. Improved Service: 

Enhance customer experience by ensuring the availability of bikes when
and where they are needed, thereby meeting customer demand effectively.
By accomplishing these objectives, the project aims to contribute to the company's operational
excellence and economic sustainability. This report outlines the methodologies, findings, and insights
derived from the project, demonstrating the practical utility of machine learning in the bike rental
industry.

## Problem definition

The only method of preventing diabetes complications is to identify and treat the disease early.
The early detection of diabetes is important because its complications increase over time. Also,
prediction of diabetes at an early stage can lead to improved treatment. The intention of this
project is to build supervised models like Logistic regression, K nearest neighbors, Support
Vector Machines, Random Forest and Decision trees and select the best algorithm which can
perform early prediction of diabetes for a patient with high accuracy and analyze the variables
which are more responsible for causing diabetes.

## Data Source

Historical data was acquired from the provided dataset, sourced from https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. The dataset contains information on date, time, weather conditions, and user
categories (casual and registered). The attributes, detailed above, encompass season, year, hour,
temperature, humidity, wind speed, and more. This extensive dataset is a valuable asset for
scrutinizing the factors influencing bike rental demand across diverse conditions and time periods.

## Data Description

The Bike Sharing Dataset is a collection of historical records capturing the usage of a bike rental
service. It encompasses a total of 17,389 instances, each described by 16 features. This dataset serves
as a valuable resource for predicting bike rental demand, encompassing a mix of numerical and
categorical data.
The dataset has bike rental details, covering information like date, season, year, hour, weather
conditions, temperature, humidity, wind speed, and user counts. The "weathersit" column classifies
weather into four categories, spanning clear skies to heavy rain or snow. Normalized temperature
metrics, "temp" and "atemp," represent the actual and perceived temperatures. "Holiday" and
"workingday" columns distinguish regular and holiday schedules, weekdays, and weekends,
impacting bike rental trends. The dataset encompasses counts of casual and registered users, along
with the overall count of rented bikes. This extensive dataset serves as a valuable resource for
analyzing the factors influencing bike rental demand and behavior across diverse conditions and time
frames.

## Data Preprocessing, Feature Selection and Engineering and Exploratory Data Analysis (EDA)


### 1. Denormalization:
To restore original information, denormalization was performed for 'temp', 'atemp', 'hum',
'windspeed' columns. This step ensures that the data is in its original scale, providing a more intuitive
understanding of temperature, humidity, and wind speed.

### 2. Feature Classification:
Features were categorized into numerical and categorical groups. Numerical features included
'temp', 'atemp', 'hum', 'windspeed', while categorical features encompassed 'season', 'holiday', 'mnth',
'hr', 'weekday', 'workingday', and 'weathersit'. This distinction aids in applying specific preprocessing
techniques tailored to each type.

### 3. Data Description:
The data description, involving measures such as mean, standard deviation, and skewness, was
crucial for understanding key characteristics of our dataset. Central tendency metrics like mean and
median offered insights into typical values, while standard deviation indicated the data's variability.
Assessing skewness helped identify the distribution's shape, guiding decisions on normalization or
transformation. This understanding informed preprocessing decisions and model selection,
contributing to effective analysis. In essence, a comprehensive data description for our dataset formed
the basis for robust and reliable machine learning analyses.

### 4. Checking for Missing Values:
A comprehensive check for missing values was conducted. The absence of missing values ensures
the integrity and completeness of the dataset, forming a solid foundation for subsequent analysis and
modeling.

### 5. Outlier Detection:
Box plots were employed to visualize and identify outliers in the data. Insights from the plots
revealed interesting patterns, such as higher bike rentals on regular workdays and peak rental times at
8 am and 5 pm. Additionally, the impact of temperature on rental patterns was observed.

### 6. Outlier Handling:
Outliers were addressed using a robust approach, utilizing the median and interquartile range
(IQR). This process helped in preserving the integrity of the dataset while mitigating the influence of
extreme values. Before the outlier analysis there were 12165 rows and after removing there are 11783
rows.

### 7. Checking Data Format:
Ensured that all columns were in a suitable numeric format, including integers, floats, and date
formats. Consistent data formatting is essential for compatibility with machine learning algorithms.

### 8. Correlation Analysis:
Investigated the correlation between features and discovered a high correlation (0.94) between
'temp' and 'atemp'. Recognizing this redundancy, the decision was made to remove the 'atemp' column
to avoid multicollinearity.
These meticulous preprocessing steps collectively contribute to a cleaner, more refined dataset, setting
the stage for effective machine learning model development and analysis.

## Model Evaluation:

Brief insights on the model evaluation based on RMSE values:
1. Closed Form Solution:
- Training Data: RMSE is 139.50
- Testing Data: RMSE is 136.22
- The closed form solution demonstrates good predictive accuracy on both training and testing datasets.
2. Closed Form Solution with Regularization:
- Training Data: RMSE is 139.50
- Testing Data: RMSE is 136.22
- The addition of regularization maintains predictive accuracy on par with the closed form solution.
3. Gradient Descent:
- Training Data: RMSE is 139.56
- Testing Data: RMSE is 136.24
- The gradient descent approach yields comparable performance to closed form solutions.
4. Gradient Descent with Regularization:
- Training Data: RMSE is 139.74
- Testing Data: RMSE is 136.31
- Regularized gradient descent maintains reasonable predictive accuracy.
5. Lasso Regression - Gradient Descent:
- Training Data: RMSE is 139.50
- Testing Data: RMSE is 136.23
- Lasso regression with gradient descent demonstrates effectiveness in predictive modeling.
6. Lasso Regression - Stochastic Gradient Descent:
- Training Data: RMSE is 143.48
- Testing Data: RMSE is 140.93
- Lasso regression with stochastic gradient descent exhibits slightly higher RMSE values.
7. Neural Networks:
- Training Data: RMSE is 68.07
- Testing Data: RMSE is 70.66
- Neural networks outperform traditional regression methods, achieving lower RMSE values on both training and testing datasets.
In summary, all models perform reasonably well, with neural networks demonstrating superior
performance in minimizing RMSE on both training and testing data.

## Conclusion:

Neural Network (rmse : 70.65) and Linear regression shows promising result, therefore it can be used
to solve this problem.
 - Bike rental count is high during week days than on weekend.
-  Bike demand shows peek around 8-9 AM in the morning and 6 - 7pm in the evening.
 - People prefer to rent bike more in summer than in winter.
 - Bike demand is more on clear days than on snowy or rainy days.
 - Temperature range from 22 to 25(°C) has more demand for bike.
'Hour', 'Temperature(°C)', 'Humidity', 'Wind_speed','Visibility ', 'Dew_point_temperature',
'Solar_Radiation', 'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'month', 'day of week '
regulates bike demand.
