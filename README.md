# Real Estate Price Prediction

## Overview
This project involves building a machine learning model to predict real estate prices based on specific features such as the distance to the nearest MRT station, the number of convenience stores, latitude, and longitude. The project also includes a web application for predicting real estate prices using these features.

## Dataset
The dataset used in this project is stored in a CSV file named Real_Estate.csv. It contains the following columns:

1. Distance to the nearest MRT station
2. Number of convenience stores
3. Latitude
4. Longitude
5. House price of unit area

## Prerequisites
Before running the code, ensure you have the following libraries installed:

1. pandas
2. scikit-learn
3. dash

## Instructions
1. Load the Dataset:
Update the path to your dataset in the pd.read_csv() function:

>>real_estate_data = pd.read_csv("Real_Estate.csv")

2. Train the Model:
The code splits the dataset into training and testing sets, initializes a linear regression model, and trains the model on the training set.

3. Run the Web Application:
The code includes a Dash web application that allows users to input the required features and get a predicted house price. To run the app, execute the script:

>>python your_script_name.py

4. Use the Web Application:
Open your web browser and go to http://127.0.0.1:8050/. You will see a form where you can input:

a. Distance to MRT Station (meters)
b. Number of Convenience Stores
c. Latitude
d. Longitude

After filling in these details, click the "Predict Price" button to see the predicted house price.



## This is the output of the project:
![alt text](image.png)