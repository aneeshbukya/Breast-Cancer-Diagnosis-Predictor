# Breast Cancer Diagnosis Predictor

## Project Overview

The Breast Cancer Predictor is a web application designed to assist medical professionals in diagnosing breast cancer. It utilizes a machine learning model trained on cytology lab measurements to predict whether a breast mass is benign or malignant.

## Features

- **Interactive Sidebar:** Input cell nuclei measurements using sliders.
- **Radar Chart:** Visual representation of the input data across various features.
- **Prediction Display:** Shows whether the mass is predicted to be benign or malignant, along with the probabilities for each class.
- **Model Training:** Includes a script for training a logistic regression model on breast cancer data.

## Project Structure

```bash
├── data/
│   └── data.csv                # Dataset file
├── frontend/
│   └── style.css               # Custom CSS for the Streamlit app
├── ml_model/
│   ├── model.pkl               # Trained machine learning model
│   └── scaler.pkl              # Scaler used to scale the data
├── Breast_Cancer_Predictor.py  # Main script for the Streamlit app
├── train_model.py              # Script to train and save the model
└── README.md                   # Project documentation

## Requirements

Python 3.7+
Required libraries are listed in requirements.txt.
Setup Instructions

# Step 1: Install Dependencies
Create a virtual environment and install the required libraries:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
#Step 2: Data Preparation
Ensure the dataset (data.csv) is placed in the data directory. The dataset should contain the necessary columns for the model to make predictions.

# Step 3: Train the Model
To train the model, run the train_model.py script. This will create and save the model and scaler in the ml_model directory.

bash
Copy code
python train_model.py
Step 4: Run the Streamlit App
To run the Streamlit application, use the following command:

bash
Copy code
streamlit run Breast_Cancer_Predictor.py
Usage

Sidebar Inputs: Use the sliders in the sidebar to input measurements.
Prediction: The app will display whether the mass is predicted to be benign or malignant, along with the probabilities.
Radar Chart: Visualize the input measurements across different categories.
Model Training

The train_model.py script handles data cleaning, scaling, and training the logistic regression model. It saves the trained model and scaler for use in the Streamlit app.

# Data Cleaning

The get_clean_data function reads the data from data.csv, drops unnecessary columns, and maps the diagnosis column to binary values (1 for malignant, 0 for benign).

# Custom Styling

The frontend/style.css file contains custom CSS to style the Streamlit app.

# Acknowledgements

This project uses a dataset from the UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set. The machine learning model and the Streamlit app provide a tool to aid in diagnosis but should not replace professional medical advice.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.
