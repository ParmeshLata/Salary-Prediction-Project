# Customer Salary Prediction

This project implements a deep learning model using TensorFlow and Keras to predict customer estimated salary based on various features such as credit score, age, balance, and other factors. The model is trained on the "Churn_Modelling.csv" dataset and deployed using Streamlit.

## Features
- Data preprocessing with label encoding and one-hot encoding.
- Feature scaling using StandardScaler.
- Deep learning model using TensorFlow/Keras.
- Early stopping to prevent overfitting.
- Streamlit-based UI for user input and salary prediction.

## Installation
To run this project, install the required dependencies:
- Scikit-Learn
- TensorFlow
- Keras
- Streamlit

## Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/ParmeshLata/Salary-Prediction-Project.git
   cd Salary-Prediction-Project
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run salary_regression_app.py
   ```

3. Enter customer details in the UI to get the predicted salary.

## Model Training
The model is trained using the following steps:
- Load and preprocess the dataset.
- Encode categorical variables.
- Scale features using StandardScaler.
- Train a neural network with ReLU activations.
- Monitor training with early stopping.

## Future Enhancements
- Improve model accuracy with hyperparameter tuning.
- Implement a more interactive UI.
- Deploy the model as a web API.

## License
The credit goes to https://github.com/krishnaik06 who taught me the Deep Learning and acted as a source for the dataset I have used.

---
### Author
Developed by https://github.com/ParmeshLata 🚀
