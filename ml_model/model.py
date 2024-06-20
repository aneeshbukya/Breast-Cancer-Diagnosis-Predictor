
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def create_model(data):
    """
    This function creates a model using the data provided. The data is split into training and testing sets.
    :param data: The data to be used to train the model
    :return: The model and the scaler used to scale the data
    """
    x_data = data.drop(["diagnosis"], axis=1)
    y_data = data["diagnosis"]

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)  # fit_transform is used to fit the data and then transform it

    # split the data into training and testing sets
    # the convention is to use 80% of the data for training and 20% for testing
    # the random state is set to 42 so that the results are reproducible which is also a convention
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Create the model and fit the data
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Test teh model and print the accuracy
    y_prediction = model.predict(x_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_prediction))
    print("Classification Report: ")
    print(classification_report(y_test, y_prediction))

    return model, scaler


def get_clean_data():
    """
    This function reads the data from the csv file and cleans it.
    :return: The cleaned data
    """
    data = pd.read_csv("data/data.csv") # read the data from the csv file
    data = data.drop(["Unnamed: 32", "id"], axis=1) # drop the columns that are not needed
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) # map the diagnosis column to 1 and 0 for M and B respectively
    return data


def main():
    """
    This is the main function which will be called when the script is run.
    """
    # Data Cleaning
    data = get_clean_data()
    # Model Creation and Training of the model using the cleaned data
    model, scaler = create_model(data)

    # Save the model and the scaler using pickle
    with open('ml_model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('ml_model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)


if __name__ == "__main__":
    main()
