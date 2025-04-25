# Importing required dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# ----- Data Overview Functions -----

def get_shape():
    print("Number of rows:", diabetes_dataset.shape[0])
    print("Number of columns:", diabetes_dataset.shape[1])

def get_head():
    print("\nHead of the dataset:")
    print(diabetes_dataset.head())

def get_description():
    print("\nStatistical measures description:")
    print(diabetes_dataset.describe())

def get_counts():
    print("\nDiabetes value counts (0 = non-diabetic, 1 = diabetic):")
    print(diabetes_dataset["Outcome"].value_counts())

def get_means():
    print("\nMean values grouped by outcome:")
    print(diabetes_dataset.groupby("Outcome").mean())

# ----- Data Preparation -----
# Split features and labels
data = diabetes_dataset.drop(columns="Outcome", axis=1)
label = diabetes_dataset["Outcome"]

# Standardize the data
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# Train-test split
data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.2, stratify=label, random_state=2
)

# ----- Model Training Function -----
def train_model():
    model = svm.SVC(kernel="linear")
    model.fit(data_train, label_train)
    return model

# ----- Accuracy Functions -----
def train_data_accuracy(model):
    predictions = model.predict(data_train)
    accuracy = accuracy_score(label_train, predictions)
    print("Accuracy score of the training data:", accuracy)

def test_data_accuracy(model):
    predictions = model.predict(data_test)
    accuracy = accuracy_score(label_test, predictions)
    print("Accuracy score of the testing data:", accuracy)

# ----- Prediction System Function -----
def predictive_system(model):
    input_data = input("Enter the data to predict (comma-separated numbers): ").split(',')

    # Filter out any empty strings and strip spaces
    cleaned_input = [i.strip() for i in input_data if i.strip() != '']

    try:
        input_data_as_numpy_array = np.asarray([float(i) for i in cleaned_input])
    except ValueError:
        print("❌ Invalid input. Please enter only numeric values separated by commas.")
        return

    # For diabetes model:
    if len(input_data_as_numpy_array) != 8:
        print("❌ Invalid number of input features. Expected 8 values.")
        return

    # Wrap input in DataFrame and standardize
    feature_names = diabetes_dataset.drop(columns="Outcome").columns
    input_df = pd.DataFrame([input_data_as_numpy_array], columns=feature_names)

    standardized_input = scaler.transform(input_df)
    prediction = model.predict(standardized_input)

    print("\nPrediction:", prediction)
    print("The person is Diabetic" if prediction[0] == 1 else "The person is Non-diabetic")


# ----- Run the System -----
if __name__ == "__main__":
    get_shape()
    get_head()
    get_description()
    get_counts()
    get_means()
    
    model = train_model()
    train_data_accuracy(model)
    test_data_accuracy(model)
    predictive_system(model)
