🎯 Diabetes-Prediction-Classifier  
A machine learning project using Support Vector Machine (SVM) to predict whether a person is diabetic based on health-related measurements. Includes data preprocessing, model training, evaluation, and an interactive CLI-based prediction system. Built with Python, NumPy, Pandas, and scikit-learn.

📖 Description  
This project implements a binary classification system using the Pima Indians Diabetes Dataset. It demonstrates a full machine learning pipeline — from data loading and exploration to feature scaling, training with SVM, model evaluation, and prediction using a command-line interface.

📁 Dataset  
**Source**: Kaggle – Pima Indians Diabetes Dataset  
- 8 numerical features including glucose level, BMI, insulin, age, etc.  
- Target label:  
  - `0` = Non-diabetic  
  - `1` = Diabetic  

🛠️ Installation  
Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn
```
🚀 How to Run
Clone the repository:

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-Classifier.git
cd Diabetes-Prediction-Classifier
```
Make sure diabetes.csv is present in the root directory.

Run the `Python` script:

```bash
python diabetes_classifier.py
```
When prompted, enter 8 comma-separated float values (e.g., glucose, BMI, age, etc.) to predict whether the person is diabetic.

🧠 Model Used
Support Vector Machine (SVM)

Linear kernel for binary classification

Effective for small to medium-sized structured datasets

📊 Output

Accuracy on training and testing sets

Interactive CLI prediction output:

"The person is Diabetic"

"The person is Non-diabetic"

📂 Project Structure

```bash
├── diabetes.csv              # Dataset file
├── diabetes_classifier.py    # Main Python script
└── README.md                 # Project documentation
📌 Notes
```
Feature standardization is used to improve model performance

The input system is robust against invalid or extra values

🔗 Dataset Link
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
