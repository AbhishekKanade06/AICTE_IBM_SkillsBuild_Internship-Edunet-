# 💼 Employee Income Prediction App

This is a Streamlit web application that predicts whether a person's income exceeds $50K/year based on demographic and employment features. The model is trained on the UCI Adult Income Dataset.

---

## 🚀 Features

- Predict income class (`<=50K` or `>50K`)
- User-friendly sidebar input controls
- Real-time prediction using a trained machine learning model
- Supports 11 input features

---

## 📊 Input Features

1. Age  
2. Final Weight (fnlwgt)  
3. Workclass  
4. Educational Number  
5. Marital Status  
6. Occupation  
7. Relationship  
8. Race  
9. Gender  
10. Hours Worked per Week  
11. Native Country  

---

## 🧠 Model Details

- Trained using multiple classifiers: Decision Tree, Random Forest, SVM, Neural Network, Logistic Regression, Naive Bayes
- Best performing model (e.g., `DecisionTreeClassifier`) saved as `best_model.pkl`
- Categorical columns encoded using `LabelEncoder` (stored in `label_encoders.pkl`)

---
## App:[Link Text] https://lf3rkxqw7bz2xiw4j9j83m.streamlit.app

## 🛠 How to Run

### 1. Clone the Repository or Upload Files

Make sure these files are in the same folder:

- `employee_income_app.py`  
- `best_model.pkl`  
- `label_encoders.pkl`  
- `requirements.txt`  

### 2. Install Dependencies

```bash
pip install -r requirements.txt
