# 🧠 Income Prediction using Machine Learning

This project predicts whether a person earns **more than $50K** per year using the **UCI Adult Income dataset**. The goal is to classify individuals based on their demographic and work-related attributes.

## 📌 Objective
To build and compare different classification models that predict income category (`<=50K` or `>50K`) based on features like age, education, occupation, and more.

## 📁 Dataset
- **Source**: [UCI Machine Learning Repository - Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
- **Target Column**: `income` (binary classification)

## ⚙️ Workflow
1. Data cleaning and preprocessing
2. Encoding categorical features
3. Model building:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest (with hyperparameter tuning)
4. Model evaluation using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

## 🏆 Best Model
- **Random Forest (with tuning)** performed the best in terms of accuracy and F1-score.

## 🔧 Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (sklearn)
- Seaborn & Matplotlib (for visualization)

## 📊 Results
- Random Forest provided the highest accuracy.
- Evaluation included confusion matrix plots and classification reports for detailed insight.

## 📌 How to Run
1. Clone this repo
2. Install required libraries: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `AdultIncomeClassifier.ipynb`

---

