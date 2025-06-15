import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

file_path = "/adult.csv"
df = pd.read_csv(file_path)

feature_encoders={}

for c in df.columns:
  if df[c].dtype=='object':
    le=LabelEncoder()
    df[c]=le.fit_transform(df[c])
    feature_encoders[c]=le

data = df.drop(df.columns[15], axis=1)

x =data.drop('income',axis=1)
y =df['income']

print(x.head())

x_train,x_test,y_train, y_test=train_test_split(x,y,test_size= 0.2,random_state=1)


#Model 1 SVM

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

svm_model = SVC()
svm_model.fit(x_train,y_train)

y_pred_svm = svm_model.predict(x_test)

print("SVM Accuracy Score:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - SVM')
plt.show()


#MOdel 2 Logistic regression

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)

y_pred_lr = lr_model.predict(x_test)

print("Logistic Regression Accuracy Score:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - LR')
plt.show()


#Model 3 Random forest

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid_rf, n_iter=10, cv=3, verbose=2, random_state=42)
random_search_rf.fit(x_train, y_train)

print("Best parameters for Random Forest:", random_search_rf.best_params_)

best_rf_model = random_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(x_test)

print("Tuned Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_best_rf))
print("Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_best_rf))

conf_matrix_best_rf = confusion_matrix(y_test, y_pred_best_rf)

plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_best_rf, annot=True, fmt='d', cmap='Purples', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Tuned Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Visualizing all Accuracy Comparision

import numpy as np

accuracies = {
    'SVM': accuracy_score(y_test, y_pred_svm),
    'Random Forest': accuracy_score(y_test, y_pred_best_rf),
    'Logistic Regression': accuracy_score(y_test, y_pred_lr)
}

plt.figure(figsize=(8,6))
plt.bar(accuracies.keys(), accuracies.values(), color=['Orange', 'Teal', 'Yellow'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()


#taking input field for testing

user_input=pd.DataFrame([list(input().split(','))],columns=x.columns)

for col in user_input.select_dtypes(include=['object']).columns:
    if col in feature_encoders:
        try:
            user_input[col] = feature_encoders[col].transform(user_input[col])
        except ValueError:
            print(f"Error: Invalid value for column {col}. Please provide a valid input.")
            exit()

example_prediction = best_rf_model.predict(user_input)

print(example_prediction[0])

if example_prediction[0] == 0:
  print('<=50K')
else:
  print('>50K')