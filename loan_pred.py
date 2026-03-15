import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# For saving model and scaler
import joblib


df = pd.read_csv('loan_data.csv')


print(df.shape)
df.info()


df = df.dropna() 

pie_colors = ['pink', 'blue'] 

temp = df['Loan_Status'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(temp.values, 
        labels=temp.index, 
        autopct='%1.1f%%', 
        colors=pie_colors,  
        startangle=90, 
        explode=(0.05, 0)) 
plt.title('Loan Status Distribution')
plt.show()

# Countplots for Categorical Columns
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status', palette='magma') # Changed color palette here
plt.tight_layout()
plt.show()

# 3. Handling Outliers

df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400] 

# 4. Data Preprocessing
def encode_labels(data):
   
    data_encoded = data.copy()
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col])
    return data_encoded

df = encode_labels(df)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# 5. Model Preparation
features = df.drop('Loan_Status', axis=1) 
target = df['Loan_Status'].values 

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=10
) 

# Handle Imbalance
ros = RandomOverSampler(sampling_strategy='minority', random_state=0) 
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train) 

# Scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val = scaler.transform(X_val)

# 6. Model Training and Evaluation
model = SVC(kernel='rbf', probability=True) 
model.fit(X_resampled, Y_resampled)

# Predictions
train_preds = model.predict(X_resampled)
val_preds = model.predict(X_val)

print('Training ROC AUC Score:', roc_auc_score(Y_resampled, train_preds))
print('Validation ROC AUC Score:', roc_auc_score(Y_val, val_preds))
print('\nClassification Report:\n', classification_report(Y_val, val_preds))

# 7. Final Visualization: Confusion Matrix
cm = confusion_matrix(Y_val, val_preds)
plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='crest', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- Save model, scaler, and feature names for API use ---
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(features.columns), 'feature_names.pkl')
print('Model, scaler, and feature names saved as .pkl files.')