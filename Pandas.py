import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for Todoism project
np.random.seed(42)
n_samples = 1000

# Simulate data for the project
data = {
    "user_id": np.arange(1, n_samples+1),
    "gender": np.random.choice(["Male", "Female", "Other"], n_samples),
    "age": np.random.randint(18, 70, n_samples),
    "subscription_rate": np.random.choice([0.1, 0.2, 0.5, 1.0], n_samples),
    "contract_length": np.random.choice([6, 12, 24, 36], n_samples),
    "completed_tasks": np.random.randint(0, 100, n_samples),
    "total_tasks": np.random.randint(100, 200, n_samples),
    "usage_hours_per_week": np.random.randint(1, 30, n_samples),
    "churned": np.random.choice([0, 1], n_samples)  # 0 = not churned, 1 = churned
}

# Create a DataFrame
df = pd.DataFrame(data)

# Gender distribution chart
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution of Todoism Users')
plt.show()

# Age distribution by gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='age', data=df)
plt.title('Age Distribution by Gender')
plt.show()

# Subscription rate distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['subscription_rate'], bins=10, kde=True)
plt.title('Subscription Rate Distribution')
plt.show()

# Contract length distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='contract_length', data=df)
plt.title('Contract Length Distribution')
plt.show()

# Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=15, kde=True)
plt.title('Age Distribution')
plt.show()

# Completed vs Total Tasks
plt.figure(figsize=(8, 6))
sns.scatterplot(x='completed_tasks', y='total_tasks', data=df)
plt.title('Completed vs Total Tasks')
plt.show()

# Usage Hours per Week vs Age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='usage_hours_per_week', data=df)
plt.title('Usage Hours per Week vs Age')
plt.show()

# Logistic Regression: Confusion Matrix
X = df[['age', 'subscription_rate', 'contract_length', 'usage_hours_per_week']]
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Random Forest Feature Importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features)
plt.title('Random Forest Feature Importance')
plt.show()

# Churn rate by contract length
plt.figure(figsize=(8, 6))
sns.barplot(x='contract_length', y='churned', data=df)
plt.title('Churn Rate by Contract Length')
plt.show()

# Churn rate by age group
age_groups = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65, 70], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66-70"])
df['age_group'] = age_groups

plt.figure(figsize=(8, 6))
sns.barplot(x='age_group', y='churned', data=df)
plt.title('Churn Rate by Age Group')
plt.show()

# Correlation Heatmap of features
corr = df[['age', 'subscription_rate', 'contract_length', 'completed_tasks', 'total_tasks', 'usage_hours_per_week']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Churn prediction with Random Forest
y_pred_rf = rf_model.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Churn prediction with Logistic Regression (Comparison)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
