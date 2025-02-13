import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = r'C:/Users/harsh/Downloads/subset_pollution_dataset-1 (3).csv'
data = pd.read_csv(file_path)

# Encode the target variable (Dangerous: N -> 0, Y -> 1)
data['Dangerous'] = data['Dangerous'].map({'N': 0, 'Y': 1})

# Prepare features (X) and target (y)
X = data[['PM10', 'Proximity_to_Industrial_Areas']]
y = data['Dangerous']

# Split the data (first 200 rows for training, rest for testing)
X_train, X_test = X.iloc[:200], X.iloc[200:]
y_train, y_test = y.iloc[:200], y.iloc[200:]

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Handle missing values in the test target
X_test_clean = X_test[~y_test.isna()]
y_test_clean = y_test[~y_test.isna()]

# Predict on the cleaned test set
y_pred_clean = model.predict(X_test_clean)

# Evaluate the model
accuracy_clean = accuracy_score(y_test_clean, y_pred_clean)
report_clean = classification_report(y_test_clean, y_pred_clean)
conf_matrix_clean = confusion_matrix(y_test_clean, y_pred_clean)

# Print results
print("Accuracy:", accuracy_clean)
print("\nClassification Report:\n", report_clean)
print("\nConfusion Matrix:\n", conf_matrix_clean)

# Visualization: Decision boundary
x_min, x_max = X['PM10'].min() - 1, X['PM10'].max() + 1
y_min, y_max = X['Proximity_to_Industrial_Areas'].min() - 1, X['Proximity_to_Industrial_Areas'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_train['PM10'], X_train['Proximity_to_Industrial_Areas'], c=y_train, edgecolor='k', cmap='coolwarm', label='Training Data')
plt.scatter(X_test_clean['PM10'], X_test_clean['Proximity_to_Industrial_Areas'], c=y_test_clean, edgecolor='k', cmap='viridis', marker='x', label='Test Data')
plt.xlabel('PM10 Concentration (µg/m³)')
plt.ylabel('Proximity to Industrial Areas (km)')
plt.title('Decision Boundary and Data Points')
plt.legend()
plt.show()

# Visualization: Confusion matrix heatmap
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_clean, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Dangerous', 'Dangerous'], yticklabels=['Not Dangerous', 'Dangerous'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
