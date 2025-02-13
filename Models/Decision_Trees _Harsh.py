import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:/Users/harsh/Downloads/subset_pollution_dataset-1 (3).csv"
data = pd.read_csv(file_path)

# Take the first 200 rows as training data
train_data = data.iloc[:200]

# Features and target variable
X = train_data[['PM10', 'Proximity_to_Industrial_Areas']]
y = train_data['Dangerous']

# Encode the target variable (e.g., 'N' -> 0, 'Y' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets (80-20 split for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# Predictions for the training and validation sets
y_train_pred = tree_clf.predict(X_train)
y_val_pred = tree_clf.predict(X_val)

# Calculate Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    tree_clf,
    feature_names=['PM10', 'Proximity_to_Industrial_Areas'],
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization", fontsize=16)
plt.savefig("decision_tree_plot.png")
plt.show()

# Plot Decision Boundaries
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X['PM10'].min() - 1, X['PM10'].max() + 1
    y_min, y_max = X['Proximity_to_Industrial_Areas'].min() - 1, X['Proximity_to_Industrial_Areas'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict for every point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X['PM10'], X['Proximity_to_Industrial_Areas'], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('PM10')
    plt.ylabel('Proximity to Industrial Areas')
    plt.title(title)
    plt.show()

# Plotting decision boundaries for the training data
plot_decision_boundary(X_train, y_train, tree_clf, "Decision Boundary (Training Data)")


