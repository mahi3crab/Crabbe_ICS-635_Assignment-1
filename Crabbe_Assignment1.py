# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Add the target column

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# Partition into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

# Scale features using StandardScaler (Only fit on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Print shape to verify
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train K-Nearest Neighbors (KNN) with n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)  # Train on scaled data

# Train Decision Tree with default parameters
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# train Random Forest with 150 trees
rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train, y_train)

# make predictions
y_predict_knn = knn.predict(X_test_scaled)
y_predict_dt = dt.predict(X_test)
y_predict_rf = rf.predict(X_test)

# evaluate models using accuracy score
accuracy_knn = accuracy_score(y_test, y_predict_knn)
accuracy_dt = accuracy_score(y_test, y_predict_dt)
accuracy_rf = accuracy_score(y_test, y_predict_rf)

# display accuracy
print(f"KNN Accuracy: {accuracy_knn:.4f}")
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Display classification reports
# Class 0 (Malignant), Class 1 (Benign)
# Precision: high precision --> fewer false positives.
# Recall (Sensitivity or True Positive Rate): high recall --> fewer false negatives.
# F1-Score: high F1-score --> a good balance between precision and recall.
# Accuracy: gives an overall measure of performance.
# Support: number of actual occurrences of each class in the test dataset.
print("\nKNN Classification Report:\n", classification_report(y_test, y_predict_knn))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_predict_dt))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_predict_rf))

# Decision Tree Hyperparameters
dt_hyper = {}
for depth in [3, 5, 10, None]:  # experiment with different depth values
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    y_predict_dt = dt.predict(X_test)
    dt_hyper[depth] = accuracy_score(y_test, y_predict_dt)

print("\nDecision Tree Accuracy for different max_depth values:")
for depth, acc in dt_hyper.items():
    print(f"max_depth={depth}: Accuracy={acc:.4f}")

# Random Forest Hyperparameters
rf_hyper = {}
for depth in [3, 5, 10, None]:  # experiment with different max_depth values
    for min_samples in [2, 5, 10]:  # experiment with different min_samples_split values
        rf = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_split=min_samples)
        rf.fit(X_train, y_train)
        y_predict_rf = rf.predict(X_test)
        rf_hyper[(depth, min_samples)] = accuracy_score(y_test, y_predict_rf)

print("\nRandom Forest Accuracy for different max_depth and min_samples_split values:")
for (depth, min_samples), acc in rf_hyper.items():
    print(f"max_depth={depth}, min_samples_split={min_samples}: Accuracy={acc:.4f}")

# More concise version of results
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# evaluate models function --> in order to display results for each model
def eval_model(name, y_test, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

# Evaluate each model
knn_results = eval_model("KNN", y_test, y_predict_knn)
dt_results = eval_model("Decision Tree", y_test, y_predict_dt)
rf_results = eval_model("Random Forest", y_test, y_predict_rf)

# Create DataFrame
df_results = pd.DataFrame([knn_results, dt_results, rf_results])

# Display results
print(df_results)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_predict, model_name):
    confusion_mat = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Oranges", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(model_name)
    plt.show()

# Generate confusion matrices
plot_confusion_matrix(y_test, y_predict_knn, 'KNN Confusion Matrix')
plot_confusion_matrix(y_test, y_predict_dt, 'Decision Tree Confusion Matrix')
plot_confusion_matrix(y_test, y_predict_rf, 'Random Forest Confusion Matrix')

# Store results
knn_results = []
dt_results = []
rf_results = []

# hyperparameter values of n_neighbors for KNN
for k in [5, 7, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_predict = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_predict)
    knn_results.append({"Model": "KNN", "n_neighbors": k, "Accuracy": acc})

# hyperparameter max_depth values for Decision Tree
for depth in [5, 7, 15, None]:
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    dt_results.append({"Model": "Decision Tree", "max_depth": depth, "Accuracy": acc})

# hyperparameter max_depth and min_samples_split for Random Forest
for depth in [5, 7, 15, None]:
    for min_samples in [2, 5, 10]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_split=min_samples)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        rf_results.append({"Model": "Random Forest", "max_depth": depth, "min_samples_split": min_samples, "Accuracy": acc})

# Convert results to DataFrame for better visualization
knn_df = pd.DataFrame(knn_results)
dt_df = pd.DataFrame(dt_results)
rf_df = pd.DataFrame(rf_results)

# Display results
print("\nKNN Hyperparameter Impact:\n", knn_df)
print("\nDecision Tree Hyperparameter Impact:\n", dt_df)
print("\nRandom Forest Hyperparameter Impact:\n", rf_df)