# Synthetic dataset
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
from collections import Counter
# Visualization
import matplotlib.pyplot as plt
# Model and performance
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Create an imbalanced dataset
X, y = make_classification(n_samples=100000, n_features=10, n_informative=10,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.95, 0.05],
                           class_sep=0.5, random_state=0)
# Convert the data from numpy array to a pandas dataframe
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})
# Check the target distribution
print(df['target'].value_counts(normalize = True))


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def show_data(X, y):
    print('The number of records in the training dataset is', X.shape[0])
    print(f"The training dataset has {sorted(Counter(y).items())[0][1]} records for the majority class and {sorted(Counter(y).items())[1][1]} records for the minority class.")
    for i in range(10):
        print('X-y', X[i], y[i])

print(f"The training dataset has {sorted(Counter(y_train).items())[0][1]} records for the majority class and {sorted(Counter(y_train).items())[1][1]} records for the minority class.")


print("The training dataset:")
show_data(X_train, y_train)
print("The test dataset:")
show_data(X_test, y_test)

# Train the one class support vector machine (SVM) model
one_class_svm = OneClassSVM(nu=0.01, kernel = 'rbf', gamma = 'auto', verbose=True)

X_train_ = X_train[y_train == 0]  # Use only the majority class for training

one_class_svm.fit(X_train_)


def prediction_probability(svm, X, y):
    # Predict the anomalies
    prediction = svm.predict(X)
    # Change the anomalies' values to make it consistent with the true values
    prediction = [1 if i==-1 else 0 for i in prediction]
    # Check the model performance
    print(classification_report(y, prediction))


print("The training dataset performance:")
prediction_probability(one_class_svm, X_train, y_train)

print("The test dataset performance:")
prediction_probability(one_class_svm, X_test, y_test)