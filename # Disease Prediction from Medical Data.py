import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# here is the dataset
data = pd.read_csv('D:/Downloads/archive (2)/heart.csv')

# Encoding of categorical variables
data = pd.get_dummies(data, drop_first=True)

# Splitting data into features and target variable
X = data.drop('output', axis=1)
y = data['output']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using three models
models = {
    'SVC': SVC(probability=True, random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Hyperparameter optimization using Grid Search
param_grid_svc = {'C': [0.1, 1, 5], 'kernel': ['linear', 'rbf']}   # using linear and rbf kernel cause these are effective after applying all 
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}   # using distance metrics
param_grid_nb = {}  # Naive Bayes doesn't have many hyperparameters

# GS for each model
grid_svc = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svc, cv=5)
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=5)

grids = {'SVC': grid_svc, 'KNeighbors': grid_knn, 'Naive Bayes': grid_nb}
for name, grid in grids.items():
    grid.fit(X_train, y_train)
    print(f"{name} best params: {grid.best_params_}")
    models[name] = grid.best_estimator_

# Cross-validation for each model
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # using 5 fold cross validation
    print(f"{name} Cross-Validation Accuracy: {np.mean(cv_scores):.2f}") 

# Ensemble model using Voting Classifier
ensemble = VotingClassifier(estimators=[
    ('svc', models['SVC']),
    ('knn', models['KNeighbors']),
    ('nb', models['Naive Bayes'])
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Evaluate ensemble model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {accuracy:.2f}")
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Ensemble Confusion Matrix')
plt.show()

# Predicting the probabilities
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

# Calculation of ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, lw=2, label=f'Ensemble (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
