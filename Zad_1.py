import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


diabetes  = pd.read_csv('data/diabetes.csv')
print(diabetes)

features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
target = 'Diabetic'

X,y = diabetes[features], diabetes[target]

X_train, X_test, y_train, y_test = train_test_split(X[['Pregnancies', 'Age']], y, test_size=0.30, random_state=0, stratify=y)
print ('Treningowe obserwacje: %d\nTestowe obserwacje: %d' % (X_train.shape[0], X_test.shape[0]))

scaler_2var = StandardScaler()
X_train_standardized = scaler_2var.fit_transform(X_train)
X_test_standardized = scaler_2var.transform(X_test)

models_names = []
predictions_proba_list = []

regularization_strengths = [0.01, 0.1, 1, 10, 100]

models = []
for strength in regularization_strengths:
    model = LogisticRegression(C=strength)
    models.append(model)

def evaluate_model(models, X_train, X_test, y_train, y_test):
    for i, model in enumerate(models):
        print(f"Model {i + 1}: Regularization Strength = {model.C}")

        # Train the model
        model.fit(X_train_standardized, y_train)

        # Calculate training metrics
        predictions_train = model.predict(X_train_standardized)
        predictions_proba_train = model.predict_proba(X_train_standardized)

        # Calculate test metrics
        predictions_test = model.predict(X_test_standardized)
        predictions_proba_test = model.predict_proba(X_test_standardized)

        # Print classification report for training set
        print("Training Set Classification Report:")
        print(classification_report(y_train, predictions_train))
        print("\n")

        # Print confusion matrix for training set
        plt.figure()
        cm = confusion_matrix(y_train, predictions_train)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')
        ax.set_title('Confusion Matrix - Training Set\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        plt.show()

        # Plot ROC curve for training set
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        fpr, tpr, thresholds = roc_curve(y_train, predictions_proba_train[:, 1])
        plt.plot(fpr, tpr, label=f'Model {i + 1}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Training Set\nModel {i + 1}')
        plt.legend(loc='lower right')
        plt.show()

        # Calculate F1 score for training set
        f1_train = f1_score(y_train, predictions_train)
        auc_train = roc_auc_score(y_train, predictions_proba_train[:, 1])

        print(f"F1 Score (Training): {f1_train:.4f}")
        print(f"AUC (Training): {auc_train:.4f}\n")

        # Print classification report for test set
        print("Test Set Classification Report:")
        print(classification_report(y_test, predictions_test))
        print("\n")

        # Print confusion matrix for test set
        plt.figure()
        cm = confusion_matrix(y_test, predictions_test)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')
        ax.set_title('Confusion Matrix - Test Set\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        plt.show()

        # Plot ROC curve for test set
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        fpr, tpr, thresholds = roc_curve(y_test, predictions_proba_test[:, 1])
        plt.plot(fpr, tpr, label=f'Model {i + 1}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Test Set\nModel {i + 1}')
        plt.legend(loc='lower right')
        plt.show()

        # Calculate F1 score for test set
        f1_test = f1_score(y_test, predictions_test)
        auc_test = roc_auc_score(y_test, predictions_proba_test[:, 1])

        print(f"F1 Score (Test): {f1_test:.4f}")
        print(f"AUC (Test): {auc_test:.4f}\n")

    return models


# Execute the evaluation
evaluate_model(models, X_train_standardized, X_test_standardized, y_train, y_test)
