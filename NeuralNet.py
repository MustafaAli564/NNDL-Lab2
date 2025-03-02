import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('Datasets/CDC_Diabetes_Health_Indicators.csv')
df2 = pd.read_csv('Datasets/aids_clinical_trials_study_175.csv')
df3 = pd.read_csv('Datasets/glioma_grading_clinical_and_mutation.csv')
df4 = pd.read_csv('Datasets/breast_cancer_wisconsin_diagnostic.csv')
df5 = pd.read_csv('Datasets/secondary_mushrooms.csv')

df3 = pd.get_dummies(df3, columns=['Race'], drop_first=True)

le = LabelEncoder()
df4['Diagnosis'] = le.fit_transform(df4['Diagnosis'])

for col in df5.select_dtypes(include=['object']).columns:
    df5[col] = le.fit_transform(df5[col])


def NN(df, target_var):
    x = df.drop(columns=[target_var])
    y = df[target_var]

    # Scale the features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # 10x10 Stratified Cross-Validation
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []

    plt.figure(figsize=(10, 6))

    for train_idx, test_idx in outer_cv.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define the 3-layer Neural Network model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary Classification

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.01),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train)

        # Predictions
        y_prob = model.predict(x_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        # Store evaluation metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold AUC = {roc_auc:.2f}')

    # Plot ROC Curve
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10x10 Fold CV')
    plt.legend(loc='lower right', fontsize=8)
    plt.show()

    print(f'Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    print(f'F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')


NN(df1, 'Diabetes_binary')
NN(df2, 'cid')
NN(df3, 'Grade')
NN(df4, 'Diagnosis')
NN(df5, 'class')
