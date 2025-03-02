import pandas as pd
from ray.data.preprocessors import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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


def lReg(df, target_var):
    x = df.drop(columns=[target_var])
    y = df[target_var]

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []

    plt.figure(figsize=(10, 6))

    for train_idx, test_idx in outer_cv.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(solver='saga')
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold AUC = {roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10x10 Fold CV')
    plt.legend(loc='lower right', fontsize=8)
    plt.show()

    print(f'Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    print(f'F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')


lReg(df1, 'Diabetes_binary')
lReg(df2, 'cid')
lReg(df3, 'Grade')
lReg(df4, 'Diagnosis')
lReg(df5, 'class')
