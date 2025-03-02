#%%
from ucimlrepo import fetch_ucirepo
import pandas as pd

#%%
# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("CDC_Diabetes_Health_Indicators.csv", index=False)
print("Done!")

#%%
# fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)

# data (as pandas dataframes)
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("secondary_mushrooms.csv", index=False)
print("Done!")
#%%
# fetch dataset
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)

# data (as pandas dataframes)
X = glioma_grading_clinical_and_mutation_features.data.features
y = glioma_grading_clinical_and_mutation_features.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("glioma_grading_clinical_and_mutation.csv", index=False)
print("Done!")
#%%
# fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

# data (as pandas dataframes)
X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("aids_clinical_trials_study_175.csv", index=False)
print("Done!")

#%%
# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("predict_students_dropout_and_academic_success.csv", index=False)
print("Done!")

#%%
# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

df = pd.concat([X,y], axis=1)
df.to_csv("breast_cancer_wisconsin_diagnostic.csv",index=False)
print("done")