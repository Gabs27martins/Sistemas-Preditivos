import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
)  # Partição de dados em treino e teste
from sklearn.ensemble import (
    RandomForestClassifier,
)  # Modelo que combina Árvores de Decisão
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)  # Modelo baseado em única árvore
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)  # Importando a taxa de acerto e o resumo de classificação
import pickle

df = pd.read_csv(r"instances/loan_data.csv", sep=",", encoding="latin1")


print(df.head())
print(df.shape)
print(df.dtypes)

print(df["loan_status"].value_counts())
print(df["person_gender"].value_counts())
print(df["person_education"].value_counts())
print(df["person_home_ownership"].value_counts())
print(df["loan_intent"].value_counts())
print(df["previous_loan_defaults_on_file"].value_counts())

print(df.isnull().sum())

print(df.duplicated().sum())

# Visualização de dados

sns.countplot(data=df, x="loan_status", hue="person_home_ownership")
plt.show()

df["person_gender"] = df["person_gender"].map({"male": 0, "female": 1})
df["person_education"] = df["person_education"].map(
    {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
)
df["person_home_ownership"] = df["person_home_ownership"].map(
    {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3}
)
df["loan_intent"] = df["loan_intent"].map(
    {
        "EDUCATION": 0,
        "MEDICAL": 1,
        "VENTURE": 2,
        "PERSONAL": 3,
        "DEBTCONSOLIDATION": 4,
        "HOMEIMPROVEMENT": 5,
    }
)
df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map({'Yes':0, 'No':1})


X = df.drop("loan_status", axis=1)
Y = df["loan_status"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, stratify=Y, test_size=0.2, random_state=0
)

DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)
DT_pred = DT.predict(X_test)

RF = RandomForestClassifier(n_estimators=150)
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

print(f"Árvore Decisão Teste: {accuracy_score(Y_test, DT_pred)}")
print(f"Random Florest Teste: {accuracy_score(Y_test, RF_pred)}")

print(classification_report(Y_test, RF_pred))


# plt.figure(figsize=(12, 8))
# features_names = X.columns.to_list()
# class_names = Y.unique().astype(str).tolist()
# plot_tree(
#     DT, feature_names=features_names, class_names=class_names, filled=True, rounded=True
# )

# plt.show()

with open('emprestimo-modelo-preditivo.pkl', 'wb') as f:
    pickle.dump(RF, f)


print(df['credit_score'].describe())
