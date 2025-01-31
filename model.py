import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split # Partição de dados em treino e teste
from sklearn.ensemble import RandomForestClassifier # Modelo que combina Árvores de Decisão
from sklearn.tree import DecisionTreeClassifier, plot_tree # Modelo baseado em única árvore
from sklearn.metrics import classification_report, accuracy_score # Importando a taxa de acerto e o resumo de classificação
import pickle

df = pd.read_csv(r'instaces/medicamentos.csv', sep=';', encoding='latin1')

'''
print(df.head())
print(df.shape)

print(df['Medicamento'].value_counts()) # Retorna frequência de valores de uma variável

print(df['Potassio'].describe()) # Retorna o resumo da descrição de uma variavel quantitativa

print(df.isnull().sum()) # Verifica a quantidade de dados nulos

print(df.duplicated().sum()) # Verifica a quantidade de dados duplicados

# Visualização dos Dados

sns.countplot(data=df, x='Medicamento', hue='Sexo') #Visualizando os medicamentos categorizados pela variável Sexo
plt.show()


sns.countplot(data=df, x='Medicamento', hue='Pressão') #Visualizando os medicamentos categorizados pela variável Sexo
plt.show()
'''

print(df['Potassio'].describe()) # Retorna o resumo da descrição de uma variavel quantitativa

df['Pressão'] = df['Pressão'].map({'Baixo': 0, 'Normal':1, 'Alto':2}) # Mapeando a variavel para números
df['Colesterol'] = df['Colesterol'].map({'Normal': 1, 'Alto':2})
df['Sexo'] = df['Sexo'].map({'M': 0, 'F':1})
df['Medicamento'] = df['Medicamento'].map({'Y' : 0, 'X' : 1, 'A' : 2, 'B' : 3, 'C' : 4})

X = df.drop('Medicamento', axis = 1) # X é o conjunto de variaveis explicativas

Y = df['Medicamento'] # Y é o Rótulo ou a variável resposta

#Função para particionar os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=0)

DT = DecisionTreeClassifier() # Técnica de Árvore de Decisão
DT.fit(X_train, Y_train) # O comando fit recebe x e y (treino) para aprender padrões
DT_pred = DT.predict(X_test) # O comando predict retorna predições, ou seja, respostas

RF = RandomForestClassifier(n_estimators=100) # Técnica de várias Árvores de Decisões
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

print(accuracy_score(Y_test, DT_pred))


'''
Os dados y_test são usados como um gabarito para verificar o quanto os modelos acertaram
comparando o valor verdadeiro (y_test) com o valor predito pelos modelos
'''
print("Árvore de Decisão:",accuracy_score(Y_test, DT_pred)) # Calcula a taxa de acerto
print("Random Forest:",accuracy_score(Y_test, RF_pred))

'''
Exibe a estrutura do modelo de Árvore de Decisão
'''
plt.figure(figsize=(12, 8))
caracteristicas_names = X.columns.tolist()
rotulo_name = Y.unique().astype(str).tolist()
plot_tree(DT, 
          feature_names=caracteristicas_names,  # Nomes das características
          class_names=rotulo_name,    # Nomes das classes do rótulo
          filled=True,                      # Preencher com cores
          rounded=True)                     # Nós arredondados
plt.show()


 # Salvando o modelo treinado e criando um arquivo

with open('modelo-preditivo.pkl', 'wb') as f:
    pickle.dump(DT, f)