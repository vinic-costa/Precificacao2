from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns


def preprocess_data(df):
    # Lista das colunas no dataframe df que não são nem numéricas nem datas
    colunas_nao_numericas_nem_datas = []

    # Percorrer as colunas do dataframe
    for coluna in df.columns:
        if not pd.api.types.is_numeric_dtype(df[coluna]) and not pd.api.types.is_datetime64_any_dtype(df[coluna]):
            colunas_nao_numericas_nem_datas.append(coluna)
    
    # Conversão no dataframe df
    # Converter todas as colunas salvas em colunas_nao_numericas_nem_datas para minúsculas e remover acentos
    for coluna in colunas_nao_numericas_nem_datas:
        df[coluna] = df[coluna].apply(lambda x: unidecode(str(x).lower()) if pd.notnull(x) else np.nan)
    
    #Dropando colunas não necessárias para o modelo
    colunas_para_dropar = ['valor da venda','quantidade', 'datapublicacao', 'datavenda', 'material_secundario', 'Material não padronizado', 'SKU']
    df.drop(colunas_para_dropar, axis=1, inplace=True)

    # Eliminar duplicatas
    df.drop_duplicates(inplace=True)

    # Eliminar linhas com valores nulos
    df.dropna(inplace=True)

    return df

# Carregar dados
df = pd.read_excel('DNC Lista 5.xlsx')

df.info()
print(df.describe())

# Verifique se as colunas contêm apenas valores numéricos antes de converter
cols_to_convert = ['altura', 'largura', 'profundidade']

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tratamento de valores ausentes nas colunas numéricas
numeric_cols = ['altura', 'largura', 'profundidade']
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print(df.describe())

# Pré-processamento dos dados
print("antes do preprocessamento")
df.info()
df = preprocess_data(df)
df.info()

# Valor individual


# Define o tamanho da figura
plt.figure(figsize=(12, 6))

# Define o Titulo
plt.title("valorindividual", size=18)

# Plot do Histograma
sns.histplot(df["valorindividual"], kde=True, alpha=0.2)

# plotando média
plt.axvline(x=df["valorindividual"].mean(), color="red", label="média")

# plotando a mediana
plt.axvline(df["valorindividual"].median(), color="green", label="mediana")

# Loop para plotar as modas
for i in range(df["valorindividual"].mode().shape[0]):
    plt.axvline(df["valorindividual"].mode()[i], color="yellow", label="moda")

plt.axvline(df["valorindividual"].quantile(0.25), color="orange", label="q1")
plt.axvline(df["valorindividual"].quantile(0.75), color="pink", label="q3")

# Cria uma legenda
plt.legend()

# Mostra o Gráfico
plt.show()

df=df[df['valorindividual'] < df['valorindividual'].quantile(.96)]
df=df[df['valormercado'] < df['valormercado'].quantile(.96)]
df=df[df['altura'] < df['altura'].quantile(.95)]
df=df[df['largura'] < df['largura'].quantile(.95)]
df=df[df['profundidade'] < df['profundidade'].quantile(.95)]

print('printando df')
df.info()

# Valor individual


# Define o tamanho da figura
plt.figure(figsize=(12, 6))

# Define o Titulo
plt.title("valorindividual", size=18)

# Plot do Histograma
sns.histplot(df["valorindividual"], kde=True, alpha=0.2)

# plotando média
plt.axvline(x=df["valorindividual"].mean(), color="red", label="média")

# plotando a mediana
plt.axvline(df["valorindividual"].median(), color="green", label="mediana")

# Loop para plotar as modas
for i in range(df["valorindividual"].mode().shape[0]):
    plt.axvline(df["valorindividual"].mode()[i], color="yellow", label="moda")

plt.axvline(df["valorindividual"].quantile(0.25), color="orange", label="q1")
plt.axvline(df["valorindividual"].quantile(0.75), color="pink", label="q3")

# Cria uma legenda
plt.legend()

# Mostra o Gráfico
plt.show()
# Supondo que o seu DataFrame seja chamado 'df'
total_linhas = len(df)
linhas_com_valor_individual_500 = len(df[df['valorindividual'] <= 500])
porcentagem = (linhas_com_valor_individual_500 / total_linhas) * 100

print(f"A porcentagem de linhas com 'valorindividual' menor ou igual a 500 é: {porcentagem:.2f}%")


# Definir features (X) e target (y)
X = df[['valormercado', 'altura', 'largura', 'profundidade']]
y = df['valorindividual']

# Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um pipeline com etapas de pré-processamento e modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Modelo de regressão
])

# Treinar o modelo usando o pipeline
pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print(f'R-squared: {r2}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root mean Squared Error: {rmse}')

# Salvar o modelo treinado usando joblib
joblib.dump(pipeline, 'modelo_precificacao2_outl.joblib')
