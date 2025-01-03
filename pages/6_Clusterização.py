import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Carregar os dados (substituir pelo seu dataset)
data = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

# Função para identificar outliers usando o IQR
def detectar_outliers_iqr(df):
    outliers = {}
    for coluna in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        outliers[coluna] = df[(df[coluna] < Q1 - 1.5 * IQR) | (df[coluna] > Q3 + 1.5 * IQR)][coluna]
    return outliers

# Streamlit App
st.title("Clusterização - Análise e Tratamento de Dados")

# Introdução
st.header("Introdução")
st.write("Nesta seção, mostramos a preparação inicial dos dados para a clusterização.")

# Verificar valores nulos
st.subheader("Verificação de Valores Nulos")
st.write("Este dataset não contém valores nulos.")
st.write(data.isnull().sum())

# Quantidade de colunas e registros
st.subheader("Quantidade de Colunas e Registros")
st.write(f"O dataset possui {data.shape[0]} registros e {data.shape[1]} colunas.")

# Colunas categóricas e numéricas
st.subheader("Colunas Categóricas e Numéricas")
categoricas = data.select_dtypes(include=['object', 'category']).columns.tolist()
numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
st.write("**Colunas Categóricas:**", categoricas)
st.write("**Colunas Numéricas:**", numericas)

# Outliers por coluna
st.subheader("Detecção de Outliers")
outliers = detectar_outliers_iqr(data)
for coluna, valores in outliers.items():
    st.write(f"Outliers na coluna {coluna}:", valores if not valores.empty else "Nenhum")

# Tratamento dos dados
st.header("Tratamento dos Dados")

# Remover colunas irrelevantes
st.subheader("Remoção de Colunas Irrelevantes")
colunas_remover = ['carimbo_de_data_hora']
data = data.drop(columns=colunas_remover, errors='ignore')
st.write("Colunas removidas: ", colunas_remover)

# Codificar dados categóricos
st.subheader("Codificação de Dados Categóricos")
# OneHotEncoding para GÊNERO e AGE
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
categoricas_para_codificar = ['GENERO', 'AGE']
onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(data[categoricas_para_codificar]),
                              columns=onehot_encoder.get_feature_names_out(categoricas_para_codificar))
data = data.drop(columns=categoricas_para_codificar)
data = pd.concat([data, onehot_encoded], axis=1)
st.write("OneHot Encoding aplicado às colunas: ", categoricas_para_codificar)

# Label Encoding para Daily_Stress
label_encoder = LabelEncoder()
data['Daily_Stress'] = label_encoder.fit_transform(data['Daily_Stress'])
st.write("Label Encoding aplicado na coluna: Daily_Stress")

# Padronizar dados numéricos
st.subheader("Padronização de Dados Numéricos")
scaler = StandardScaler()
data[numericas] = scaler.fit_transform(data[numericas])
st.write("Padronização aplicada às colunas numéricas.")

# Remover outliers
st.subheader("Remoção de Outliers")
data_sem_outliers = data.copy()
for coluna in numericas:
    Q1 = data[coluna].quantile(0.25)
    Q3 = data[coluna].quantile(0.75)
    IQR = Q3 - Q1
    data_sem_outliers = data_sem_outliers[(data_sem_outliers[coluna] >= Q1 - 1.5 * IQR) &
                                          (data_sem_outliers[coluna] <= Q3 + 1.5 * IQR)]
st.write("Outliers removidos com base no IQR.")
st.write(data_sem_outliers.head())

# Salvar o dataset tratado
data_sem_outliers.to_csv('dataset_tratado_clusterizacao.csv', index=False)
st.write("Dataset tratado salvo como 'dataset_tratado_clusterizacao.csv'.")
