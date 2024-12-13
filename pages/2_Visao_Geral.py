import streamlit as st
import pandas as pd

st.title("Visão Geral do Dataset")

df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

st.header("Primeiros Registros do Dataset")
st.dataframe(df.head())

# Descrição Geral
st.header("Descrição Geral")
st.markdown("""
    Este dataset contém um conjunto de dados mostrando hábitos e comportamentos relacionados ao estilo de vida das pessoas.
""")

# Resumo estatístico
st.header("Resumo estatístico")
# Variáveis numéricas
st.subheader("Variáveis Numéricas")
st.dataframe(df.describe())

# Variáveis categóricas
st.subheader("Variáveis Categóricas")
categorical_stats = df.describe(include=['object'])
st.dataframe(categorical_stats)

# Distribuição de valores ausentes
st.subheader("Valores Ausentes no Dataset")
missing_values = df.isnull().sum()
st.dataframe(missing_values[missing_values > 0])
