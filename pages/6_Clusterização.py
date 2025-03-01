import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Função para carregar o dataset
@st.cache
def load_data():
    data = pd.read_csv("Wellbeing_and_lifestyle_data_Kaggle.csv", encoding="latin1")
    return data

# Função para tratar outliers
def tratar_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    st.write(f"Coluna: {coluna} - Outliers detectados: {len(outliers)}")
    mediana = df[coluna].median()
    df.loc[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior), coluna] = mediana

# Função para codificar a coluna 'DAILY_STRESS' corretamente
def encode_daily_stress(value):
    try:
        value = str(value).strip()
        mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        return mapping.get(value, np.nan)
    except Exception:
        return np.nan

# Carregar o dataset
data = load_data()

# Verificação de valores nulos
st.header("Verificação de Valores Nulos")
if data.isnull().sum().sum() == 0:
    st.write("Este dataset não contém valores nulos.")
else:
    st.write("Este dataset contém valores nulos.")
st.write(data.isnull().sum())

# Quantidade de colunas e registros
st.header("Quantidade de Colunas e Registros")
st.write(f"O dataset possui {data.shape[0]} registros e {data.shape[1]} colunas.")

# Identificar colunas categóricas e numéricas
categoricas = data.select_dtypes(include=['object', 'category']).columns.tolist()
numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

st.header("Colunas Categóricas e Numéricas")
st.write(f"**Colunas Categóricas:** {categoricas}")
st.write(f"**Colunas Numéricas:** {numericas}")

# Tratamento de outliers
st.header("Tratamento de Outliers")
colunas_com_outliers = ['SLEEP_HOURS', 'DAILY_SHOUTING', 'WORK_LIFE_BALANCE_SCORE']
for coluna in colunas_com_outliers:
    tratar_outliers(data, coluna)
st.write("Limpeza de dados concluída. O dataset tratado está armazenado na variável 'data'.")

# Codificação de variáveis categóricas
df = data.copy()
df = pd.get_dummies(df, columns=['GENDER', 'AGE'], drop_first=True)

df['DAILY_STRESS'] = df['DAILY_STRESS'].apply(encode_daily_stress)
df['DAILY_STRESS'].fillna(df['DAILY_STRESS'].median(), inplace=True)
df['DAILY_STRESS'] = df['DAILY_STRESS'].astype(int)

if 'Timestamp' in df.columns:
    df.drop(columns=['Timestamp'], inplace=True)

# Seleção de features para clustering
features = ['SLEEP_HOURS', 'DAILY_SHOUTING', 'WORK_LIFE_BALANCE_SCORE', 'DAILY_STRESS',
            'FRUITS_VEGGIES', 'PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS',
            'SOCIAL_NETWORK', 'ACHIEVEMENT', 'DONATION', 'LIVE_VISION', 'LOST_VACATION',
            'SUFFICIENT_INCOME', 'PERSONAL_AWARDS', 'TIME_FOR_PASSION', 'WEEKLY_MEDITATION']
data_features = df[features]

# Adicionar filtros interativos
st.sidebar.header("Filtros de Hábitos")
habitos_selecionados = st.sidebar.multiselect("Selecione os hábitos para análise", features, default=features)

data_filtrada = df[habitos_selecionados]

# Redução do dataset com base no tamanho da amostra
sample_size = st.sidebar.slider("Tamanho da Amostra", min_value=0.01, max_value=1.0, value=0.1)
sample_data, _ = train_test_split(data_filtrada, train_size=sample_size, random_state=42)

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(sample_data)

# Aplicar KMeans
num_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=6)
modelo_kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = modelo_kmeans.fit_predict(reduced_data)
sample_data['Cluster'] = clusters

# Visualização dos clusters com Plotly
st.header("Visualização Interativa dos Clusters")
fig = px.scatter(
    reduced_data, 
    x=0, y=1, 
    color=sample_data['Cluster'].astype(str),
    title="Clusters de Hábitos e Bem-Estar",
    labels={'0': 'Componente Principal 1', '1': 'Componente Principal 2'},
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig.add_scatter(
    x=modelo_kmeans.cluster_centers_[:, 0], 
    y=modelo_kmeans.cluster_centers_[:, 1], 
    mode='markers', 
    marker=dict(size=12, color='red', symbol='x', line=dict(width=2, color='black')),
    name="Centroides"
)
st.plotly_chart(fig)

# Resumo dos clusters
st.header("Resumo dos Clusters")
cluster_summary = sample_data.groupby('Cluster').mean()
st.write("Média dos hábitos por cluster:")
st.dataframe(cluster_summary)

# Visualização das distribuições dos hábitos
st.header("Distribuição dos Hábitos Selecionados")
for habit in habitos_selecionados:
    fig = px.histogram(
        df, 
        x=habit, 
        title=f'Distribuição de {habit}',
        labels={habit: habit}
    )
    st.plotly_chart(fig)
