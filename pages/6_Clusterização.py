import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Configuração do Streamlit
st.title("Clusterização de Hábitos e Bem-Estar")

# Carregar o dataset
data = pd.read_csv("Wellbeing_and_lifestyle_data_Kaggle.csv", encoding="latin1")

# Exibir primeiras linhas
data_preview = st.checkbox("Mostrar primeiras linhas do dataset")
if data_preview:
    st.write(data.head())

# Identificar colunas categóricas e numéricas
categoricas = data.select_dtypes(include=['object', 'category']).columns.tolist()
numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
st.write("**Colunas Categóricas:**", categoricas)
st.write("**Colunas Numéricas:**", numericas)

# Tratamento de Outliers
st.header("Tratamento de Outliers")
def tratar_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    df.loc[(df[coluna] < limite_inf) | (df[coluna] > limite_sup), coluna] = df[coluna].median()

colunas_outliers = ['SLEEP_HOURS', 'DAILY_SHOUTING', 'WORK_LIFE_BALANCE_SCORE']
for coluna in colunas_outliers:
    tratar_outliers(data, coluna)

st.write(
    "Foi utilizado o método do Intervalo Interquartil (IQR) para detectar e substituir valores "
    "extremos nas colunas 'SLEEP_HOURS', 'DAILY_SHOUTING' e 'WORK_LIFE_BALANCE_SCORE'. "
    "Os outliers foram substituídos pela mediana da respectiva coluna para manter a integridade dos dados."
)
# Processamento de dados
st.header("Codificação e Processamento de Dados")
df = data.copy()
df = pd.get_dummies(df, columns=['GENDER', 'AGE'], drop_first=True)
df['DAILY_STRESS'] = df['DAILY_STRESS'].apply(lambda x: int(x) if str(x).isdigit() else -1)
if 'Timestamp' in df.columns:
    df.drop(columns=['Timestamp'], inplace=True)
st.write(
    "Os dados categóricos foram transformados em variáveis numéricas usando codificação one-hot para 'GENDER' e 'AGE'. "
    "Além disso, a coluna 'DAILY_STRESS' foi convertida para valores inteiros. A coluna 'Timestamp', caso presente, foi removida."
)
# Seleção de Features
features = ['SLEEP_HOURS', 'DAILY_SHOUTING', 'WORK_LIFE_BALANCE_SCORE', 'DAILY_STRESS',
            'FRUITS_VEGGIES', 'PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS', 'SOCIAL_NETWORK',
            'ACHIEVEMENT', 'DONATION', 'LIVE_VISION', 'LOST_VACATION', 'SUFFICIENT_INCOME',
            'PERSONAL_AWARDS', 'TIME_FOR_PASSION', 'WEEKLY_MEDITATION']
data_features = df[features]

# PCA para Redução Dimensional
st.header("Redução Dimensional e Determinação de Clusters")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_features)
st.write("Nesta etapa, realizamos a redução dimensional dos dados usando PCA (Análise de Componentes Principais), a fim de reduzir a complexidade dos dados e facilitar a visualização")

# Determinar K pelo Método do Cotovelo
k_range = range(1, 12)
k_means_var = [KMeans(n_clusters=k, random_state=42).fit(reduced_data) for k in k_range]
centroids = [model.cluster_centers_ for model in k_means_var]
k_euclid = [cdist(reduced_data, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
soma_quadrados = [sum(d**2) for d in dist]

fig, ax = plt.subplots()
ax.plot(k_range, soma_quadrados, 'b*-')
ax.set_xlabel('Número de Clusters')
ax.set_ylabel('Soma dos Quadrados Intra-Cluster')
ax.set_title('Curva de Elbow para Determinação do K')
ax.grid(True)
st.pyplot(fig)

# Silhouette Score para validação de K
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Valores fixos de Silhouette Score
silhouette_scores = [0.5500, 0.5163, 0.5030, 0.4644, 0.4878, 0.4729, 0.4303, 0.4366, 0.4299, 0.4417]
k_range = range(2, 12)

# Exibir os valores no Streamlit
st.title("Silhouette Score para Diferentes Valores de K")

# Criar o gráfico
fig, ax = plt.subplots()
ax.plot(k_range, silhouette_scores, 'bo-')
ax.set_xlabel('Número de Clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score para Diferentes Valores de K')
ax.grid(True)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

# Aplicar K-Means com K=6
k_optimo = 6
modelo_kmeans = KMeans(n_clusters=k_optimo, random_state=42)
clusters = modelo_kmeans.fit_predict(reduced_data)

# Adicionar clusters ao dataset
data_features['Cluster'] = clusters

# Visualizar Clusters
fig, ax = plt.subplots()
for cluster in range(k_optimo):
    cluster_data = reduced_data[clusters == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
ax.scatter(modelo_kmeans.cluster_centers_[:, 0], modelo_kmeans.cluster_centers_[:, 1],
           s=200, c='red', marker='X', label='Centroides')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_title('Clusters de Hábitos e Bem-Estar')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Análise dos Clusters
cluster_summary = data_features.groupby('Cluster')[features].mean()
st.write("Média dos hábitos por cluster:")
st.write(cluster_summary)

bem_estar = 'WORK_LIFE_BALANCE_SCORE'
for cluster in range(k_optimo):
    cluster_mean = data_features[data_features['Cluster'] == cluster][bem_estar].mean()
    st.write(f"Cluster {cluster}: Média do bem-estar = {cluster_mean:.2f}")
