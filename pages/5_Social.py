import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.title("Impacto de hábitos e comportamentos no bem-estar")

st.write(
        """
        Interações sociais e ações altruistas podem influenciar positivamente no bem estar
        """
    )

# Para Mudança com Parquet
df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')
df.to_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')
df = pd.read_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')

# PARA CORRIGIR STRING DO ESTRESSE DIÁRIO UTILIZAR MEDIANA
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce') 

# Distribuição de SOCIAL_NETWORK e sua relação com DAILY_STRESS.

st.header("Análise de rede de pessoas e estresse diário")

st.write("""
Este gráfico interativo mostra a distribuição de quantidade de pessoas que interage ao dia  a frequencia que a quantidade se repete no dia, com cores diferenciadas
para diferentes valores estresse diário. Use os filtros abaixo para ajustar a análise.
""")

# Filtro para ajustar faixa de SOCIAL_NETWORK
social_range = st.slider(
    'Selecione o intervalo de SOCIAL_NETWORK',
    min_value=int(df['SOCIAL_NETWORK'].min()),
    max_value=int(df['SOCIAL_NETWORK'].max()),
    value=(int(df['SOCIAL_NETWORK'].min()), int(df['SOCIAL_NETWORK'].max()))
)

filtered_df = df[(df['SOCIAL_NETWORK'] >= social_range[0]) & (df['SOCIAL_NETWORK'] <= social_range[1])]

plt.figure(figsize=(10, 6))
sns.histplot(
    x='SOCIAL_NETWORK',
    hue='DAILY_STRESS',
    data=filtered_df,
    multiple='stack',
    palette='viridis'
)
plt.title('Distribuição de quantidade de pessoas com que interage ao dia colorida por estresse diário')
plt.xlabel('Interação com pessoas ao dia')
plt.ylabel('Frequência')

st.pyplot(plt)

# Destaque diferentes faixas de SOCIAL_NETWORK e observe padrões de estresse.

# Impacto de CORE_CIRCLE (pessoas próximas) e SUPPORTING_OTHERS no nível de estresse.

st.header("Relação entre Pessoas próximas, Ajudar os outros e estresse diário")

st.write("""
Este gráfico de dispersão mostra a relação entre quantidade de pessoas próximas e  quantidade de pessoas que ajudou,
com coloração baseada nos valores de Estresse diário.
""")

# plt.figure(figsize=(10, 6))
# scatter_plot = sns.scatterplot(
#     x='CORE_CIRCLE',
#     y='SUPPORTING_OTHERS',
#     hue='DAILY_STRESS',
#     data=df,
#     palette='viridis'
# )

# plt.title('Relação entre quantidade de pessoas proximas,  quantidade de pessoas que ajudou e estresse diário')
# plt.xlabel('Pessoas próximas')
# plt.ylabel('Ajudou pessoas')

# # Ajustar a posição da legenda 
# plt.legend(title='Estresse diário', bbox_to_anchor=(1.05, 1), loc='upper left')

# st.pyplot(plt)

plt.figure(figsize=(12, 6))

# Boxplot para mostrar estresse diário em função de 'CORE_CIRCLE' e 'SUPPORTING_OTHERS'
sns.boxplot(
    x='CORE_CIRCLE', 
    y='SUPPORTING_OTHERS',
    hue='DAILY_STRESS',  
    data=df,
    palette='viridis'
)

plt.title('Relação entre quantidade de pessoas proximas,  quantidade de pessoas que ajudou e estresse diário')
plt.xlabel('Pessoas Próximas')
plt.ylabel('Quantidade de pessoas que ajudou')

plt.tight_layout()
# Ajustar a posição da legenda 
plt.legend(title='Estresse diário', bbox_to_anchor=(1.05, 1), loc='upper left')


st.pyplot(plt)
