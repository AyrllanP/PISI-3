import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
Este gráfico interativo mostra a distribuição de `SOCIAL_NETWORK`  com cores diferenciadas
para diferentes valores de `DAILY_STRESS`. Use os filtros abaixo para ajustar a análise.
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
plt.title('Distribuição de SOCIAL_NETWORK colorida por DAILY_STRESS')
plt.xlabel('SOCIAL_NETWORK')
plt.ylabel('Frequência')

st.pyplot(plt)

# Impacto de CORE_CIRCLE (pessoas próximas) e SUPPORTING_OTHERS no nível de estresse.

st.header("Relação entre Pessoas próximas, Ajudar os outros e estresse diário")

st.write("""
Este gráfico de dispersão mostra a relação entre `CORE_CIRCLE` e `SUPPORTING_OTHERS`,
com coloração baseada nos valores de `DAILY_STRESS`.
""")

plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(
    x='CORE_CIRCLE',
    y='SUPPORTING_OTHERS',
    hue='DAILY_STRESS',
    data=df,
    palette='viridis'
)

plt.title('Relação entre CORE_CIRCLE, SUPPORTING_OTHERS e DAILY_STRESS')
plt.xlabel('CORE_CIRCLE')
plt.ylabel('SUPPORTING_OTHERS')

# Ajustar a posição da legenda 
plt.legend(title='DAILY_STRESS', bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(plt)
