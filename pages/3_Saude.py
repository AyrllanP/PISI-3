import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Influencia na saúde fisica ou psicológica
st.title("Impacto na saúde fisica ou psicológica")

st.write(
        """
        Hábitos podem possuir impactos positivos ou negativos na saúde física ou Psicológica.
        Na saúde psicológica geralmente está relacionado a ao bem estar, e a parametros como estresse diário e visão de vida.
        A saúde fisica pode ser influenciada principalmente por hábitos alimentares e de exercícios regulares, além de horas de sono.
        """
    )

# Para Mudança com Parquet
df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')
df.to_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')
df = pd.read_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')

# PARA CORRIGIR STRING DO ESTRESSE DIÁRIO UTILIZANDO MEDIANA
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce') 

st.header("Gráfico de Calor: Escala Normalizada (0 a 1)")
st.write(
        """
        Esse gráfico apresenta a relação entre passos diários, consumo de frutas e vegetais, 
    e metas de vida em uma escala normalizada de 0 a 1. Ele ajuda a identificar padrões 
    e intensidade nas metas de vida baseadas em hábitos de saúde e atividade física.
        """
    )

heatmap_data = df.pivot_table(
    values="LIVE_VISION", 
    index="DAILY_STEPS",  
    columns="FRUITS_VEGGIES",  
    aggfunc="mean" 
)
normalized_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    normalized_data,
    cmap="coolwarm",
    cbar_kws={"label": "Escala Normalizada (0 a 1)"},
    linewidths=0.5,  
    square=True,     
    annot=True,     
    fmt=".2f",       
    ax=ax           
)


ax.set_title("Relação Normalizada: Passos Diários, Consumo de Frutas/Veg e Metas de Vida", fontsize=16)
ax.set_xlabel("Frutas e Vegetais Consumidos Diariamente", fontsize=14)
ax.set_ylabel("Passos Diários", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)


st.pyplot(fig)


# Gráfico de barras relacionando FRUITS_VEGGIES, SLEEP_HOURS, DAILY_STEPS com DAILY_STRESS.

st.header("Relação entre estresse diários e outras variáveis relacionadas a saúde")
st.write(
        """
          Foram escolhidas algumas variáveis relacionadas a saúde para verificação da  sua relação com o  nível de estresse diário.
          As variáveis escolhidas foram FRUITS_VEGGIES , SLEEP_HOURS e DAILY_STEPS
          
          
          Foram observados:
          Níveis de estresse menores em dias com maiores consumo de frutas e vegetais,
          Níveis de estresse menores com uma quantidade de 8 horas de sono, e em quantidades de sono menores um desvio padrão maior em relação a quantidade de sono.
          Não foi observado grandes variações do nível de estresse em relação a quantidade de passos.

        """ 
    )

# Filtro
option = st.selectbox(
    'Escolha a variável para ver a relação com o Estresse diário',
    ['Consumo de frutas e vegetais', 'Horas de sono', 'Atividade/Passos diários']
)
plt.figure(figsize=(7, 4))


if option == 'Consumo de frutas e vegetais':
    fruit_veggies = df['FRUITS_VEGGIES'].value_counts().sort_index()
    sns.barplot(x='FRUITS_VEGGIES', y='DAILY_STRESS', data=df, color='lightblue', edgecolor='black', width=0.8)
    plt.xlabel('Consumo de frutas e vegetais')
    plt.ylabel('Estresse diário')
    plt.title("Consumo de frutas e vegetais vs. Estresse diário")
    plt.xticks(range(len(fruit_veggies)), fruit_veggies.index.astype(str), rotation=0)

elif option == 'Horas de sono':
    sleep = df['SLEEP_HOURS'].value_counts().sort_index()
    sns.barplot(x='SLEEP_HOURS', y='DAILY_STRESS', data=df, color='lightblue', edgecolor='black', width=0.8)
    plt.xlabel('Horas de sono')
    plt.ylabel('Estresse diário')
    plt.title('Horas de sono vs. Estresse diário')
    plt.xticks(range(len(sleep)), sleep.index.astype(str), rotation=0)

else:  # DAILY_STEPS
    steps = df['DAILY_STEPS'].value_counts().sort_index()
    sns.barplot(x='DAILY_STEPS', y='DAILY_STRESS', data=df, color='lightblue', edgecolor='black', width=0.8)
    plt.xlabel('Passos diários (x 1000)')
    plt.ylabel('Estresse diário')
    plt.title('Passos diários vs. Estresse diário')
    plt.xticks(range(len(steps)), steps.index.astype(str), rotation=0)

# Exibir o gráfico no Streamlit
st.pyplot(plt)


