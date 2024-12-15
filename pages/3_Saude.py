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

st.header("Gráfico de Calor: Hábitos de Vida")
st.write(
        """
        Esse gráfico relaciona os valores de visão de vida com hábitos de DAILY_STEPS que refere-se a nível de atividade e FRUITS_VEGGIES que refere-se a alimentação saudável
        """
    )

heatmap_data = df.pivot_table(
    values="LIVE_VISION", 
    index="DAILY_STEPS",  
    columns="FRUITS_VEGGIES",  
    aggfunc="mean" 
)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    cbar_kws={"label": "Intensidade das Metas de Vida"},
    linewidths=0.5, 
    square=True,     
    annot=False,     
    ax=ax          
)

ax.set_title("Relação entre Passos Diários, Consumo de Frutas/Veg e Metas de Vida", fontsize=16)
ax.set_xlabel("Frutas e Vegetais Consumidos Diariamente", fontsize=14)
ax.set_ylabel("Passos Diários", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

st.pyplot(fig)


# Gráfico de barras relacionando FRUITS_VEGGIES, SLEEP_HOURS, DAILY_STEPS com DAILY_STRESS.

st.header("Relação entre estresse diários e outras variáveis relacionadas a saúde")
# Filtro
option = st.selectbox(
    'Escolha a variável para ver a relação com o Daily Stress',
    ['Consumo de frutas e vegetais', 'Horas de sono', 'Atividade/Passos diários']
)
plt.figure(figsize=(10, 6))


if option == 'Consumo de frutas e vegetais':
    fruit_veggies = df['FRUITS_VEGGIES'].value_counts().sort_index()
    sns.barplot(x='FRUITS_VEGGIES', y='DAILY_STRESS', data=df)
    plt.xlabel('Fruits and Veggies Consumption')
    plt.ylabel('Daily Stress')
    plt.title("FRUITS_VEGGIES vs. DAILY_STRESS")
    plt.xticks(range(len(fruit_veggies)), fruit_veggies.index.astype(str), rotation=0)

elif option == 'Horas de sono':
    sleep = df['SLEEP_HOURS'].value_counts().sort_index()
    sns.barplot(x='SLEEP_HOURS', y='DAILY_STRESS', data=df)
    plt.xlabel('Sleep Hours')
    plt.ylabel('Daily Stress')
    plt.title('SLEEP_HOURS vs. DAILY_STRESS')
    plt.xticks(range(len(sleep)), sleep.index.astype(str), rotation=0)

else:  # DAILY_STEPS
    steps = df['DAILY_STEPS'].value_counts().sort_index()
    sns.barplot(x='DAILY_STEPS', y='DAILY_STRESS', data=df)
    plt.xlabel('Daily Steps')
    plt.ylabel('Daily Stress')
    plt.title('DAILY_STEPS vs. DAILY_STRESS')
    plt.xticks(range(len(steps)), steps.index.astype(str), rotation=0)

# Exibir o gráfico no Streamlit
st.pyplot(plt)


