import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px


st.title("Impacto de hábitos e comportamentos no bem-estar")

df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  # substituindo a string de DAILY_STRESS por 3 por que é a média dos valores 
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce') # transformando em números

# Como hábitos e comportamentos impactam o bem-estar

# Gráficos de Ayrllan

st.header("Gráfico de Calor: Metas de Vida")

heatmap_data = df.pivot_table(
    values="LIVE_VISION",  # Metas de vida como valores
    index="DAILY_STEPS",   # Passos diários no eixo Y
    columns="FRUITS_VEGGIES",  # Frutas e vegetais no eixo X
    aggfunc="mean"  # Cálculo da média
)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    cbar_kws={"label": "Intensidade das Metas de Vida"},
    linewidths=0.5,  # Separação entre células
    square=True,     # Deixa as células quadradas
    annot=False,     # Remove os valores numéricos
    ax=ax            # Passa o eixo para a função de plot
)
# Ajustando título e rótulos
ax.set_title("Relação entre Passos Diários, Consumo de Frutas/Veg e Metas de Vida", fontsize=16)
ax.set_xlabel("Frutas e Vegetais Consumidos Diariamente", fontsize=14)
ax.set_ylabel("Passos Diários", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Exibindo o gráfico
st.pyplot(fig)


st.header("Relação entre Meditação e Equilíbrio Vida-Trabalho")


# Criando jitter maior nos dados para melhorar visualização
df["WEEKLY_MEDITATION_JITTER"] = df["WEEKLY_MEDITATION"] + np.random.uniform(-0.5, 0.5, size=len(df))

# Reduzindo o número de pontos com amostragem aleatória
sampled_df = df.sample(n=190, random_state=42)

# Criando a figura para o gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Gráfico de dispersão com jitter
sns.scatterplot(
    data=sampled_df,
    x="WEEKLY_MEDITATION_JITTER",
    y="WORK_LIFE_BALANCE_SCORE",
    alpha=0.7,
    color="purple",
    s=70,
    ax=ax
)

# Adicionando uma linha de tendência baseada em toda a base de dados
sns.regplot(
    data=df,
    x="WEEKLY_MEDITATION",
    y="WORK_LIFE_BALANCE_SCORE",
    scatter=False,
    color="blue",
    line_kws={"label": "Tendência", "lw": 2},
    ax=ax
)

# Ajustando título e rótulos
ax.set_title("Relação entre Meditação Semanal e Equilíbrio Vida-Trabalho (Amostragem)", fontsize=16)
ax.set_xlabel("Horas de Meditação por Semana (com jitter mais amplo)", fontsize=14)
ax.set_ylabel("Pontuação de Equilíbrio Vida-Trabalho", fontsize=14)
ax.legend(fontsize=12)

# Exibindo o gráfico no Streamlit
st.pyplot(fig)

st.header("Distribuição de Tarefas Concluídas por Horas de Sono")
# Filtrar dados para considerar apenas até 9 horas de sono
df_filtered = df[df["SLEEP_HOURS"] <= 9]

# Criando o gráfico de boxplot com uma figura explícita
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(
data=df_filtered,
x="SLEEP_HOURS",
y="TODO_COMPLETED",
palette="RdYlGn",
width=0.5,
saturation=0.55,
ax=ax
)

# Ajustando título e rótulos
ax.set_title("Distribuição de Tarefas Concluídas por Horas de Sono (até 9h)", fontsize=16)
ax.set_xlabel("Horas de Sono", fontsize=14)
ax.set_ylabel("Tarefas Concluídas", fontsize=14)

# Exibindo o gráfico no Streamlit
st.pyplot(fig)


# Barras ou scatter plots relacionando FRUITS_VEGGIES, SLEEP_HOURS, DAILY_STEPS com DAILY_STRESS.

plt.figure(figsize=(10, 6))

fruit_veggies = df['FRUITS_VEGGIES'].value_counts().sort_index()

fruit_veggies.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)

sns.barplot(x='FRUITS_VEGGIES', y='DAILY_STRESS', data=df)
plt.xlabel('Fruits and Veggies Consumption')
plt.ylabel('Daily Stress')
plt.title("FRUITS_VEGGIES vs. DAILY_STRESS")
ax.set_xticks(range(len(fruit_veggies)))  # Define os ticks no eixo x
ax.set_xticklabels(fruit_veggies.index.astype(int), rotation=0)
st.pyplot(plt)


sleep = df['SLEEP_HOURS'].value_counts().sort_index()

sleep.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)
sns.barplot(x='SLEEP_HOURS', y='DAILY_STRESS', data=df)
plt.xlabel('Sleep Hours')
plt.ylabel('Daily Stress')
plt.title('SLEEP_HOURS vs. DAILY_STRESS')
ax.set_xticks(range(len(sleep)))  # Define os ticks no eixo x
ax.set_xticklabels(sleep.index.astype(int), rotation=0)
st.pyplot(plt)

steps = df['SLEEP_HOURS'].value_counts().sort_index()
steps.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)
sns.barplot(x='DAILY_STEPS', y='DAILY_STRESS', data=df)
plt.xlabel('Daily Steps')
plt.ylabel('Daily Stress')
plt.title('DAILY_STEPS vs. DAILY_STRESS')
ax.set_xticks(range(len(steps)))  # Define os ticks no eixo x
ax.set_xticklabels(steps.index.astype(int), rotation=0)
st.pyplot(plt)

# Heatmap para explorar combinações de hábitos TODO_COMPLETED, WEEKLY_MEDITATION, e TIME_FOR_PASSION.
habits_df = df[["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"]]
    
# Converter as colunas para numéricas (caso necessário)
habits_df = habits_df.apply(pd.to_numeric, errors="coerce")

# Calcular a matriz de correlação
correlation_matrix = habits_df.corr()

# Criar o heatmap
plt.figure(figsize=(8, 6))  # Tamanho do gráfico
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlação entre Hábitos")

# Exibir o gráfico no Streamlit
st.pyplot(plt)


# Comparação de médias de variáveis de hábitos entre diferentes faixas AGE ou por GENDER.



# Filtros interativos
st.sidebar.header("Filtros")
genero = st.sidebar.multiselect("Selecione o Gênero:", options=df["GENDER"].unique(), default=df["GENDER"].unique())
faixa_etaria = st.sidebar.multiselect("Selecione a Faixa Etária:", options=df["AGE"].unique(), default=df["AGE"].unique())

# Aplicar filtros
filtered_df = df[(df["GENDER"].isin(genero)) & (df["AGE"].isin(faixa_etaria))]

# Gráfico interativo com Plotly
st.subheader("Análise Interativa dos Hábitos")
habit_choice = st.selectbox("Selecione o Hábito:", ["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"])

# Criar gráfico interativo
fig = px.box(
    filtered_df,
    x="AGE",
    y=habit_choice,
    color="GENDER",
    title=f"Distribuição de '{habit_choice}' por Faixa Etária e Gênero",
    labels={"AGE": "Faixa Etária", habit_choice: "Valores do Hábito", "GENDER": "Gênero"},
    template="plotly",
)

# Exibir o gráfico
st.plotly_chart(fig)
