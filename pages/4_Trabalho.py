import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

st.title("Impacto na Produtividade")

st.write(
        """
        Os hábitos podem ter um grande impacto na produtividade e na forma que as pessoas se relacionam no trabalho
        """
    )

# Para Mudança com Parquet
df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')
df.to_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')
df = pd.read_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')

# PARA CORRIGIR STRING DO ESTRESSE DIÁRIO UTILIZAR MEDIANA
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce') 


# Análise de SLEEP_HOURS em relação a outros fatores como DAILY_STEPS e WORK_LIFE_BALANCE_SCORE.

st.header("Como hábitos impactam no equilíbrio entre vida e trabalho")
st.write(
        """
        Este gráfico mostra a relação de DAILY_STEPS (representa atividade) SLEEP_HOURS (representando horas de sono)  e WORK_LIFE_BALANCE_SCORE (representando o equilibrio entre a vida e o trabalho)

        Mais horas de sono e uma quantidade maior de atividade física estão ligados a maiores valores de equilibrio entre a vida e o trabalho

        """
    )

plt.figure(figsize=(10, 6))
plt.scatter(df['DAILY_STEPS'], df['SLEEP_HOURS'], c=df['WORK_LIFE_BALANCE_SCORE'], cmap='viridis')
plt.colorbar(label='WORK_LIFE_BALANCE_SCORE')

plt.title('SLEEP_HOURS vs. DAILY_STEPS, colored by WORK_LIFE_BALANCE_SCORE')
plt.xlabel('DAILY_STEPS')
plt.ylabel('SLEEP_HOURS')

st.pyplot(plt)


# Gráfico de dispersão mostrando DAILY_STRESS em função de SUFFICIENT_INCOME e LIVE_VISION.
st.header("Como estresse e visão de vida são impactados pela Renda")
st.write(
        """
        Quanto a quantidade de pessoas com renda  1 (Insuficiente) é mais frequente ter um maior o nível de estresse diário (3 e 4) e mais baixo a visão de vida (0 até 4)
        Visões de vida Maiores (6-10) e niveis de estresse baixos (0-2) estão menos associados a uma renda suficiente(2) principalmente quando essas duas características estão somadas
        """
    )


plt.figure(figsize=(10, 6))

plt.scatter(df['DAILY_STRESS'], df['LIVE_VISION'], c=df['SUFFICIENT_INCOME'], cmap='viridis')
plt.colorbar(label='SUFFICIENT_INCOME')

plt.title('DAILY_STRESS vs. LIVE_VISION, colored by SUFFICIENT_INCOME')
plt.xlabel('DAILY_STRESS')
plt.ylabel('LIVE_VISION')

st.pyplot(plt)



# Relação entre Meditação e Equilíbrio Vida-Trabalho
st.header("Relação entre Meditação e Equilíbrio Vida-Trabalho")

df["WEEKLY_MEDITATION_JITTER"] = df["WEEKLY_MEDITATION"] + np.random.uniform(-0.5, 0.5, size=len(df))
sampled_df = df.sample(n=190, random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=sampled_df,
    x="WEEKLY_MEDITATION_JITTER",
    y="WORK_LIFE_BALANCE_SCORE",
    alpha=0.7,
    color="purple",
    s=70,
    ax=ax
)

sns.regplot(
    data=df,
    x="WEEKLY_MEDITATION",
    y="WORK_LIFE_BALANCE_SCORE",
    scatter=False,
    color="blue",
    line_kws={"label": "Tendência", "lw": 2},
    ax=ax
)

ax.set_title("Relação entre Meditação Semanal e Equilíbrio Vida-Trabalho (Amostragem)", fontsize=16)
ax.set_xlabel("Horas de Meditação por Semana (com jitter mais amplo)", fontsize=14)
ax.set_ylabel("Pontuação de Equilíbrio Vida-Trabalho", fontsize=14)
ax.legend(fontsize=12)

st.pyplot(fig)



# Distribuição de Tarefas Concluídas por Horas de Sono
st.header("Distribuição de Tarefas Concluídas por Horas de Sono")

df_filtered = df[df["SLEEP_HOURS"] <= 9]

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

ax.set_title("Distribuição de Tarefas Concluídas por Horas de Sono (até 9h)", fontsize=16)
ax.set_xlabel("Horas de Sono", fontsize=14)
ax.set_ylabel("Tarefas Concluídas", fontsize=14)

st.pyplot(fig)


# Heatmap para explorar combinações de hábitos TODO_COMPLETED, WEEKLY_MEDITATION, e TIME_FOR_PASSION.
habits_df = df[["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"]]

habits_df = habits_df.apply(pd.to_numeric, errors="coerce")
correlation_matrix = habits_df.corr()
plt.figure(figsize=(10, 6))  
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlação entre Hábitos")

st.pyplot(plt)

# Filtros
st.sidebar.header("Filtros")
genero = st.sidebar.multiselect("Selecione o Gênero:", options=df["GENDER"].unique(), default=df["GENDER"].unique())
faixa_etaria = st.sidebar.multiselect("Selecione a Faixa Etária:", options=df["AGE"].unique(), default=df["AGE"].unique())

filtered_df = df[(df["GENDER"].isin(genero)) & (df["AGE"].isin(faixa_etaria))]

st.subheader("Análise Interativa dos Hábitos")
habit_choice = st.selectbox("Selecione o Hábito:", ["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"])

fig = px.box(
    filtered_df,
    x="AGE",
    y=habit_choice,
    color="GENDER",
    title=f"Distribuição de '{habit_choice}' por Faixa Etária e Gênero",
    labels={"AGE": "Faixa Etária", habit_choice: "Valores do Hábito", "GENDER": "Gênero"},
    template="plotly",
)

st.plotly_chart(fig)