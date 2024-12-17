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
        Este gráfico mostra a relação de atividade, quantidade de horas de sono e equilíbrio entre a vida e o trabalho

        Mais horas de sono e uma quantidade maior de atividade física estão ligados a maiores valores de equilibrio entre a vida e o trabalho

        """
    )

plt.figure(figsize=(10, 6))
plt.scatter(df['DAILY_STEPS'], df['SLEEP_HOURS'], c=df['WORK_LIFE_BALANCE_SCORE'], cmap='viridis')
plt.colorbar(label='Equilíbrio vida-trabalho')

plt.title('Horas de sono e Atividade física relacionado a Equilíbrio vida-trabalho')
plt.xlabel('Atividade física')
plt.ylabel('Horas de sono')

st.pyplot(plt)

# Relacionar estilo de vida e estresse

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
plt.colorbar(label='Suficiência da renda')

plt.title('Estresse diário e Visão de vida  relacionado a suficiência da renda')
plt.xlabel('Estresse diário')
plt.ylabel('Visão de vida')

st.pyplot(plt)

# impactos econômicos e visão de propósito


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

# Combine com a distribuição por gênero/faixa etária em abas separadas ou filtros para clareza.

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

st.header("Heatmap de tarefas completadas, meditação semanal e tempo por paixão")
# Heatmap para explorar combinações de hábitos TODO_COMPLETED, WEEKLY_MEDITATION, e TIME_FOR_PASSION.
habits_df = df[["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"]]

habits_df = habits_df.apply(pd.to_numeric, errors="coerce")
correlation_matrix = habits_df.corr()

translations = {
    "TODO_COMPLETED": "Tarefas Completadas",
    "WEEKLY_MEDITATION": "Meditação Semanal",
    "TIME_FOR_PASSION": "Tempo para Paixão"
}

plt.figure(figsize=(10, 6))  
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

# Traduzir  eixos
plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=[translations[col] for col in correlation_matrix.columns], rotation=45)
plt.yticks(ticks=range(len(correlation_matrix.index)), labels=[translations[index] for index in correlation_matrix.index], rotation=0)


plt.title("Correlação entre Hábitos")

st.pyplot(plt)

# Mapeamento das faixas etárias
faixa_etaria_traduzida = {
    "Less than 20": "Menos de 20",
    "21 to 35": "21 a 35",
    "36 to 50": "36 a 50",
    "51 or more": "51 ou mais"
}

habit_translation = {
    "TODO_COMPLETED": "Tarefas Completadas",
    "WEEKLY_MEDITATION": "Meditação Semanal",
    "TIME_FOR_PASSION": "Tempo para Paixão"
}

gender_translation = {
    "Male": "Masculino",
    "Female": "Feminino"
}

# Traduzir as faixas etárias e os gêneros
df["AGE_TRANSLATED"] = df["AGE"].map(faixa_etaria_traduzida)
df["GENDER_TRANSLATED"] = df["GENDER"].map(gender_translation)

# Ordenar as faixas etárias
df["AGE_TRANSLATED"] = pd.Categorical(df["AGE_TRANSLATED"], categories=["Menos de 20", "21 a 35", "36 a 50", "51 ou mais"], ordered=True)

# Filtros
st.sidebar.header("Filtros")
genero = st.sidebar.multiselect("Selecione o Gênero:", options=df["GENDER_TRANSLATED"].unique(), default=df["GENDER_TRANSLATED"].unique())
faixa_etaria = st.sidebar.multiselect("Selecione a Faixa Etária:", options=df["AGE_TRANSLATED"].unique(), default=df["AGE_TRANSLATED"].unique())

filtered_df = df[(df["GENDER_TRANSLATED"].isin(genero)) & (df["AGE_TRANSLATED"].isin(faixa_etaria))]

# Análise Interativa dos Hábitos
st.subheader("Análise Interativa dos Hábitos")
habit_choice = st.selectbox("Selecione o Hábito:", options=["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"])

# Traduzir o hábito selecionado
habit_choice_translated = habit_translation[habit_choice]

# Plotando o gráfico com as faixas etárias e gêneros traduzidos
fig = px.box(
    filtered_df,
    x="AGE_TRANSLATED",
    y=habit_choice,
    color="GENDER_TRANSLATED",
    title=f"Distribuição de {habit_choice_translated} por Faixa Etária e Gênero",
    labels={
        "AGE_TRANSLATED": "Faixa Etária", 
        habit_choice: "Valores do Hábito", 
        "GENDER_TRANSLATED": "Gênero"
    },
    template="plotly",
)

st.plotly_chart(fig)