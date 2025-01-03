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

st.header("Relação entre passos diários, horas de sono e equilíbrio vida-trabalho")
st.write(
        """
        Este gráfico mostra a relação de atividade, quantidade de horas de sono e equilíbrio entre a vida e o trabalho, é possivel verificar que o equlíbrio de vida e trabalho está com valores maiores quando relacionado a uma quantidade maior de passos diários e  de horas de sono.

        """
    )



# plt.figure(figsize=(10, 6))

# plt.scatter(df['DAILY_STEPS'], df['SLEEP_HOURS'], 
#             c=df['WORK_LIFE_BALANCE_SCORE'], 
#             cmap='viridis', 
#             s=50, alpha=0.8)

# plt.colorbar(label='Equilíbrio Vida-Trabalho (Baixo → Alto)')

# plt.title('Horas de sono e Atividade física relacionado a Equilíbrio vida-trabalho')
# plt.xlabel('Passos diários (x1000)')
# plt.ylabel('Horas de sono por noite')


# st.pyplot(plt)


heatmap_data = df.pivot_table(index='SLEEP_HOURS', 
                              columns='DAILY_STEPS', 
                              values='WORK_LIFE_BALANCE_SCORE', 
                              aggfunc='mean')

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Equilíbrio Vida-Trabalho (Baixo → Alto)'})
plt.title('Heatmap: Atividade Física x Horas de Sono')
plt.xlabel('Passos Diários (x1000)')
plt.ylabel('Horas de Sono por Noite')

st.pyplot(plt)

# plt.figure(figsize=(10, 6))
# sns.kdeplot(x=df['DAILY_STEPS'], y=df['SLEEP_HOURS'], cmap='viridis', fill=True)
# plt.title('Distribuição de Passos Diários e Horas de Sono')
# plt.xlabel('Passos Diários')
# plt.ylabel('Horas de Sono por Noite')
# st.pyplot(plt)


# Relacionar estilo de vida e estresse

# Gráfico de dispersão mostrando DAILY_STRESS em função de SUFFICIENT_INCOME e LIVE_VISION.
st.header("Como estresse e visão de vida são impactados pela Renda")
st.write(
        """
A maior parte das pessoas com renda Insuficiente (1) possuem uma visão de vida projetada para periodos curtos (1 a 4 anos).
As pessoas com renda Suficiente (2) especialmente no menor nível de estresse possuem uma visão de vida mais ampla (até 10 anos, com mediana em 5 anos), em niveis maiores estresse moderados a alto (2 à 5) a visão de vida possui no 3º quartil 5 anos

Dessa forma é possivel verificar que niveis de estresse mais altos estão relacionadas a visão de vida e uma dificuldade maior em planejamento de longo prazo.
        """
    )


# plt.figure(figsize=(10, 6))

# plt.scatter(df['DAILY_STRESS'], df['LIVE_VISION'], c=df['SUFFICIENT_INCOME'], cmap='viridis')
# plt.colorbar(label='Suficiência da renda')

# plt.title('Estresse diário e Visão de vida  relacionado a suficiência da renda')
# plt.xlabel('Estresse diário')
# plt.ylabel('Visão de vida')

# st.pyplot(plt)

plt.figure(figsize=(12, 6))
sns.boxplot(x='DAILY_STRESS', y='LIVE_VISION', hue='SUFFICIENT_INCOME', data=df, palette='viridis')
plt.title('Boxplot: Estresse Diário vs. Visão de Vida (Segmentado por Suficiência de Renda)')
plt.xlabel('Estresse Diário')
plt.ylabel('Visão de Vida')
# Ajustar a posição da legenda 
plt.legend(title='Renda', bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(plt)




# impactos econômicos e visão de propósito


# Relação entre Meditação e Equilíbrio Vida-Trabalho
st.header("Relação entre Meditação e Equilíbrio Vida-Trabalho")

st.write(
        """ Esse gráfico tem como objetivo explorar a relação entre as horas semanais dedicadas à meditação e o equilíbrio entre vida pessoal e trabalho. Ele utiliza um gráfico de dispersão para representar os dados de uma amostra da população, adicionando um "jitter" para facilitar a visualização dos pontos. Além disso, uma linha de tendência é incluída para destacar a correlação geral entre meditação e equilíbrio vida-trabalho.
        """
    )

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

ax.set_title("Relação entre Meditação Semanal e Equilíbrio Vida-Trabalho", fontsize=16)
ax.set_xlabel("Horas de Meditação por Semana", fontsize=14)
ax.set_ylabel("Pontuação de Equilíbrio Vida-Trabalho", fontsize=14)
ax.legend(fontsize=12)

st.pyplot(fig)

# Combine com a distribuição por gênero/faixa etária em abas separadas ou filtros para clareza.

# Distribuição de Tarefas Concluídas por Horas de Sono
st.header("Distribuição de Tarefas Concluídas por Horas de Sono")

st.write(
        """ Esse gráfico analisa como as horas de sono influenciam na quantidade de tarefas concluídas. Ele utiliza um gráfico de caixa para exibir a distribuição dos dados, focando em indivíduos que dormem até 9 horas por noite. O objetivo é observar padrões e variações na produtividade associadas a diferentes períodos de sono.
         """
    )

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

st.header("Heatmap de tarefas completadas, meditação semanal e Hobbies")
# Heatmap para explorar combinações de hábitos TODO_COMPLETED, WEEKLY_MEDITATION, e TIME_FOR_PASSION.
# habits_df = df[["TODO_COMPLETED", "WEEKLY_MEDITATION", "TIME_FOR_PASSION"]]

# habits_df = habits_df.apply(pd.to_numeric, errors="coerce")
# correlation_matrix = habits_df.corr()

# translations = {
#     "TODO_COMPLETED": "Tarefas Completadas",
#     "WEEKLY_MEDITATION": "Meditação Semanal",
#     "TIME_FOR_PASSION": "Hobbies"
# }

# plt.figure(figsize=(10, 6))  
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

# # Traduzir  eixos
# plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=[translations[col] for col in correlation_matrix.columns], rotation=45)
# plt.yticks(ticks=range(len(correlation_matrix.index)), labels=[translations[index] for index in correlation_matrix.index], rotation=0)


# plt.title("Correlação entre Hábitos")

# st.pyplot(plt)
st.write(
        """ 
O gráfico mostra que quanto mais tarefas concluidas e mais vezes fazer meditação na semana maior a quantidade de tempo disponível para hobbies
        """
    )


heatmap_data = df.pivot_table(index='TODO_COMPLETED', 
                              columns='WEEKLY_MEDITATION', 
                              values='TIME_FOR_PASSION', 
                              aggfunc='mean')

# Plotando o heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='coolwarm',  cbar_kws={'label': 'Tempo para Hobbies (em horas)'})
plt.title('Heatmap: Conclusão de Tarefas vs. Meditação Semanal e Tempo para Hobbies')
plt.xlabel('Meditação Semanal')
plt.ylabel('Conclusão de Tarefas')

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
    "TIME_FOR_PASSION": "Tempo para Hobbies"
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
st.write(
        """ 
No gráfico a seguir é possivel navegar pelos hábitos mostrados no heatmap a cima e verificar a influência desses hábitos no gênero e na idade
        """
    )
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