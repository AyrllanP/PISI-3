import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Título da página
st.title("Relação entre Meditação e Equilíbrio Vida-Trabalho")

# Leitura dos dados
file_path = st.text_input("Caminho do arquivo CSV:", "C:/Users/peido/Desktop/Dataset/Wellbeing_and_lifestyle_data_Kaggle.csv")
if st.button("Carregar dados"):
    try:
        # Carregar o dataset
        df = pd.read_csv(file_path)

        # Configurando o estilo do Seaborn
        sns.set(style="whitegrid")

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

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
