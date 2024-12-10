import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Título da página
st.title("Distribuição de Tarefas Concluídas por Horas de Sono")

# Leitura dos dados
file_path = st.text_input("Caminho do arquivo CSV:", "C:/Users/peido/Desktop/Dataset/Wellbeing_and_lifestyle_data_Kaggle.csv")
if st.button("Carregar dados"):
    try:
        # Carregar o dataset
        df = pd.read_csv(file_path)

        # Filtrar dados para considerar apenas até 9 horas de sono
        df_filtered = df[df["SLEEP_HOURS"] <= 9]

        # Configurando o estilo do Seaborn
        sns.set(style="whitegrid")

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

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
