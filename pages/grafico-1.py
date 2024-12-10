import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Título da página
st.title("Gráfico de Calor: Metas de Vida")

# Leitura dos dados
file_path = st.text_input("Caminho do arquivo CSV:", "C:/Users/peido/Desktop/Dataset/Wellbeing_and_lifestyle_data_Kaggle.csv")
if st.button("Carregar dados"):
    try:
        df = pd.read_csv(file_path)

        # Configurando o estilo do Seaborn
        sns.set(style="white")

        # Criando a tabela pivô para o mapa de calor
        heatmap_data = df.pivot_table(
            values="LIVE_VISION",  # Metas de vida como valores
            index="DAILY_STEPS",   # Passos diários no eixo Y
            columns="FRUITS_VEGGIES",  # Frutas e vegetais no eixo X
            aggfunc="mean"  # Cálculo da média
        )

        # Criando o mapa de calor
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
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
