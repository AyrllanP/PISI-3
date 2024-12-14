import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Visão Geral do Dataset")

df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

st.header("Primeiros Registros do Dataset")
st.dataframe(df.head())

# Descrição Geral
st.header("Descrição Geral")
st.markdown("""
    Este dataset contém um conjunto de dados mostrando hábitos e comportamentos relacionados ao estilo de vida das pessoas.
""")

# Resumo estatístico
st.header("Resumo estatístico")
# Variáveis numéricas
st.subheader("Variáveis Numéricas")
st.dataframe(df.describe())

# Variáveis categóricas
st.subheader("Variáveis Categóricas")
categorical_stats = df.describe(include=['object'])
st.dataframe(categorical_stats)

# Contagem de valores ausentes
st.subheader("Valores Ausentes no Dataset")
missing_values = df.isnull().sum()
st.dataframe(missing_values[missing_values > 0])

# Gráficos de distribuição para variáveis-chave
st.header("Histogramas")
df["GENDER"] = df["GENDER"].replace({"Male": "Masculino", "Female": "Feminino"})
# Gênero
st.subheader("Distribuição de Gênero")
gender_counts = df['GENDER'].value_counts()
fig, ax = plt.subplots()
gender_counts.plot(kind='bar', ax=ax, color=['red', 'blue'])
ax.set_title("Distribuição de Gênero")
st.pyplot(fig)

# Função para categorizar faixas etárias como numéricas
def convert_age_to_numeric_range(age_str):
    if isinstance(age_str, str):
        if 'Less than' in age_str:
            return 10  # Representando a faixa 20 or less
        elif 'to' in age_str:
            start, end = age_str.split(' to ')
            return (int(start) + int(end)) / 2  # Média da faixa
        elif 'or more' in age_str:
            return 55  # Representar faixa 51+
    return age_str

# Aplicar a conversão para a coluna 'AGE'
df['AGE_NUMERIC'] = df['AGE'].apply(convert_age_to_numeric_range)

# Lista de rótulos para as faixas
age_bins = [0, 20, 36, 51, 80]  # Bins numéricos representando os intervalos
age_labels = ['Menos de 20', '21 à 30', '36 à 50', 'Mais de 51']

# Criar nova coluna categórica baseada nos bins
df['AGE_BINS'] = pd.cut(df['AGE_NUMERIC'], bins=age_bins, labels=age_labels, right=False)

# Criar o histograma
plt.figure(figsize=(10, 6))
sns.histplot(df, x='AGE_BINS', hue=None, bins=4, color='green')

plt.xlabel("Faixa Etária")
plt.ylabel("Frequência")
plt.title("Distribuição de Idade por Faixa Etária")

plt.title("Distribuição de idades")
# Mostrar o gráfico no Streamlit
st.pyplot(plt)

# Estresse diário
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  # substituindo a string de DAILY_STRESS por 3 por que é a média dos valores 
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce') # transformando em números

st.subheader("Distribuição de Estresse Diário")
stress_counts = df['DAILY_STRESS'].value_counts().sort_index()

fig, ax = plt.subplots()
stress_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)

# Configurações do gráfico
ax.set_title("Distribuição de Estresse Diário")
ax.set_xlabel("Nível de Estresse")
ax.set_ylabel("Frequência")
ax.set_xticks(range(len(stress_counts)))  # Define os ticks no eixo x
ax.set_xticklabels(stress_counts.index.astype(int), rotation=0)  # Garante que os rótulos são inteiros e alinhados
st.pyplot(fig)

# Horas de sono
df['SLEEP_HOURS'] = pd.to_numeric(df['SLEEP_HOURS'], errors='coerce')

st.subheader("Distribuição de Horas de Sono")
sleep_counts = df['SLEEP_HOURS'].value_counts().sort_index()

fig, ax = plt.subplots()
sleep_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)

# Configurações do gráfico
ax.set_title("Distribuição de Horas de Sono")
ax.set_xlabel("Horas de Sono")
ax.set_ylabel("Frequência")
ax.set_xticks(range(len(sleep_counts)))  # Define os ticks no eixo x
ax.set_xticklabels(sleep_counts.index.astype(int), rotation=0)  # Garante que os rótulos são inteiros e alinhados
st.pyplot(fig)

# Passos por dia

df['DAILY_STEPS'] = pd.to_numeric(df['DAILY_STEPS'], errors='coerce')

st.subheader("Distribuição de Nível de atividade")
steps_counts = df['DAILY_STEPS'].value_counts().sort_index()

fig, ax = plt.subplots()
steps_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)

# Configurações do gráfico
ax.set_title("Distribuição de Nível de atividade")
ax.set_xlabel("Nível de atividade")
ax.set_ylabel("Frequência")
ax.set_xticks(range(len(steps_counts)))  # Define os ticks no eixo x
ax.set_xticklabels(steps_counts.index.astype(int), rotation=0)  # Garante que os rótulos são inteiros e alinhados
st.pyplot(fig)

# Uma escala de 1 a 10 indica que essa variável provavelmente foi normalizada ou transformada de alguma forma para representar categorias de intensidade ou níveis de atividade.


# Distribuição por genero em estresse diário
st.subheader("Distribuição por genero em estresse diário")

df["GENDER"] = df["GENDER"].replace({"Male": "Masculino", "Female": "Feminino"}) # Arrumar legendas

plt.figure(figsize=(10, 6))

sns.boxplot(x="GENDER", y="DAILY_STRESS", data=df, palette="Set2")

# Calcular medianas e adicionar destaque
medians = df.groupby("GENDER")["DAILY_STRESS"].median()

# Adiciona linha horizontal em vermelho na mediana
for i, median in enumerate(medians):
    plt.scatter(i, median, color='red', zorder=3)  # Adiciona ponto vermelho na mediana
    plt.text(i, median + 0.1, f'Mediana: {median:.2f}', 
             color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel("Gênero")
plt.ylabel("Estresse diário")

plt.title("Distribuição do Estresse Diário por Gênero")
st.pyplot(plt)


# Informação sobre correlação entre variáveis
st.header("Correlação entre variáveis do Dataset")

numerical_df = df.select_dtypes(include=['number'])

correlation_matrix = numerical_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação')

st.pyplot(plt)