import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Visão Geral do Dataset")

# Para Mudança com Parquet
df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')
df.to_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')
df = pd.read_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')

# Tratando valor de string do estresse diário
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')


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

# Distribuição de Genero
st.header("Histogramas")
df["GENDER"] = df["GENDER"].replace({"Male": "Masculino", "Female": "Feminino"})
st.subheader("Distribuição de Gênero")
gender_counts = df['GENDER'].value_counts()
fig, ax = plt.subplots()
gender_counts.plot(kind='bar', ax=ax, color=['red', 'blue'])
ax.set_title("Distribuição de Gênero")
st.pyplot(fig)

# Distribuição de idades
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

df['AGE_NUMERIC'] = df['AGE'].apply(convert_age_to_numeric_range)
age_bins = [0, 20, 36, 51, 80] 
age_labels = ['Menos de 20', '21 à 30', '36 à 50', 'Mais de 51']
df['AGE_BINS'] = pd.cut(df['AGE_NUMERIC'], bins=age_bins, labels=age_labels, right=False)
plt.figure(figsize=(10, 6))
sns.histplot(df, x='AGE_BINS', hue=None, bins=4, color='green')
plt.xlabel("Faixa Etária")
plt.ylabel("Frequência")
plt.title("Distribuição de Idade por Faixa Etária")
plt.title("Distribuição de idades")
st.pyplot(plt)

# Estresse diário - Tratado
st.subheader("Distribuição de Estresse Diário")
stress_counts = df['DAILY_STRESS'].value_counts().sort_index()
fig, ax = plt.subplots()
stress_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)
ax.set_title("Distribuição de Estresse Diário")
ax.set_xlabel("Nível de Estresse")
ax.set_ylabel("Frequência")
ax.set_xticks(range(len(stress_counts))) 
ax.set_xticklabels(stress_counts.index.astype(int), rotation=0) 
st.pyplot(fig)

# Horas de sono
df['SLEEP_HOURS'] = pd.to_numeric(df['SLEEP_HOURS'], errors='coerce')
st.subheader("Distribuição de Horas de Sono")
sleep_counts = df['SLEEP_HOURS'].value_counts().sort_index()
fig, ax = plt.subplots()
sleep_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)
ax.set_title("Distribuição de Horas de Sono")
ax.set_xlabel("Horas de Sono")
ax.set_ylabel("Frequência")
ax.set_xticks(range(len(sleep_counts)))  
ax.set_xticklabels(sleep_counts.index.astype(int), rotation=0)  
st.pyplot(fig)

# Passos por dia - Nível de Atividade
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

# Destaque para medianas
medians = df.groupby("GENDER")["DAILY_STRESS"].median()
for i, median in enumerate(medians):
    plt.scatter(i, median, color='red', zorder=3)  
    plt.text(i, median + 0.1, f'Mediana: {median:.2f}', 
             color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xlabel("Gênero")
plt.ylabel("Estresse diário")
plt.title("Distribuição do Estresse Diário por Gênero")
st.pyplot(plt)


# Estresse diário por idade
st.subheader("Distribuição por faixa etária em estresse diário")
age_filter = st.selectbox('Selecione uma faixa etária:', df['AGE'].unique())
filtered_df = df[df['AGE'] == age_filter]
st.subheader(f"Distribuição de Estresse Diário para {age_filter}")
fig, ax = plt.subplots()
filtered_df['DAILY_STRESS'].value_counts().plot(kind='bar', ax=ax, color='lightblue')
st.pyplot(fig)


# Informação sobre correlação entre variáveis
st.header("Correlação entre variáveis do Dataset")
numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação')
st.pyplot(plt)