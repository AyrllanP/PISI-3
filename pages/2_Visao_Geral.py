import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder

st.title("Visão Geral do Dataset")

# Para Mudança com Parquet
df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')
df.to_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')
df = pd.read_parquet('Wellbeing_and_lifestyle_data_Kaggle.parquet')

# Tratando valor de string do estresse diário
df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)  
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')

# Descrição Geral
st.header("Descrição Geral")
st.markdown("""
    Este dataset contém um conjunto de dados mostrando hábitos e comportamentos relacionados ao estilo de vida das pessoas.
            
    Explore as informações abaixo para entender melhor a composição e características dos dados.
""")

st.header("Vizualização do dataset")

st.subheader("Visualização do Dataset")
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationAutoPageSize=True)  # Paginação automática
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_side_bar()  # Barra lateral com filtros
grid_options = gb.build()

AgGrid(df, gridOptions=grid_options, height=300, theme='streamlit')



# Resumo estatístico
st.header("Resumo estatístico")
# Variáveis numéricas
st.subheader("Variáveis Numéricas")
translated_metrics_num = {
    "count": "Contagem",
    "mean": "Média",
    "std": "Desvio Padrão",
    "min": "Valor Mínimo",
    "25%": "1º Quartil (25%)",
    "50%": "Mediana (50%)",
    "75%": "3º Quartil (75%)",
    "max": "Valor Máximo",
}
st.dataframe(df.describe().rename(index=translated_metrics_num))

# Variáveis categóricas
st.subheader("Variáveis Categóricas")
translated_metrics_cat = {
    "count": "Contagem",
    "unique": "Valores Únicos",
    "top": "Valor Mais Frequente",
    "freq": "Frequência do Mais Frequente",
}
categorical_stats = df.describe(include=['object'],).rename(index=translated_metrics_cat)
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
gender_counts.plot(kind='bar', ax=ax, color=['pink', 'lightblue'], edgecolor='black', align='center', width=0.8 )
ax.set_title("Distribuição de Gênero")
plt.xlabel("Gênero")
plt.ylabel("Frequência")
ax.set_xticklabels(gender_counts.index, rotation=0)
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
sns.histplot(df, x='AGE_BINS', hue=None, bins=4, color='lightgreen')
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
age_translation = {
    "Less than 20": "Menos de 20",
    "21 to 35": "de 21 até 35",
    "36 to 50": "de 36 até 50",
    "51 or more": "50 ou mais ",
}

df['AGE_TRANSLATED'] = df["AGE"].map(age_translation)

# ordem 
age_order = ["Menos de 20", "de 21 até 35", "de 36 até 50", "50 ou mais "]
age_translation_reverse = {v: k for k, v in age_translation.items()}

# filtro
st.subheader("Distribuição por faixa etária em estresse diário")
age_filter = st.selectbox('Selecione uma faixa etária:', age_order)

original_age_filter = age_translation_reverse[age_filter]

filtered_df = df[df['AGE'] == original_age_filter]

st.subheader(f"Distribuição de Estresse Diário para {age_filter}")
fig, ax = plt.subplots()

# Ordem estresse 
stress_order = [1, 2, 3, 4, 5]
filtered_df['DAILY_STRESS'] = pd.Categorical(filtered_df['DAILY_STRESS'], categories=stress_order, ordered=True)

# Contar e ordenar os valores de estresse
stress_counts = filtered_df['DAILY_STRESS'].value_counts().sort_index()

# Plotar o gráfico de barras
stress_counts.plot(kind='bar', ax=ax, color='lightblue', edgecolor='black', align='center', width=0.8)

ax.set_xlabel("Nível de Estresse Diário")
ax.set_ylabel("Frequência")
st.pyplot(fig)


# Informação sobre correlação entre variáveis
st.header("Correlação entre variáveis do Dataset")
numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação')
st.pyplot(plt)
