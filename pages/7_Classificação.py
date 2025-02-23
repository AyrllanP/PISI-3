import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar e pré-processar os dados
def load_data():
    file_url = "https://raw.githubusercontent.com/AyrllanP/PISI-3/main/Wellbeing_and_lifestyle_data_Kaggle.csv"
    df = pd.read_csv(file_url)
    
    # Pré-processamento
    df['DAILY_STRESS'] = df['DAILY_STRESS'].replace("1/1/00", 3)
    df['DAILY_STRESS'] = df['DAILY_STRESS'].astype(int)
    df['DAILY_STRESS'] = df['DAILY_STRESS'].replace({0: 'baixo', 1: 'baixo', 2: 'moderado', 3: 'moderado', 4: 'alto', 5: 'alto'})
    df = df.drop(["Timestamp"], axis=1, errors='ignore')
    df['DAILY_STRESS'] = df['DAILY_STRESS'].replace({'baixo': 0, 'moderado': 1, 'alto': 2})
    
    # Codificação de variáveis categóricas
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['GENDER', 'AGE'])], remainder='passthrough')
    X = np.array(ct.fit_transform(df.drop('DAILY_STRESS', axis=1)))
    y = df['DAILY_STRESS'].values
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


# Função para treinar e avaliar o Random Forest
def random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    
    # Métricas
    st.write("### Resultados do Random Forest")
    st.write("#### Conjunto de Treino")
    st.write(f"Acurácia: {accuracy_score(y_train, y_train_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    st.write("#### Conjunto de Teste")
    st.write(f"Acurácia: {accuracy_score(y_test, y_test_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Função para treinar e avaliar o KNN
def knn(X_train, X_test, y_train, y_test, k=20):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = knn_classifier.predict(X_train)
    y_test_pred = knn_classifier.predict(X_test)
    
    # Métricas
    st.write("### Resultados do KNN")
    st.write("#### Conjunto de Treino")
    st.write(f"Acurácia: {accuracy_score(y_train, y_train_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    st.write("#### Conjunto de Teste")
    st.write(f"Acurácia: {accuracy_score(y_test, y_test_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Função para treinar e avaliar o SVM
def svm(X_train, X_test, y_train, y_test):
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_classifier.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = svm_classifier.predict(X_train)
    y_test_pred = svm_classifier.predict(X_test)
    
    # Métricas
    st.write("### Resultados do SVM")
    st.write("#### Conjunto de Treino")
    st.write(f"Acurácia: {accuracy_score(y_train, y_train_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    st.write("#### Conjunto de Teste")
    st.write(f"Acurácia: {accuracy_score(y_test, y_test_pred):.2f}")
    st.write("Matriz de Confusão:")
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Interface do Streamlit
def main():
    st.title("Classificação de Níveis de Estresse")
    st.write("Escolha um algoritmo para classificar os níveis de estresse diário.")
    
    # Carregar dados
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Seleção do algoritmo
    algoritmo = st.selectbox("Selecione o algoritmo:", ["Random Forest", "KNN", "SVM"])
    
    if algoritmo == "Random Forest":
        random_forest(X_train, X_test, y_train, y_test)
    elif algoritmo == "KNN":
        k = st.slider("Selecione o valor de k:", 1, 30, 20)
        knn(X_train, X_test, y_train, y_test, k)
    elif algoritmo == "SVM":
        svm(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()