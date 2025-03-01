import streamlit as st
import pandas as pd
import joblib

# Modelo SVM (Sem SMOTE - Previsão com dados gerais)
def carregar_modelo_e_scaler():
    svm_model = joblib.load("./svm_model.pkl")
    scaler = joblib.load("./scaler.pkl")

    return svm_model, scaler

# Função para pré-processar os dados de entrada do usuário
def preprocessar_entrada_usuario(dados_usuario, scaler, colunas_originais):
    df_usuario = pd.DataFrame([dados_usuario])
    
    for coluna in colunas_originais:
        if coluna not in df_usuario.columns:
            df_usuario[coluna] = 0
    
    
    df_usuario = df_usuario[colunas_originais]
    
    df_usuario['GENDER'] = df_usuario['GENDER'].replace({'Male': 0, 'Female': 1})
    df_usuario['AGE'] = df_usuario['AGE'].replace({'Less than 20': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3})
    
    dados_preprocessados = scaler.transform(df_usuario)
    
    return dados_preprocessados


def main():
    st.title("Previsão de Nível de Estresse")
    
    svm_model, scaler = carregar_modelo_e_scaler()
    
    colunas_originais = [
        'FRUITS_VEGGIES', 'PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS', 'SOCIAL_NETWORK',
        'ACHIEVEMENT', 'DONATION', 'BMI_RANGE', 'TODO_COMPLETED', 'FLOW', 'DAILY_STEPS', 'LIVE_VISION',
        'SLEEP_HOURS', 'LOST_VACATION', 'DAILY_SHOUTING', 'SUFFICIENT_INCOME', 'PERSONAL_AWARDS',
        'TIME_FOR_PASSION', 'WEEKLY_MEDITATION', 'AGE', 'GENDER', 'WORK_LIFE_BALANCE_SCORE'
    ]
    
    # Formulário de entrada de dados
    with st.form("form_estresse"):
        st.header("Insira seus dados:")
        
        fruits_veggies = st.number_input("Quantas porções de frutas ou vegetais você come por dia? (0-5)", min_value=0, max_value=5, value=3)
        places_visited = st.number_input("Quantos lugares novos você visita? (0-10)", min_value=0, max_value=10, value=5)
        core_circle = st.number_input("Quantas pessoas são muito próximas de você? (0-10)", min_value=0, max_value=10, value=5)
        supporting_others = st.number_input("Quantas pessoas você ajuda a alcançar uma vida melhor? (0-10)", min_value=0, max_value=10, value=5)
        social_network = st.number_input("Com quantas pessoas você interage em um dia típico? (0-10)", min_value=0, max_value=10, value=5)
        achievement = st.number_input("Quantas conquistas notáveis você tem orgulho? (0-10)", min_value=0, max_value=10, value=5)
        donation = st.number_input("Quantas vezes você doa seu tempo ou dinheiro para boas causas? (0-5)", min_value=0, max_value=5, value=2)
        bmi_range = st.number_input("Qual é o seu índice de massa corporal (BMI)? (1-2)", min_value=1, max_value=2, value=1)
        todo_completed = st.number_input("Quão bem você completa suas listas de tarefas semanais? (0-10)", min_value=0, max_value=10, value=5)
        flow = st.number_input("Em um dia típico, quantas horas você experimenta 'flow'? (0-10)", min_value=0, max_value=10, value=5)
        daily_steps = st.number_input("Quantos passos (em milhares) você dá por dia? (1-10)", min_value=1, max_value=10, value=5)
        live_vision = st.number_input("Por quantos anos à frente sua visão de vida é muito clara? (0-10)", min_value=0, max_value=10, value=5)
        sleep_hours = st.number_input("Quantas horas você dorme por noite? (1-10)", min_value=1, max_value=10, value=7)
        lost_vacation = st.number_input("Quantos dias de férias você perde por ano? (0-10)", min_value=0, max_value=10, value=2)
        daily_shouting = st.number_input("Com que frequência você grita ou fica emburrado com alguém? (0-10)", min_value=0, max_value=10, value=2)
        sufficient_income = st.number_input("Quão suficiente é sua renda para cobrir despesas básicas? (1-2)", min_value=1, max_value=2, value=1)
        personal_awards = st.number_input("Quantos reconhecimentos você recebeu em sua vida? (0-10)", min_value=0, max_value=10, value=2)
        time_for_passion = st.number_input("Quantas horas por dia você dedica ao que é apaixonado? (0-10)", min_value=0, max_value=10, value=2)
        weekly_meditation = st.number_input("Quantas vezes por semana você medita? (0-10)", min_value=0, max_value=10, value=2)
        age = st.selectbox("Idade", ["Less than 20", "21 to 35", "36 to 50", "51 or more"])
        gender = st.selectbox("Gênero", ["Male", "Female"])
        work_life_balance_score = st.number_input("Pontuação de equilíbrio vida-trabalho (480-820)", min_value=480.0, max_value=820.0, value=650.0)
        
        submitted = st.form_submit_button("Prever Nível de Estresse")
        
        if submitted:
            dados_usuario = {
                'FRUITS_VEGGIES': fruits_veggies,
                'PLACES_VISITED': places_visited,
                'CORE_CIRCLE': core_circle,
                'SUPPORTING_OTHERS': supporting_others,
                'SOCIAL_NETWORK': social_network,
                'ACHIEVEMENT': achievement,
                'DONATION': donation,
                'BMI_RANGE': bmi_range,
                'TODO_COMPLETED': todo_completed,
                'FLOW': flow,
                'DAILY_STEPS': daily_steps,
                'LIVE_VISION': live_vision,
                'SLEEP_HOURS': sleep_hours,
                'LOST_VACATION': lost_vacation,
                'DAILY_SHOUTING': daily_shouting,
                'SUFFICIENT_INCOME': sufficient_income,
                'PERSONAL_AWARDS': personal_awards,
                'TIME_FOR_PASSION': time_for_passion,
                'WEEKLY_MEDITATION': weekly_meditation,
                'AGE': age,
                'GENDER': gender,
                'WORK_LIFE_BALANCE_SCORE': work_life_balance_score
            }

            dados_preprocessados = preprocessar_entrada_usuario(dados_usuario, scaler, colunas_originais)

            previsao = svm_model.predict(dados_preprocessados)
            
            niveis_estresse = {0: 'baixo', 1: 'moderado', 2: 'alto'}
            nivel_estresse = niveis_estresse[previsao[0]]

            st.success(f"Previsão de nível de estresse: {nivel_estresse}")

if __name__ == "__main__":
    main()