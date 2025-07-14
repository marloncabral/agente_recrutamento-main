import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import shap
import matplotlib.pyplot as plt

import utils
import ml_logic

st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

# Funções de IA Generativa (sem alterações)
# ... (mantenha suas funções de IA Generativa aqui)

# --- Inicialização e Carregamento de Dados ---
if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

vagas_data_dict = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data_dict = utils.carregar_prospects()
df_vagas_ui = utils.carregar_vagas()

st.title("✨ Assistente de Recrutamento da Decision")

# --- Sidebar e Treinamento do Modelo com Feedback por Etapas ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
    st.markdown("---")
    st.header("Motor de Machine Learning")

    # --- INÍCIO DA LÓGICA DE TREINAMENTO POR ETAPAS ---
    # Verifica se o modelo já foi treinado e está no cache da sessão
    if 'modelo_match' not in st.session_state:
        try:
            # Etapa 1
            with st.spinner('Etapa 1 de 3: Preparando e limpando os dados...'):
                df_treino = ml_logic.etapa_1_preparar_dados(vagas_data_dict, prospects_data_dict)
            st.success("Etapa 1 de 3: Dados preparados com sucesso!")

            # Etapa 2
            with st.spinner('Etapa 2 de 3: Treinando o modelo... (Esta etapa pode levar um momento)'):
                if df_treino is not None:
                    pipeline, X_test, y_test = ml_logic.etapa_2_treinar_modelo(df_treino)
                else:
                    pipeline, X_test, y_test = None, None, None
            st.success("Etapa 2 de 3: Modelo treinado com sucesso!")
            
            # Etapa 3
            with st.spinner('Etapa 3 de 3: Avaliando e finalizando...'):
                performance_string = ml_logic.etapa_3_avaliar_e_finalizar(pipeline, X_test, y_test)
            st.success("Etapa 3 de 3: Avaliação concluída!")

            # Armazena o modelo treinado no cache da sessão
            if pipeline:
                st.session_state.modelo_match = pipeline
                st.session_state.model_performance = performance_string
                st.session_state.ml_mode_available = True
            else:
                st.session_state.ml_mode_available = False
                st.error("Falha crítica no treinamento do modelo.")

        except Exception as e:
            st.error(f"Ocorreu um erro durante o treinamento: {e}")
            st.session_state.ml_mode_available = False
    
    # Exibe a performance final após o treinamento ou se já estiver em cache
    if st.session_state.get('ml_mode_available', False):
        st.info(st.session_state.get('model_performance', 'Modelo carregado.'))
    # --- FIM DA LÓGICA DE TREINAMENTO POR ETAPAS ---

# --- Inicialização do Session State ---
# ... (mantenha o restante do seu código do app.py aqui, sem alterações)
# O código das abas, etc., não precisa ser modificado.

with tab1:
    # ... (código da tab1)
with tab2:
    # ... (código da tab2)
with tab3:
    # ... (código da tab3)
