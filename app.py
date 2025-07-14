import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib # Importa a biblioteca para carregar o modelo

import utils

# --- FUNÇÃO PARA CARREGAR O MODELO PRÉ-TREINADO ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado. Por favor, execute o script 'train.py' localmente e envie o arquivo para o repositório.")
        return None

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

vagas_data_dict = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data_dict = utils.carregar_prospects()
df_vagas_ui = utils.carregar_vagas()

st.title("✨ Assistente de Recrutamento da Decision")

# --- Sidebar e CARREGAMENTO RÁPIDO do Modelo ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
    if google_api_key:
        genai.configure(api_key=google_api_key)
        
    st.markdown("---")
    st.header("Motor de Machine Learning")
    
    with st.spinner("Carregando modelo de Machine Learning..."):
        modelo_match = carregar_modelo_treinado()

    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.session_state.ml_mode_available = True
        st.success("Modelo de Matching carregado com sucesso!")
    else:
        st.session_state.ml_mode_available = False
        st.error("Motor de ML indisponível.")

# --- Inicialização do Session State ---
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
# ... (outros session_state que você usa) ...

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if st.session_state.get('ml_mode_available', False):
        st.header("Matching com Machine Learning")
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x])

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            with st.spinner("Analisando candidatos..."):
                prospects_da_vaga = prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])
                if not prospects_da_vaga:
                    st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")
                else:
                    ids_candidatos = [p['codigo'] for p in prospects_da_vaga]
                    df_detalhes = utils.buscar_detalhes_candidatos(ids_candidatos)

                    if not df_detalhes.empty:
                        vaga_selecionada_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                        perfil_vaga_texto = vaga_selecionada_data['perfil_vaga_texto']

                        if 'nome' in df_detalhes.columns:
                            df_detalhes.rename(columns={'nome': 'nome_candidato'}, inplace=True)

                        df_detalhes['texto_completo'] = perfil_vaga_texto + ' ' + df_detalhes['candidato_texto_completo'].fillna('')
                        df_detalhes.dropna(subset=['nome_candidato'], inplace=True)

                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probabilidades[:, 1] * 100).astype(int)

                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)

    # O resto do seu código para as abas continua aqui, sem alterações na lógica de exibição.
    if not st.session_state.df_analise_resultado.empty:
        # ... (código do data_editor e botão de confirmar)
        pass # Mantenha seu código aqui

with tab2:
    # ... (código da tab2)
    pass # Mantenha seu código aqui

with tab3:
    # ... (código da tab3)
    pass # Mantenha seu código aqui
