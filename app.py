import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json # Importa a biblioteca json

# --- FUNÇÃO PARA CARREGAR O MODELO PRÉ-TREINADO ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado. Certifique-se de que ele foi gerado pelo 'train.py' e está no repositório.")
        return None

# --- FUNÇÃO PARA CARREGAR A BASE COMPLETA DE CANDIDATOS ---
@st.cache_data
def carregar_candidatos_completos():
    """
    Carrega e prepara a base completa de candidatos de forma robusta,
    lendo o arquivo linha por linha para ignorar JSONs malformados.
    """
    data = []
    erros = 0
    with open(utils.NDJSON_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                erros += 1
                continue # Pula a linha com erro e continua para a próxima
    
    if erros > 0:
        st.warning(f"Atenção: {erros} registro(s) de candidatos foram ignorados devido a erros de formatação no arquivo de dados.")

    if not data:
        st.error("Nenhum registro de candidato pôde ser carregado.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df_normalized = pd.json_normalize(df.to_dict('records'), sep='_')
    
    coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
    if coluna_nome in df_normalized.columns:
        df_normalized.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
    else:
        df_normalized['nome_candidato'] = 'Nome não encontrado'
        
    df_normalized['codigo_candidato'] = df_normalized['codigo_candidato'].astype(str)
    return df_normalized

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
df_applicants_completo = carregar_candidatos_completos()

st.title("✨ Assistente de Recrutamento da Decision")

# --- Sidebar e CARREGAMENTO RÁPIDO do Modelo ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
    if google_api_key:
        genai.configure(api_key=google_api_key)
    st.markdown("---")
    st.header("Motor de Machine Learning")
    with st.spinner("Carregando modelo de IA..."):
        modelo_match = carregar_modelo_treinado()
    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.success("Modelo de Matching carregado!")
    else:
        st.error("Motor de ML indisponível.")

# O restante do seu código permanece o mesmo
# (Inicialização do Session State e as 3 abas)
