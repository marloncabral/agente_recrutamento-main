import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json

# --- FUNÇÃO PARA CARREGAR O MODELO PRÉ-TREINADO ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado.")
        return None

# --- FUNÇÃO PARA CARREGAR A BASE COMPLETA DE CANDIDATOS ---
@st.cache_data
def carregar_candidatos_completos():
    """Carrega e prepara a base de candidatos de forma robusta."""
    data = []
    erros = 0
    with open(utils.NDJSON_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                erros += 1
                continue
    if erros > 0:
        st.warning(f"Atenção: {erros} registro(s) de candidatos foram ignorados devido a erros de formatação.")
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

# Prepara os arquivos de dados (download, conversão)
if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos arquivos de dados. A aplicação será interrompida.")
    st.stop()

# Carrega os dados essenciais em memória
df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
df_applicants_completo = carregar_candidatos_completos()
modelo_match = carregar_modelo_treinado()

# --- VERIFICAÇÃO DE INTEGRIDADE DOS DADOS CARREGADOS ---
dados_carregados_com_sucesso = True
if df_vagas_ui is None or df_vagas_ui.empty:
    st.error("Falha ao carregar os dados das vagas (`vagas.json`). A aplicação não pode continuar.")
    dados_carregados_com_sucesso = False

if prospects_data_dict is None:
    st.error("Falha ao carregar os dados dos prospects (`prospects.json`). A aplicação não pode continuar.")
    dados_carregados_com_sucesso = False

if df_applicants_completo.empty:
    st.error("Falha ao carregar os dados dos candidatos (`applicants.nd.json`). A aplicação não pode continuar.")
    dados_carregados_com_sucesso = False

if modelo_match is None:
    st.error("Falha ao carregar o modelo de Machine Learning (`modelo_recrutamento.joblib`).")
    dados_carregados_com_sucesso = False

# --- Início do App (só executa se os dados estiverem OK) ---
if dados_carregados_com_sucesso:
    st.title("✨ Assistente de Recrutamento da Decision")

    with st.sidebar:
        st.header("Configuração Essencial")
        google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
        if google_api_key:
            genai.configure(api_key=google_api_key)
        st.markdown("---")
        st.header("Motor de Machine Learning")
        st.success("Modelo de Matching carregado!")

    # Inicialização do Session State
    if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
    if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
    if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}

    tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

    with tab1:
        st.header("Matching com Machine Learning")
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x])

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            # A lógica de análise continua aqui...
            pass # (O restante do seu código da tab1 permanece o mesmo)
