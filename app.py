import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils

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
    """Carrega e prepara a base completa de candidatos uma única vez."""
    try:
        df = pd.read_json(utils.NDJSON_FILENAME, lines=True)
        df_normalized = pd.json_normalize(df.to_dict('records'), sep='_')
        coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
        if coluna_nome in df_normalized.columns:
            df_normalized.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
        else:
            df_normalized['nome_candidato'] = 'Nome não encontrado'
        df_normalized['codigo_candidato'] = df_normalized['codigo_candidato'].astype(str)
        return df_normalized
    except Exception as e:
        st.error(f"Erro ao carregar a base de dados de candidatos: {e}")
        return pd.DataFrame()

# --- FUNÇÕES DE IA GENERATIVA (ENTREVISTA) ---
def gerar_proxima_pergunta(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Chave de API não configurada."
    prompt = f"""
    Você é um entrevistador de IA da Decision. Sua tarefa é conduzir uma entrevista concisa, fazendo UMA PERGUNTA DE CADA VEZ.
    **Regra Principal:** Formule a PRÓXIMA pergunta com base no histórico. Não repita perguntas. Se já fez 5-6 perguntas, finalize a entrevista.

    **Contexto da Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Contexto do Candidato:** {candidato.get('nome_candidato', 'N/A')} | {candidato.get('candidato_texto_completo', '')}
    **Histórico da Entrevista até agora:**
    {historico_chat}

    **Sua Ação:** Formule a próxima pergunta para o candidato.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt
