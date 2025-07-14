import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import re
import json
import shap
import matplotlib.pyplot as plt

import utils
import ml_logic

st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

# Funções de IA Generativa (sem alterações)
def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.\n**Vaga:** {vaga.get('titulo_vaga', 'N/A')}\n**Candidato:** {candidato.get('nome_candidato', 'N/A')}\n**Transcrição:**\n{historico_chat}\n**Sua Tarefa:** Gere um relatório estruturado em markdown com: ### Relatório Final de Entrevista, 1. Score Geral (0 a 10), 2. Pontos Fortes, 3. Pontos de Atenção, 4. Recomendação Final ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado")."""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Ocorreu um erro ao gerar o relatório: {e}"

# --- Inicialização e Carregamento de Dados ---
if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

vagas_data_dict = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data_dict = utils.carregar_prospects()
df_vagas_ui = utils.carregar_vagas()

st.title("✨ Assistente de Recrutamento da Decision")

# --- Sidebar e Treinamento do Modelo ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
    st.markdown("---")
    st.header("Motor de Machine Learning")
    modelo_match, X_train_data = ml_logic.treinar_modelo_matching(vagas_data_dict, prospects_data_dict)
    
    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.session_state.ml_mode_available = True
        if 'model_performance' in st.session_state:
             st.success(st.session_state.model_performance)
    else:
        st.session_state.ml_mode_available = False
        st.warning("Fallback: Usando IA Generativa para o matching.")

# --- Inicialização do Session State ---
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if st.session_state.get('ml_mode_available', False):
        st.header("Matching com Machine Learning")
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x], key="ml_vaga_select")

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            with st.spinner("Analisando candidatos com o modelo de ML..."):
                df_mestre_completo = ml_logic.criar_dataframe_mestre_otimizado(vagas_data_dict, prospects_data_dict)
                df_detalhes = df_mestre_completo[df_mestre_completo['codigo_vaga'] == codigo_vaga_selecionada].copy()

                if not df_detalhes.empty:
                    df_detalhes_preparado = ml_logic.preparar_dados_para_treino(df_detalhes)
                    
                    if df_detalhes_preparado is not None:
                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes_preparado[['texto_completo']])
                        df_detalhes_preparado['score'] = (probabilidades[:, 1] * 100).astype(int)
                        st.session_state.df_analise_resultado = df_detalhes_preparado.sort_values(by='score', ascending=False).head(20)
                    else:
                        st.warning("Não foi possível preparar os dados dos candidatos para o modelo.")
                else:
                    st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")

    if not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        # --- CORREÇÃO AQUI: Assegura que as colunas certas sejam usadas ---
        colunas_para_exibir = ['codigo_candidato', 'nome_candidato', 'score']
        df_para_editar = st.session_state.df_analise_resultado[colunas_para_exibir].copy()
        df_para_editar['selecionar'] = False
        
        df_editado = st.data_editor(
            df_para_editar, 
            column_config={
                "selecionar": st.column_config.CheckboxColumn("Selecionar p/ Entrevista", default=False),
                "codigo_candidato": None, # Oculta a coluna de ID
                "nome_candidato": st.column_config.TextColumn("Nome do Candidato"),
                "score": st.column_config.ProgressColumn("Score de Match (%)", min_value=0, max_value=100)
            }, 
            hide_index=True, 
            use_container_width=True
        )

        if st.button("Confirmar Seleção para Entrevista"):
            # --- CORREÇÃO AQUI: Usa o 'codigo_candidato' para fazer a seleção ---
            codigos_selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
            if codigos_selecionados:
                df_completo = st.session_state.df_analise_resultado
                df_selecionados_final = df_completo[df_completo['codigo_candidato'].isin(codigos_selecionados)]
                
                st.session_state.candidatos_para_entrevista = df_selecionados_final.to_dict('records')
                vaga_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                st.session_state.vaga_selecionada = vaga_data
                st.success(f"{len(codigos_selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                time.sleep(1); st.rerun()
            else:
                st.warning("Nenhum candidato selecionado.")

with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        
        # --- CORREÇÃO AQUI: Usa as chaves corretas do dicionário ---
        nomes_candidatos = {c['codigo_candidato']: c['nome_candidato'] for c in st.session_state.candidatos_para_entrevista}
        id_candidato_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x])
        
        # Lógica da entrevista continua...

with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    # Lógica da análise final continua...
