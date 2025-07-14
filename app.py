import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import shap
import matplotlib.pyplot as plt

import utils
import ml_logic

st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

# Funções de IA Generativa
def gerar_relatorio_final(vaga, candidato, api_key):
    if not api_key: return "Erro: Chave de API do Google não configurada."
    # ... (código da função omitido para brevidade, mantenha o seu)

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
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x])

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            with st.spinner("Analisando candidatos com o modelo de ML..."):
                prospects_da_vaga = prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])
                if not prospects_da_vaga:
                    st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")
                else:
                    ids_candidatos = [p['codigo'] for p in prospects_da_vaga]
                    df_detalhes = utils.buscar_detalhes_candidatos(ids_candidatos)

                    if not df_detalhes.empty:
                        # --- INÍCIO DA LÓGICA CORRIGIDA ---
                        vaga_selecionada_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                        perfil_vaga_texto = vaga_selecionada_data['perfil_vaga_texto']

                        # Renomeia a coluna para o padrão do app
                        if 'nome' in df_detalhes.columns:
                            df_detalhes.rename(columns={'nome': 'nome_candidato'}, inplace=True)

                        # Prepara o texto completo para a predição
                        df_detalhes['texto_completo'] = perfil_vaga_texto + ' ' + df_detalhes['candidato_texto_completo'].fillna('')

                        # Faz a predição e calcula o score
                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probabilidades[:, 1] * 100).astype(int)

                        # Salva o DataFrame completo no session_state
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)
                        # --- FIM DA LÓGICA CORRIGIDA ---
                    else:
                        st.error("Não foi possível buscar detalhes dos candidatos.")

    if not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'nome_candidato', 'score']].copy()
        df_para_editar['selecionar'] = False
        
        df_editado = st.data_editor(
            df_para_editar,
            column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": None, "nome_candidato": "Nome do Candidato", "score": st.column_config.ProgressColumn("Score (%)", min_value=0, max_value=100)},
            hide_index=True, use_container_width=True
        )

        if st.button("Confirmar Seleção para Entrevista"):
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
        
        nomes_candidatos = {c['codigo_candidato']: c['nome_candidato'] for c in st.session_state.candidatos_para_entrevista}
        id_candidato_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x])
        # O restante da lógica da tab2 continua...

# A tab3 permanece a mesma
with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    # Lógica da tab3 continua...
