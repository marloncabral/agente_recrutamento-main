import streamlit as st
import pandas as pd
import google.generativeai as genai
import time

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
    if google_api_key:
        genai.configure(api_key=google_api_key)
        
    st.markdown("---")
    st.header("Motor de Machine Learning")

    if 'modelo_match' not in st.session_state:
        try:
            with st.spinner('Etapa 1 de 3: Preparando e limpando os dados...'):
                df_treino = ml_logic.etapa_1_preparar_dados(vagas_data_dict, prospects_data_dict)
            st.success("Etapa 1 de 3: Dados preparados!")

            with st.spinner('Etapa 2 de 3: Treinando o modelo...'):
                pipeline, X_test, y_test = ml_logic.etapa_2_treinar_modelo(df_treino)
            st.success("Etapa 2 de 3: Modelo treinado!")
            
            with st.spinner('Etapa 3 de 3: Avaliando e finalizando...'):
                performance_string = ml_logic.etapa_3_avaliar_e_finalizar(pipeline, X_test, y_test)
            st.success("Etapa 3 de 3: Avaliação concluída!")

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
    
    if st.session_state.get('ml_mode_available', False):
        st.info(st.session_state.get('model_performance', 'Modelo carregado.'))

# --- Inicialização do Session State ---
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
if "messages" not in st.session_state: st.session_state.messages = {}
if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}

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

with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        
        nomes_candidatos = {c['codigo_candidato']: c.get('nome_candidato', 'Nome não encontrado') for c in st.session_state.candidatos_para_entrevista}
        id_candidato_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x])
        # O restante do seu código para a tab2...

with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    # O restante do seu código para a tab3...
