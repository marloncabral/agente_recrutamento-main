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

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

vagas_data_dict = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
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
    
    with st.spinner("Carregando modelo de IA..."):
        modelo_match = carregar_modelo_treinado()

    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.session_state.ml_mode_available = True
        st.success("Modelo de Matching carregado!")
    else:
        st.session_state.ml_mode_available = False
        st.error("Motor de ML indisponível.")

# --- Inicialização do Session State ---
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}

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
                    df_detalhes_candidatos = utils.buscar_detalhes_candidatos(ids_candidatos)

                    if not df_detalhes_candidatos.empty:
                        vaga_selecionada_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                        perfil_vaga_texto = vaga_selecionada_data['perfil_vaga_texto']

                        if 'nome' in df_detalhes_candidatos.columns:
                            df_detalhes_candidatos.rename(columns={'nome': 'nome_candidato'}, inplace=True)
                        
                        df_detalhes_candidatos['nome_candidato'].fillna('Candidato não cadastrado', inplace=True)
                        
                        # --- CORREÇÃO FINAL ---
                        # Cria a coluna 'texto_completo' ANTES de passar para o modelo
                        df_detalhes_candidatos['texto_completo'] = perfil_vaga_texto + ' ' + df_detalhes_candidatos['candidato_texto_completo'].fillna('')
                        
                        modelo = st.session_state.modelo_match
                        
                        # Garante que o DataFrame não está vazio antes de prever
                        if not df_detalhes_candidatos.empty:
                            probabilidades = modelo.predict_proba(df_detalhes_candidatos[['texto_completo']])
                            df_detalhes_candidatos['score'] = (probabilidades[:, 1] * 100).astype(int)
                            st.session_state.df_analise_resultado = df_detalhes_candidatos.sort_values(by='score', ascending=False).head(20)
                        else:
                            st.warning("Nenhum candidato com dados suficientes para análise.")
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

# O restante das abas permanece o mesmo
with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar.")
    else:
        # A lógica da tab2 continua aqui...
        pass

with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    # A lógica da tab3 continua aqui...
    pass
