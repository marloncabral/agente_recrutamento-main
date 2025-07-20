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
        
        # Caminho correto para o nome do candidato
        coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
        if coluna_nome in df_normalized.columns:
            df_normalized.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
        else:
            df_normalized['nome_candidato'] = 'Nome não encontrado' # Garante que a coluna sempre exista
            
        df_normalized['codigo_candidato'] = df_normalized['codigo_candidato'].astype(str)
        return df_normalized
    except Exception as e:
        st.error(f"Erro ao carregar a base de dados de candidatos: {e}")
        return pd.DataFrame()

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

# --- Inicialização do Session State ---
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if 'modelo_match' in st.session_state:
        st.header("Matching com Machine Learning")
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x])

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            with st.spinner("Analisando candidatos..."):
                prospects_da_vaga = prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])
                if not prospects_da_vaga:
                    st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")
                else:
                    ids_candidatos = [str(p['codigo']) for p in prospects_da_vaga]
                    df_detalhes = df_applicants_completo[df_applicants_completo['codigo_candidato'].isin(ids_candidatos)].copy()

                    if not df_detalhes.empty:
                        vaga_selecionada_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                        perfil_vaga_texto = vaga_selecionada_data['perfil_vaga_texto']
                        
                        text_cols = ['informacoes_profissionais_resumo_profissional', 'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en']
                        for col in text_cols:
                            if col not in df_detalhes.columns: df_detalhes[col] = ''
                        df_detalhes['candidato_texto_completo'] = df_detalhes[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
                        df_detalhes['texto_completo'] = perfil_vaga_texto + ' ' + df_detalhes['candidato_texto_completo']
                        
                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probabilidades[:, 1] * 100).astype(int)
                        
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)

    if not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'score']].copy()
        df_para_editar['selecionar'] = False
        df_editado = st.data_editor(
            df_para_editar,
            column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": "ID do Candidato", "score": st.column_config.ProgressColumn("Score (%)")},
            hide_index=True, use_container_width=True
        )

        if st.button("Confirmar Seleção para Entrevista"):
            codigos_selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
            if codigos_selecionados:
                df_completo = st.session_state.df_analise_resultado
                df_selecionados_final = df_completo[df_completo['codigo_candidato'].isin(codigos_selecionados)]
                st.session_state.candidatos_para_entrevista = df_selecionados_final.to_dict('records')
                st.session_state.vaga_selecionada = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                st.success(f"{len(codigos_selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                time.sleep(1); st.rerun()

with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar e confirmar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        
        # --- CORREÇÃO E IMPLEMENTAÇÃO DA SUA SUGESTÃO ---
        # O menu de seleção agora mostra o ID do candidato, mas o sistema sabe o nome internamente.
        opcoes_entrevista = {c['codigo_candidato']: f"Candidato ID: {c['codigo_candidato']} ({c.get('nome_candidato', 'Nome não encontrado')})" for c in st.session_state.candidatos_para_entrevista}
        id_candidato_selecionado = st.selectbox(
            "Selecione o candidato para entrevistar:", 
            options=list(opcoes_entrevista.keys()), 
            format_func=lambda x: opcoes_entrevista[x]
        )
        # O restante da lógica para conduzir a entrevista...

with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    # A lógica da tab3 continua aqui...
