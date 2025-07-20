import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import shap
import matplotlib.pyplot as plt

# --- FUNÇÃO PARA CARREGAR O MODELO PRÉ-TREINADO ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado. Execute 'train.py' e envie o arquivo para o repositório.")
        return None

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")
utils.preparar_dados_candidatos()
df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)

st.title("✨ Assistente de Recrutamento da Decision")

# --- Sidebar e CARREGAMENTO RÁPIDO do Modelo ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
    if google_api_key: genai.configure(api_key=google_api_key)
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
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if 'modelo_match' in st.session_state:
        st.header("Matching com Machine Learning")
        opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x])

        if st.button("Analisar Candidatos", type="primary"):
            with st.spinner("Analisando candidatos..."):
                prospects = prospects_data_dict.get(codigo_vaga, {}).get('prospects', [])
                if prospects:
                    ids = [p['codigo'] for p in prospects]
                    df_detalhes = utils.buscar_detalhes_candidatos(ids)
                    if not df_detalhes.empty:
                        vaga_texto = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga].iloc[0]['perfil_vaga_texto']
                        df_detalhes.rename(columns={'nome': 'nome_candidato'}, inplace=True)
                        df_detalhes['nome_candidato'].fillna('Não cadastrado', inplace=True)
                        df_detalhes['texto_completo'] = vaga_texto + ' ' + df_detalhes['candidato_texto_completo'].fillna('')
                        
                        modelo = st.session_state.modelo_match
                        probs = modelo.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probs[:, 1] * 100).astype(int)
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)
                else:
                    st.warning("Nenhum candidato (prospect) encontrado para esta vaga.")

    if not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'nome_candidato', 'score']].copy()
        df_para_editar['selecionar'] = False
        df_editado = st.data_editor(df_para_editar, column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": None, "nome_candidato": "Nome", "score": st.column_config.ProgressColumn("Score (%)")}, hide_index=True)

        if st.button("Confirmar Seleção para Entrevista"):
            selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
            if selecionados:
                df_final = st.session_state.df_analise_resultado[st.session_state.df_analise_resultado['codigo_candidato'].isin(selecionados)]
                st.session_state.candidatos_para_entrevista = df_final.to_dict('records')
                st.success(f"{len(selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                time.sleep(1); st.rerun()

        with st.expander("🔍 Entenda a Pontuação do Melhor Candidato (Análise SHAP)"):
            try:
                modelo = st.session_state.modelo_match
                df_resultado = st.session_state.df_analise_resultado
                melhor_candidato = df_resultado.head(1)
                
                preprocessor = modelo.named_steps['preprocessor']
                classifier = modelo.named_steps['clf']
                
                explainer = shap.LinearExplainer(classifier, preprocessor.transform(df_resultado[['texto_completo']]))
                shap_values = explainer.shap_values(melhor_candidato[['texto_completo']])
                
                st.write(f"Análise para **{melhor_candidato['nome_candidato'].iloc[0]}**:")
                st.write("Palavras que mais influenciaram na pontuação (vermelho aumenta, azul diminui).")
                
                shap.force_plot(explainer.expected_value, shap_values, melhor_candidato[['texto_completo']].iloc[0], matplotlib=True, show=False)
                st.pyplot(bbox_inches='tight')
                plt.close()
            except Exception as e:
                st.warning(f"Não foi possível gerar a análise SHAP. Erro: {e}")

with tab2:
    # Cole seu código da tab2 aqui
    pass

with tab3:
    # Cole seu código da tab3 aqui
    pass
