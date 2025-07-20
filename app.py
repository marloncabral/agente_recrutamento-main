import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def carregar_modelo_treinado():
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado.")
        return None

@st.cache_data
def carregar_candidatos_completos():
    data = []
    with open(utils.NDJSON_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: continue
    df = pd.DataFrame(data)
    df_normalized = pd.json_normalize(df.to_dict('records'), sep='_')
    coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
    if coluna_nome in df_normalized.columns:
        df_normalized.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
    else:
        df_normalized['nome_candidato'] = 'Nome não encontrado'
    df_normalized['codigo_candidato'] = df_normalized['codigo_candidato'].astype(str)
    return df_normalized

# --- Configuração da Página ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")
if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
df_applicants_completo = carregar_candidatos_completos()
modelo_match = carregar_modelo_treinado()

if all(obj is not None for obj in [df_vagas_ui, prospects_data_dict, df_applicants_completo, modelo_match]):
    st.title("✨ Assistente de Recrutamento da Decision")

    with st.sidebar:
        st.header("Configuração Essencial")
        google_api_key = st.text_input("Chave de API do Google Gemini", type="password")
        if google_api_key: genai.configure(api_key=google_api_key)
        st.markdown("---")
        st.header("Motor de Machine Learning")
        st.success("Modelo de Matching carregado!")

    if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
    if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []

    tab1, tab2, tab3 = st.tabs(["Agente 1: Matching", "Agente 2: Entrevistas", "Análise Final"])

    with tab1:
        st.header("Matching com Machine Learning")
        opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} ({row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x])

        if st.button("Analisar Candidatos", type="primary"):
            with st.spinner("Analisando candidatos..."):
                prospects = prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])
                if prospects:
                    ids = [str(p['codigo']) for p in prospects]
                    df_detalhes = df_applicants_completo[df_applicants_completo['codigo_candidato'].isin(ids)].copy()
                    if not df_detalhes.empty:
                        vaga_texto = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]['perfil_vaga_texto']
                        text_cols = ['informacoes_profissionais_resumo_profissional', 'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en']
                        for col in text_cols:
                            if col not in df_detalhes.columns: df_detalhes[col] = ''
                        df_detalhes['candidato_texto_completo'] = df_detalhes[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
                        df_detalhes['texto_completo'] = vaga_texto + ' ' + df_detalhes['candidato_texto_completo']
                        probs = modelo_match.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probs[:, 1] * 100).astype(int)
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)

        if not st.session_state.df_analise_resultado.empty:
            st.subheader("Candidatos Recomendados (Anônimo)")
            df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'score']].copy()
            df_para_editar['selecionar'] = False
            df_editado = st.data_editor(df_para_editar, column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": "ID do Candidato", "score": st.column_config.ProgressColumn("Score (%)")}, hide_index=True)
            if st.button("Confirmar para Entrevista"):
                selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
                if selecionados:
                    df_final = st.session_state.df_analise_resultado[st.session_state.df_analise_resultado['codigo_candidato'].isin(selecionados)]
                    st.session_state.candidatos_para_entrevista = df_final.to_dict('records')
                    st.session_state.vaga_selecionada = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                    st.success(f"{len(selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                    time.sleep(1); st.rerun()

            with st.expander("🔍 Entenda a Pontuação do Melhor Candidato (Análise SHAP)"):
                try:
                    df_resultado = st.session_state.df_analise_resultado
                    melhor_candidato = df_resultado.head(1)
                    
                    preprocessor = modelo_match.named_steps['preprocessor']
                    classifier = modelo_match.named_steps['clf']
                    
                    # Cria o explainer com o classificador e os dados de texto já transformados pelo TF-IDF
                    explainer = shap.LinearExplainer(classifier, preprocessor.transform(df_resultado[['texto_completo']]))
                    shap_values = explainer.shap_values(melhor_candidato[['texto_completo']])
                    
                    st.write(f"Análise para o Candidato ID: **{melhor_candidato['codigo_candidato'].iloc[0]}**")
                    st.write("Palavras que mais influenciaram na pontuação (vermelho aumenta, azul diminui).")
                    
                    shap.force_plot(explainer.expected_value, shap_values, melhor_candidato['texto_completo'].iloc[0], matplotlib=True, show=False)
                    st.pyplot(bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    st.warning(f"Não foi possível gerar a análise SHAP: {e}")

    with tab2:
        st.header("Agente 2: Condução das Entrevistas")
        if not st.session_state.candidatos_para_entrevista:
            st.info("Nenhum candidato selecionado.")
        else:
            vaga_atual = st.session_state.vaga_selecionada
            st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
            opcoes = {c['codigo_candidato']: f"ID: {c['codigo_candidato']} ({c.get('nome_candidato', 'N/A')})" for c in st.session_state.candidatos_para_entrevista}
            id_selecionado = st.selectbox("Selecione o candidato:", options=list(opcoes.keys()), format_func=lambda x: opcoes[x])
            # Lógica completa do chatbot aqui...

    with tab3:
        st.header("Agente 3: Análise Final Comparativa")
        # Lógica completa da análise final aqui...
