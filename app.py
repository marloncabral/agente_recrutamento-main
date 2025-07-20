import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json
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
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado.")
        return None

# --- FUNÇÃO PARA CARREGAR A BASE COMPLETA DE CANDIDATOS ---
@st.cache_data
def carregar_candidatos_completos():
    """Carrega e prepara a base de candidatos de forma robusta."""
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

# --- FUNÇÕES DE IA GENERATIVA ---
def gerar_proxima_pergunta(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Chave de API do Google Gemini não configurada."
    prompt = f"""
    Você é um entrevistador de IA da Decision. Sua tarefa é conduzir uma entrevista concisa, fazendo UMA PERGUNTA DE CADA VEZ.
    **Regra Principal:** Formule a PRÓXIMA pergunta com base no histórico. Não repita perguntas. Se já fez 5-6 perguntas, finalize a entrevista agradecendo o candidato.
    **Contexto da Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Contexto do Candidato:** {candidato.get('nome_candidato', 'N/A')} | {candidato.get('candidato_texto_completo', '')}
    **Histórico da Entrevista até agora:**
    {historico_chat}
    **Sua Ação:** Formule a próxima pergunta para o candidato.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro na IA: {e}"

def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Chave de API não configurada."
    prompt = f"""
    Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.
    **Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Candidato:** {candidato.get('nome_candidato', 'N/A')}
    **Transcrição:**\n{historico_chat}
    **Sua Tarefa:** Gere um relatório estruturado em markdown com: ### Relatório Final de Entrevista, 1. Score Geral (0 a 10), 2. Pontos Fortes, 3. Pontos de Atenção, 4. Recomendação Final ("Recomendado", "Recomendado com Ressalvas" ou "Não Recomendado").
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar o relatório: {e}"

# --- Configuração da Página e Carregamento de Dados ---
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
    if "messages" not in st.session_state: st.session_state.messages = {}
    if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}
    if "vaga_selecionada" not in st.session_state: st.session_state.vaga_selecionada = {}

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
            st.subheader("Candidatos Recomendados")
            df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'nome_candidato', 'score']].copy()
            df_para_editar['selecionar'] = False
            df_editado = st.data_editor(df_para_editar, column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": None, "nome_candidato": "Nome do Candidato", "score": st.column_config.ProgressColumn("Score (%)")}, hide_index=True)
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
                    melhor_candidato_texto = df_resultado.head(1)[['texto_completo']]
                    
                    preprocessor = modelo_match.named_steps['preprocessor']
                    classifier = modelo_match.named_steps['clf']
                    
                    explainer = shap.Explainer(classifier, preprocessor.transform(df_resultado[['texto_completo']]))
                    shap_values = explainer(preprocessor.transform(melhor_candidato_texto))
                    
                    st.write(f"Análise para o Candidato ID: **{df_resultado.head(1)['codigo_candidato'].iloc[0]}**")
                    st.write("Palavras que mais influenciaram na pontuação (vermelho aumenta, azul diminui).")
                    
                    shap.plots.text(shap_values)
                    st.pyplot(bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    st.warning(f"Não foi possível gerar a análise SHAP: {e}")

    with tab2:
        st.header("Agente 2: Condução das Entrevistas")
        if not st.session_state.candidatos_para_entrevista:
            st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar e confirmar.")
        else:
            vaga_atual = st.session_state.vaga_selecionada
            st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
            
            opcoes_entrevista = {c['codigo_candidato']: f"ID: {c['codigo_candidato']} ({c.get('nome_candidato', 'N/A')})" for c in st.session_state.candidatos_para_entrevista}
            id_candidato_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(opcoes_entrevista.keys()), format_func=lambda x: opcoes_entrevista[x])
            candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato_selecionado][0]

            if id_candidato_selecionado not in st.session_state.messages:
                st.session_state.messages[id_candidato_selecionado] = [
                    {"role": "assistant", "content": f"Olá! Sou o assistente de IA da Decision. Estou pronto para iniciar a entrevista com **{candidato_atual.get('nome_candidato', 'o candidato')}**. Podemos começar?"}
                ]

            for message in st.session_state.messages[id_candidato_selecionado]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Digite a resposta do candidato aqui..."):
                st.session_state.messages[id_candidato_selecionado].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("IA está formulando a próxima pergunta..."):
                    historico_formatado = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[id_candidato_selecionado]])
                    proxima_pergunta = gerar_proxima_pergunta(vaga_atual, candidato_atual, historico_formatado, google_api_key)
                    
                    with st.chat_message("assistant"):
                        st.markdown(proxima_pergunta)
                    st.session_state.messages[id_candidato_selecionado].append({"role": "assistant", "content": proxima_pergunta})
            
            if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para {candidato_atual.get('nome_candidato', 'este candidato')}"):
                with st.spinner("Analisando entrevista e gerando relatório..."):
                    historico_final = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[id_candidato_selecionado]])
                    relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final, google_api_key)
                    if vaga_atual['codigo_vaga'] not in st.session_state.relatorios_finais:
                        st.session_state.relatorios_finais[vaga_atual['codigo_vaga']] = {}
                    st.session_state.relatorios_finais[vaga_atual['codigo_vaga']][id_candidato_selecionado] = relatorio
                    st.success("Relatório gerado! Verifique a aba 'Análise Final'.")
                    st.markdown(relatorio)

    with tab3:
        st.header("Agente 3: Análise Final Comparativa")
        # A lógica da tab3 continua aqui...
