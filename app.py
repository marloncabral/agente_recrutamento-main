import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json

# --- FUNÇÕES DE CARREGAMENTO E IA ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado.")
        return None

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

def gerar_proxima_pergunta(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Chave de API do Google Gemini não configurada."
    prompt = f"""
    Você é um entrevistador de IA. Formule a PRÓXIMA pergunta com base no histórico. Não repita perguntas. Se já fez 5-6 perguntas, finalize a entrevista.
    Vaga: {vaga.get('titulo_vaga', 'N/A')}
    Candidato: {candidato.get('nome_candidato', 'N/A')} | {candidato.get('candidato_texto_completo', '')}
    Histórico: {historico_chat}
    Sua Ação: Formule a próxima pergunta.
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

def gerar_analise_comparativa(vaga, relatorios, api_key):
    if not api_key: return "Chave de API não configurada."
    cliente = vaga.get('cliente', 'empresa contratante')
    prompt = f"""
    Você é um Diretor de Recrutamento da Decision. Crie um parecer final para apresentar ao cliente '{cliente}'.
    Analise os relatórios dos finalistas para a vaga de {vaga.get('titulo_vaga', 'N/A')}.
    **Relatórios:**\n{relatorios}\n
    **Sua Tarefa:** Crie um ranking e escreva um parecer final justificando a recomendação do candidato ideal.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar a análise comparativa: {e}"

# --- CONFIGURAÇÃO INICIAL DO APP ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos arquivos de dados. A aplicação será interrompida.")
    st.stop()

df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
df_applicants_completo = carregar_candidatos_completos()
modelo_match = carregar_modelo_treinado()

dados_ok = all(obj is not None and not (isinstance(obj, pd.DataFrame) and obj.empty) for obj in [df_vagas_ui, df_applicants_completo]) and prospects_data_dict and modelo_match

if dados_ok:
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
    if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
    if "messages" not in st.session_state: st.session_state.messages = {}
    if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}

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

    with tab2:
        st.header("Agente 2: Condução das Entrevistas")
        if not st.session_state.candidatos_para_entrevista:
            st.info("Nenhum candidato selecionado.")
        else:
            vaga_atual = st.session_state.vaga_selecionada
            st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
            
            opcoes = {c['codigo_candidato']: f"ID: {c['codigo_candidato']} ({c.get('nome_candidato', 'N/A')})" for c in st.session_state.candidatos_para_entrevista}
            id_selecionado = st.selectbox("Selecione o candidato:", options=list(opcoes.keys()), format_func=lambda x: opcoes[x])
            candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_selecionado][0]

            if id_selecionado not in st.session_state.messages:
                st.session_state.messages[id_selecionado] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{candidato_atual.get('nome_candidato', 'o candidato')}**. Podemos começar?"}]

            for message in st.session_state.messages[id_selecionado]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Digite a resposta do candidato..."):
                st.session_state.messages[id_selecionado].append({"role": "user", "content": prompt})
                with st.spinner("IA está formulando a próxima pergunta..."):
                    historico_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[id_selecionado]])
                    proxima_pergunta = gerar_proxima_pergunta(vaga_atual, candidato_atual, historico_chat, google_api_key)
                    st.session_state.messages[id_selecionado].append({"role": "assistant", "content": proxima_pergunta})
                st.rerun()

            if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para ID {id_selecionado}"):
                with st.spinner("Gerando relatório..."):
                    historico_final = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[id_selecionado]])
                    relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final, google_api_key)
                    if vaga_atual['codigo_vaga'] not in st.session_state.relatorios_finais:
                        st.session_state.relatorios_finais[vaga_atual['codigo_vaga']] = {}
                    st.session_state.relatorios_finais[vaga_atual['codigo_vaga']][id_selecionado] = relatorio
                    st.success("Relatório gerado! Verifique a aba 'Análise Final'.")
                    st.markdown(relatorio)

    with tab3:
        st.header("Agente 3: Análise Final Comparativa")
        vaga_selecionada = st.session_state.get('vaga_selecionada', {})
        if not vaga_selecionada:
            st.info("Selecione uma vaga e finalize entrevistas na Etapa 2 para poder fazer a análise comparativa.")
        else:
            codigo_vaga_atual = vaga_selecionada.get('codigo_vaga')
            relatorios_vaga_atual = st.session_state.relatorios_finais.get(codigo_vaga_atual, {})
            if not relatorios_vaga_atual:
                st.info("Nenhum relatório de entrevista foi gerado para esta vaga ainda.")
            else:
                st.subheader(f"Finalistas para a vaga: {vaga_selecionada.get('titulo_vaga')}")
                mapa_nomes = {c['codigo_candidato']: c.get('nome_candidato', 'N/A') for c in st.session_state.candidatos_para_entrevista}
                for id_candidato, relatorio in relatorios_vaga_atual.items():
                    nome_cand = mapa_nomes.get(id_candidato, 'ID não encontrado')
                    with st.expander(f"Ver relatório de ID: {id_candidato} ({nome_cand})"):
                        st.markdown(relatorio)
                if len(relatorios_vaga_atual) >= 2:
                    if st.button("Gerar Análise Comparativa Final com IA", type="primary"):
                        with st.spinner("IA está analisando todos os finalistas..."):
                            todos_relatorios = "\n\n---\n\n".join(relatorios_vaga_atual.values())
                            analise_final = gerar_analise_comparativa(vaga_selecionada, todos_relatorios, google_api_key)
                            st.subheader("Parecer Final do Assistente de IA")
                            st.markdown(analise_final)
