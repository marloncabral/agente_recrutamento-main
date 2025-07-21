import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json
import shap
import matplotlib.pyplot as plt

# --- FUNÇÕES DE CARREGAMENTO E IA ---
@st.cache_resource
def carregar_modelo_treinado():
    """Carrega o pipeline de ML a partir do arquivo .joblib."""
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado. Execute o script de treino e faça o upload.")
        st.stop()

@st.cache_resource
def carregar_explicador_shap():
    """Carrega o explicador SHAP a partir do arquivo .joblib."""
    try:
        explainer = joblib.load("shap_explainer.joblib")
        return explainer
    except FileNotFoundError:
        st.error("Arquivo 'shap_explainer.joblib' não encontrado. Execute o script de treino e faça o upload.")
        st.stop()

@st.cache_data
def carregar_candidatos_completos():
    """Carrega e prepara a base de candidatos de forma robusta."""
    data = []
    with open(utils.NDJSON_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(data)
    df_normalized = pd.json_normalize(df.to_dict('records'), sep='_')
    coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
    if coluna_nome in df_normalized.columns:
        df_normalized.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
    else:
        df_normalized['nome_candidato'] = 'Nome não encontrado'
    df_normalized['codigo_candidato'] = df_normalized['codigo_candidato'].astype(str)
    return df_normalized

def exibir_explicacao_shap(explainer, preprocessor, texto_candidato):
    """Gera e exibe um gráfico de cascata (waterfall) do SHAP."""
    try:
        # 1. Transforma o texto do candidato usando o pré-processador do pipeline
        texto_transformado = preprocessor.transform(pd.DataFrame([texto_candidato], columns=['texto_completo']))
        
        # 2. Calcula os valores SHAP para essa instância específica
        shap_values = explainer(texto_transformado)
        
        # 3. Gera e exibe o gráfico de cascata
        st.subheader("Análise de Contribuição das Palavras-Chave")
        st.markdown("Este gráfico mostra como as principais palavras-chave (features) impactaram o score final do candidato, partindo de um score base.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=14, show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig) # Fecha a figura para liberar memória

    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar a explicação SHAP: {e}")


def gerar_proxima_pergunta(vaga, candidato, historico_chat):
    """Formula a próxima pergunta da entrevista usando a IA Generativa."""
    prompt = f"""
    Você é um entrevistador de IA da empresa Decision. Sua tarefa é conduzir uma entrevista focada e eficiente.
    Com base no histórico da conversa, formule a PRÓXIMA pergunta para o candidato.
    - **Não repita perguntas.**
    - **Seja conciso e direto.**
    - **Após 5 ou 6 perguntas, finalize a entrevista de forma cordial, agradecendo o candidato.**

    **Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Candidato:** {candidato.get('nome_candidato', 'N/A')}
    **Resumo do Candidato:** {candidato.get('candidato_texto_completo', '')}
    **Histórico da Entrevista:**
    {historico_chat}

    **Sua Ação:** Formule a próxima pergunta ou finalize a entrevista.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Ocorreu um erro na comunicação com a IA: {e}")
        return "Peço desculpas, tive um problema de comunicação. Podemos tentar novamente?"

def gerar_relatorio_final(vaga, candidato, historico_chat):
    """Gera um relatório final estruturado da entrevista."""
    prompt = f"""
    Você é um especialista em recrutamento da Decision. Sua tarefa é analisar a transcrição de uma entrevista e gerar um relatório final conciso e objetivo.

    **Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Candidato:** {candidato.get('nome_candidato', 'N/A')}
    **Transcrição da Entrevista:**
    {historico_chat}

    **Sua Tarefa:** Gere um relatório em formato markdown com a seguinte estrutura:
    ### Relatório Final de Entrevista
    1.  **Score Geral (0 a 10):** Uma nota geral de adequação do candidato à vaga.
    2.  **Pontos Fortes:** Liste 2 ou 3 pontos positivos principais observados.
    3.  **Pontos de Atenção:** Liste 1 ou 2 pontos que requerem atenção ou desenvolvimento.
    4.  **Recomendação Final:** Classifique como "Recomendado", "Recomendado com Ressalvas" ou "Não Recomendado".
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar o relatório: {e}")
        return "Falha ao gerar o relatório."

def gerar_analise_comparativa(vaga, relatorios):
    """Gera um parecer final comparando os candidatos finalistas."""
    cliente = vaga.get('cliente', 'empresa contratante')
    prompt = f"""
    Você é um Diretor de Recrutamento da Decision. Sua tarefa é criar um parecer final para apresentar ao cliente '{cliente}'.
    Analise os relatórios dos candidatos finalistas para a vaga de **{vaga.get('titulo_vaga', 'N/A')}**.

    **Relatórios dos Finalistas:**
    {relatorios}

    **Sua Tarefa:** Crie um parecer final em formato markdown. O parecer deve incluir:
    1.  **Ranking dos Candidatos:** Uma lista ordenada do mais recomendado ao menos recomendado.
    2.  **Justificativa da Recomendação Principal:** Um parágrafo explicando por que o candidato número 1 é a melhor escolha para a vaga e para o cliente.
    3.  **Considerações Adicionais:** Breves comentários sobre os outros candidatos, if relevante.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar a análise comparativa: {e}")
        return "Falha ao gerar a análise comparativa."

# --- CONFIGURAÇÃO INICIAL DO APP ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")

if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos arquivos de dados. A aplicação não pode continuar.")
    st.stop()

# Carrega todos os dados, o modelo e o explicador
df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
df_applicants_completo = carregar_candidatos_completos()
modelo_match = carregar_modelo_treinado()
shap_explainer = carregar_explicador_shap()

# Verifica se todos os componentes essenciais foram carregados
dados_ok = all(obj is not None for obj in [df_vagas_ui, df_applicants_completo, prospects_data_dict, modelo_match, shap_explainer])

if dados_ok:
    st.title("✨ Assistente de Recrutamento da Decision")
    
    with st.sidebar:
        st.header("Configuração Essencial")
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
            st.success("API do Google Gemini configurada!")
        else:
            st.error("Chave de API do Google Gemini não encontrada.")
            st.stop()

        st.markdown("---")
        st.header("Status dos Modelos")
        st.success("Modelo de Matching ✔️")
        st.success("Explicador SHAP ✔️")

    # Inicializa o session_state
    if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()
    if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
    if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
    if "messages" not in st.session_state: st.session_state.messages = {}
    if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}

    tab1, tab2, tab3 = st.tabs(["Agente 1: Matching e Análise", "Agente 2: Entrevistas", "Análise Final"])

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
            st.markdown("Marque os candidatos que deseja mover para a fase de entrevistas.")
            
            df_resultados = st.session_state.df_analise_resultado.copy()
            
            df_resultados['selecionar'] = False
            colunas_para_mostrar = ['selecionar', 'codigo_candidato', 'score']
            df_editado = st.data_editor(
                df_resultados[colunas_para_mostrar], 
                column_config={
                    "selecionar": st.column_config.CheckboxColumn("Selecionar"), 
                    "codigo_candidato": "ID do Candidato", 
                    "score": st.column_config.ProgressColumn("Score (%)", min_value=0, max_value=100)
                }, 
                hide_index=True,
                key="data_editor_candidatos"
            )

            st.subheader("Análise Detalhada do Score")
            ids_para_analise = df_resultados['codigo_candidato'].tolist()
            id_candidato_selecionado = st.selectbox("Selecione um candidato para entender seu score:", options=ids_para_analise)
            
            if id_candidato_selecionado:
                candidato_data = df_resultados[df_resultados['codigo_candidato'] == id_candidato_selecionado].iloc[0]
                exibir_explicacao_shap(
                    shap_explainer, 
                    modelo_match.named_steps['preprocessor'], 
                    candidato_data['texto_completo']
                )

            if st.button("Confirmar para Entrevista"):
                selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
                if selecionados:
                    df_final = st.session_state.df_analise_resultado[st.session_state.df_analise_resultado['codigo_candidato'].isin(selecionados)]
                    st.session_state.candidatos_para_entrevista = df_final.to_dict('records')
                    st.session_state.vaga_selecionada = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                    st.success(f"{len(selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                    time.sleep(1)
                    st.rerun()

    with tab2:
        st.header("Agente 2: Condução das Entrevistas")
        if not st.session_state.candidatos_para_entrevista:
            st.info("Nenhum candidato selecionado. Volte para a aba 'Matching' para selecionar.")
        else:
            vaga_atual = st.session_state.vaga_selecionada
            st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
            
            opcoes = {c['codigo_candidato']: f"ID: {c['codigo_candidato']} ({c.get('nome_candidato', 'N/A')})" for c in st.session_state.candidatos_para_entrevista}
            id_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(opcoes.keys()), format_func=lambda x: opcoes[x])
            candidato_atual = next((c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_selecionado), None)

            if candidato_atual:
                if id_selecionado not in st.session_state.messages:
                    st.session_state.messages[id_selecionado] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA da Decision. Estou pronto para iniciar a entrevista com **{candidato_atual.get('nome_candidato', 'o candidato')}**. Podemos começar?"}]

                for message in st.session_state.messages[id_selecionado]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                if prompt := st.chat_input("Digite a resposta do candidato..."):
                    st.session_state.messages[id_selecionado].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.spinner("IA está formulando a próxima pergunta..."):
                        historico_chat = "\n".join([f"{'Entrevistador' if m['role'] == 'assistant' else 'Candidato'}: {m['content']}" for m in st.session_state.messages[id_selecionado]])
                        proxima_pergunta = gerar_proxima_pergunta(vaga_atual, candidato_atual, historico_chat)
                        st.session_state.messages[id_selecionado].append({"role": "assistant", "content": proxima_pergunta})
                    st.rerun()

                if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para ID {id_selecionado}"):
                    with st.spinner("Gerando relatório..."):
                        historico_final = "\n".join([f"{'Entrevistador' if m['role'] == 'assistant' else 'Candidato'}: {m['content']}" for m in st.session_state.messages[id_selecionado]])
                        relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final)
                        
                        if vaga_atual['codigo_vaga'] not in st.session_state.relatorios_finais:
                            st.session_state.relatorios_finais[vaga_atual['codigo_vaga']] = {}
                        st.session_state.relatorios_finais[vaga_atual['codigo_vaga']][id_selecionado] = relatorio
                        
                        st.success("Relatório gerado! Verifique a aba 'Análise Final'.")
                        with st.expander("Ver relatório gerado", expanded=True):
                            st.markdown(relatorio)

    with tab3:
        st.header("Agente 3: Análise Final Comparativa")
        vaga_selecionada = st.session_state.get('vaga_selecionada', {})
        if not vaga_selecionada:
            st.info("Selecione uma vaga e finalize entrevistas na aba 'Entrevistas' para poder fazer a análise comparativa.")
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
                            todos_relatorios = "\n\n---\n\n".join(
                                f"Relatório do Candidato {mapa_nomes.get(id_cand, id_cand)}:\n{rel}" 
                                for id_cand, rel in relatorios_vaga_atual.items()
                            )
                            analise_final = gerar_analise_comparativa(vaga_selecionada, todos_relatorios)
                            st.subheader("Parecer Final do Assistente de IA")
                            st.markdown(analise_final)
else:
    st.error("Falha ao carregar os dados ou o modelo. Verifique os arquivos e a configuração.")

