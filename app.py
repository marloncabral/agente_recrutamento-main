import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import re
import json

# Importa as funções dos novos arquivos
import utils
import ml_logic

# --- Configuração da Página ---
st.set_page_config(
    page_title="Decision - Assistente de Recrutamento IA",
    page_icon="✨",
    layout="wide"
)

# --- ETAPA 1: GARANTIR QUE OS DADOS EXISTAM ---
# Esta é a correção mais importante. A preparação dos dados deve ser a
# primeira coisa a acontecer, antes de qualquer outra lógica.
utils.preparar_dados_candidatos()

# --- Funções dos Agentes de IA (Gemini) ---

def analisar_competencias_vaga(competencias_texto, api_key):
    """Usa a IA Generativa para extrair competências da descrição de uma vaga."""
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analise a descrição de competências de uma vaga de TI e extraia as informações em formato JSON.
        Descrição: "{competencias_texto}"
        Seu objetivo é identificar:
        1.  `obrigatorias`: Uma lista das 5 competências técnicas mais essenciais.
        2.  `desejaveis`: Uma lista de outras competências.
        3.  `sinonimos`: Para cada competência obrigatória, gere uma lista de 2-3 sinônimos ou tecnologias relacionadas.
        Retorne APENAS o objeto JSON.
        """
        response = model.generate_content(prompt)
        json_response = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_response)
    except Exception as e:
        st.error(f"Erro na IA ao analisar competências: {e}")
        return None

def calcular_score_hibrido(candidato_texto, competencias_analisadas):
    """Calcula um score baseado na presença de palavras-chave."""
    if not competencias_analisadas or not isinstance(candidato_texto, str):
        return 0
    score = 0
    candidato_texto_lower = candidato_texto.lower()
    for comp, sinonimos in competencias_analisadas.get('sinonimos', {}).items():
        if comp.lower() in candidato_texto_lower: score += 10
        for s in sinonimos:
            if s.lower() in candidato_texto_lower:
                score += 5
                break
    for comp in competencias_analisadas.get('desejaveis', []):
        if comp.lower() in candidato_texto_lower: score += 3
    return score

def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    """Gera um relatório final de entrevista usando a IA Generativa."""
    if not api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""
    Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.
    **Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Candidato:** {candidato.get('nome', 'N/A')}
    **Transcrição:**\n{historico_chat}
    **Sua Tarefa:** Gere um relatório estruturado.
    **Formato (markdown):**
    ### Relatório Final de Entrevista - {candidato.get('nome', 'N/A')}
    **1. Score Geral:** (0 a 10)
    **2. Pontos Fortes:** (bullet points)
    **3. Pontos de Atenção:** (bullet points)
    **4. Recomendação Final:** ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado", com justificativa)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar o relatório: {e}"

def gerar_analise_comparativa(vaga, relatorios, api_key):
    """Gera uma análise comparativa de múltiplos candidatos para uma vaga."""
    if not api_key: return "Erro: Chave de API do Google não configurada."
    cliente = vaga.get('cliente', 'empresa contratante')
    prompt = f"""
    Você é um Diretor de Recrutamento da **Decision**. Sua tarefa é criar um parecer final para apresentar ao seu cliente, a empresa **'{cliente}'**.
    Analise os relatórios de entrevista dos finalistas para a vaga de **{vaga.get('titulo_vaga', 'N/A')}**.
    **Relatórios dos Finalistas:**
    ---
    {relatorios}
    ---
    **Sua Tarefa:**
    1.  Crie um ranking dos candidatos, do mais recomendado para o menos.
    2.  Escreva um parecer final direcionado ao cliente ('{cliente}'), justificando por que você, como representante da Decision, recomenda a contratação do candidato ideal para a vaga deles. Enfatize como as habilidades e o perfil do candidato escolhido beneficiarão o cliente.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar a análise comparativa: {e}"

# --- Interface Principal ---
st.title("✨ Assistente de Recrutamento da Decision v2.2")
st.markdown("Bem-vindo ao assistente híbrido de IA: **Machine Learning** para matching e **IA Generativa** para entrevistas.")

# --- Sidebar e Configurações ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password", help="Necessária para os Agentes de Entrevista e Análise.")
    
    st.markdown("---")
    st.header("Motor de Machine Learning")
    modelo_match = ml_logic.treinar_modelo_matching()
    
    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.session_state.ml_mode_available = True
        if 'model_performance' in st.session_state:
             st.success(st.session_state.model_performance)
        else:
             st.success("Modelo de Matching carregado.")
    else:
        st.session_state.ml_mode_available = False
        st.warning("Fallback: Usando IA Generativa para o matching.")

# --- ETAPA 2: Carregar os dados para a UI ---
# Agora que temos certeza que os arquivos existem, podemos carregá-los.
df_vagas = utils.carregar_vagas()
prospects_data = utils.carregar_prospects()

# Inicializa o session_state
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if "messages" not in st.session_state: st.session_state.messages = {}
if "question_count" not in st.session_state: st.session_state.question_count = {}
if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if st.session_state.get('ml_mode_available', False):
        st.header("Matching com Machine Learning")
        st.info("O modelo de ML foi treinado com sucesso e está sendo usado para esta análise.")
        
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x], key="ml_vaga_select")

        if st.button("Analisar Candidatos com Machine Learning", type="primary", key="ml_button"):
            vaga_selecionada_data = df_vagas[df_vagas['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
            texto_vaga_selecionada = vaga_selecionada_data['perfil_vaga_texto']
            candidatos_prospect_ids = [p['codigo'] for p in prospects_data.get(codigo_vaga_selecionada, {}).get('prospects', [])]

            if not candidatos_prospect_ids:
                st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")
            else:
                with st.spinner("Buscando candidatos e aplicando o modelo de ML..."):
                    df_detalhes = utils.buscar_detalhes_candidatos(candidatos_prospect_ids)
                    if not df_detalhes.empty:
                        df_detalhes['texto_vaga'] = texto_vaga_selecionada
                        df_detalhes['texto_completo_ml'] = df_detalhes['texto_vaga'] + ' ' + df_detalhes['candidato_texto_completo']
                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes['texto_completo_ml'])
                        df_detalhes['score'] = (probabilidades[:, 1] * 100).astype(int)
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)
                    else:
                        st.error("Não foi possível buscar detalhes dos candidatos.")
    
    else:
        st.header("Matching com IA Generativa (Modo de Fallback)")
        st.warning("Não foi possível treinar o modelo de ML devido à falta de dados históricos. O sistema está usando a IA Generativa para analisar as competências.")

        if not google_api_key:
            st.error("Por favor, insira sua chave de API do Google na barra lateral para usar o modo de fallback.")
        else:
            opcoes_vagas_gemini = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas.iterrows()}
            codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_gemini.keys()), format_func=lambda x: opcoes_vagas_gemini[x], key="gemini_vaga_select")

            if st.button("Analisar Candidatos com IA Generativa", type="primary", key="gemini_button"):
                vaga_selecionada_data = df_vagas[df_vagas['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                competencias_texto = vaga_selecionada_data['perfil_vaga_texto']
                
                with st.spinner("Agente de IA está analisando as competências da vaga..."):
                    competencias_analisadas = analisar_competencias_vaga(competencias_texto, google_api_key)
                
                if competencias_analisadas:
                    st.success("Competências analisadas pela IA!")
                    candidatos_prospect_ids = [p['codigo'] for p in prospects_data.get(codigo_vaga_selecionada, {}).get('prospects', [])]
                    
                    if candidatos_prospect_ids:
                        with st.spinner("Buscando e pontuando candidatos..."):
                            df_detalhes = utils.buscar_detalhes_candidatos(candidatos_prospect_ids)
                            if not df_detalhes.empty:
                                df_detalhes['score'] = df_detalhes['candidato_texto_completo'].apply(lambda x: calcular_score_hibrido(x, competencias_analisadas))
                                st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(10)
                    else:
                        st.warning("Nenhum prospect encontrado para esta vaga.")
    
    if 'df_analise_resultado' in st.session_state and not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        df_para_editar = st.session_state.df_analise_resultado.copy()
        df_para_editar['selecionar'] = False
        
        score_column_config = st.column_config.ProgressColumn("Score de Match (%)", min_value=0, max_value=100) if st.session_state.get('ml_mode_available', False) else st.column_config.NumberColumn("Score")

        df_editado = st.data_editor(
            df_para_editar[['selecionar', 'nome', 'score']],
            column_config={
                "selecionar": st.column_config.CheckboxColumn("Selecionar", default=False),
                "score": score_column_config
            },
            hide_index=True, use_container_width=True, key=f"editor_{codigo_vaga_selecionada}"
        )
        
        if st.button("Confirmar Seleção para Entrevista", key="confirm_selection"):
            selecionados_df = df_editado[df_editado['selecionar']]
            if not selecionados_df.empty:
                df_selecionados_completo = pd.merge(selecionados_df, st.session_state.df_analise_resultado, on=['nome', 'score'])
                st.session_state.candidatos_para_entrevista = df_selecionados_completo.to_dict('records')
                vaga_data = df_vagas[df_vagas['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                st.session_state.vaga_selecionada = vaga_data
                st.session_state.relatorios_finais[codigo_vaga_selecionada] = {}
                st.session_state.df_analise_resultado = pd.DataFrame()
                st.success(f"{len(selecionados_df)} candidato(s) movido(s) para a aba de entrevistas!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Nenhum candidato selecionado.")

with tab2:
    st.header("Condução das Entrevistas com IA Generativa")
    if not google_api_key:
        st.warning("Por favor, insira sua chave de API do Google na barra lateral para usar o Agente 2.")
    elif not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        nomes_candidatos = {c['codigo_candidato']: c['nome'] for c in st.session_state.candidatos_para_entrevista}
        id_candidato = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x], key="interview_select")
        candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato][0]
        
        if id_candidato not in st.session_state.messages:
            st.session_state.messages[id_candidato] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{candidato_atual['nome']}**. Para começar, pode me dizer 'Olá' ou 'Podemos começar'."}]
            st.session_state.question_count[id_candidato] = 0

        chat_container = st.container(height=400)
        for message in st.session_state.messages[id_candidato]:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Digite a resposta do candidato..."):
            st.session_state.messages[id_candidato].append({"role": "user", "content": prompt})
            with st.spinner("Agente 2 está pensando..."):
                historico_formatado = "\n".join([f"{'Recrutador (resposta do candidato)' if m['role'] == 'user' else 'IA (sua pergunta)'}: {m['content']}" for m in st.session_state.messages[id_candidato]])
                question_count = st.session_state.question_count.get(id_candidato, 0)
                
                prompt_ia = f"""
                Você é um entrevistador de IA da Decision. Sua tarefa é conduzir uma entrevista concisa e eficaz, fazendo UMA PERGUNTA DE CADA VEZ.
                **Regras:**
                1. Você está na pergunta {question_count + 1} de um total de 5 a 7 perguntas.
                2. Analise o histórico e o contexto para formular a próxima pergunta lógica.
                3. NÃO FAÇA MAIS DE UMA PERGUNTA.
                4. Se já fez 6 ou 7 perguntas, finalize a entrevista.
                **Contexto Vaga:** {vaga_atual.get('titulo_vaga')} | {vaga_atual.get('perfil_vaga_texto')}
                **Contexto Candidato:** {candidato_atual.get('nome')} | {candidato_atual.get('candidato_texto_completo')}
                **Histórico:** {historico_formatado}
                **Sua próxima ação:**
                """
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt_ia)
                st.session_state.messages[id_candidato].append({"role": "assistant", "content": response.text})
                st.session_state.question_count[id_candidato] += 1
            st.rerun()

        codigo_vaga_atual = vaga_atual.get('codigo_vaga')
        if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para {candidato_atual['nome']}"):
            with st.spinner("Analisando entrevista e gerando relatório..."):
                historico_final = "\n".join([f"{'Candidato' if m['role'] == 'user' else 'Entrevistador'}: {m['content']}" for m in st.session_state.messages[id_candidato]])
                relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final, google_api_key)
                if codigo_vaga_atual not in st.session_state.relatorios_finais:
                    st.session_state.relatorios_finais[codigo_vaga_atual] = {}
                st.session_state.relatorios_finais[codigo_vaga_atual][id_candidato] = relatorio
                st.success("Relatório gerado! Verifique a aba 'Análise Final Comparativa'.")
                st.markdown(relatorio)

with tab3:
    st.header("Análise Final e Decisão")
    if not google_api_key:
        st.warning("Por favor, insira sua chave de API do Google na barra lateral para usar esta funcionalidade.")
    elif not st.session_state.vaga_selecionada:
        st.info("Nenhuma vaga em processo de análise. Comece pela aba de Matching.")
    else:
        codigo_vaga_atual = st.session_state.vaga_selecionada.get('codigo_vaga')
        relatorios_vaga_atual = st.session_state.relatorios_finais.get(codigo_vaga_atual, {})
        if not relatorios_vaga_atual:
            st.info("Nenhum relatório de entrevista foi gerado para esta vaga ainda. Finalize as entrevistas na aba 'Entrevistas'.")
        else:
            st.subheader(f"Finalistas para a vaga: {st.session_state.vaga_selecionada.get('titulo_vaga')}")
            for id_candidato, relatorio in relatorios_vaga_atual.items():
                nome_candidato_lista = [c['nome'] for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato]
                if nome_candidato_lista:
                    nome_candidato = nome_candidato_lista[0]
                    with st.expander(f"Ver relatório de {nome_candidato}"):
                        st.markdown(relatorio)
            
            if len(relatorios_vaga_atual) >= 2:
                if st.button("Gerar Análise Comparativa Final com IA", type="primary"):
                    with st.spinner("IA está analisando todos os finalistas para eleger o melhor..."):
                        todos_relatorios = "\n\n---\n\n".join(relatorios_vaga_atual.values())
                        analise_final = gerar_analise_comparativa(st.session_state.vaga_selecionada, todos_relatorios, google_api_key)
                        st.markdown("---")
                        st.subheader("Parecer Final do Assistente de IA")
                        st.markdown(analise_final)
            else:
                st.info("Você precisa finalizar pelo menos duas entrevistas para gerar uma análise comparativa.")
