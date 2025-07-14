import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import re
import json
import shap
import matplotlib.pyplot as plt

import utils
import ml_logic

# --- Configuração da Página ---
st.set_page_config(
    page_title="Decision - Assistente de Recrutamento IA",
    page_icon="✨",
    layout="wide"
)

# --- Funções de IA Generativa ---
def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.\n**Vaga:** {vaga.get('titulo_vaga', 'N/A')}\n**Candidato:** {candidato.get('nome_candidato', 'N/A')}\n**Transcrição:**\n{historico_chat}\n**Sua Tarefa:** Gere um relatório estruturado em markdown com: ### Relatório Final de Entrevista, 1. Score Geral (0 a 10), 2. Pontos Fortes, 3. Pontos de Atenção, 4. Recomendação Final ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado")."""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Ocorreu um erro ao gerar o relatório: {e}"

def gerar_analise_comparativa(vaga, relatorios, api_key):
    if not api_key: return "Erro: Chave de API do Google não configurada."
    cliente = vaga.get('cliente', 'empresa contratante')
    prompt = f"""Você é um Diretor de Recrutamento da **Decision**. Sua tarefa é criar um parecer final para apresentar ao seu cliente, a empresa **'{cliente}'**.\nAnalise os relatórios de entrevista dos finalistas para a vaga de **{vaga.get('titulo_vaga', 'N/A')}**.\n**Relatórios dos Finalistas:**\n---\n{relatorios}\n---\n**Sua Tarefa:**\n1. Crie um ranking dos candidatos, do mais recomendado para o menos.\n2. Escreva um parecer final direcionado ao cliente ('{cliente}'), justificando por que você recomenda a contratação do candidato ideal."""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Ocorreu um erro ao gerar a análise comparativa: {e}"

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

            with st.spinner('Etapa 2 de 3: Treinando o modelo... (Pode levar um momento)'):
                pipeline, X_test, y_test = ml_logic.etapa_2_treinar_modelo(df_treino)
            st.success("Etapa 2 de 3: Modelo treinado!")
            
            with st.spinner('Etapa 3 de 3: Avaliando e finalizando...'):
                performance_string = ml_logic.etapa_3_avaliar_e_finalizar(pipeline, X_test, y_test)
            st.success("Etapa 3 de 3: Avaliação concluída!")

            if pipeline:
                st.session_state.modelo_match = pipeline
                st.session_state.X_train_data_for_shap = X_test # Usamos X_test para o SHAP
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

# --- LÓGICA DA ABA 1: MATCHING ---
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

# --- LÓGICA DA ABA 2: ENTREVISTAS ---
with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba de Matching para selecionar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        
        nomes_candidatos = {c['codigo_candidato']: c['nome_candidato'] for c in st.session_state.candidatos_para_entrevista}
        id_candidato_selecionado = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x])
        candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato_selecionado][0]

        if id_candidato_selecionado not in st.session_state.messages:
            st.session_state.messages[id_candidato_selecionado] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{candidato_atual['nome_candidato']}**."}]
        
        chat_container = st.container(height=400)
        for message in st.session_state.messages[id_candidato_selecionado]:
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Digite a resposta do candidato..."):
            st.session_state.messages[id_candidato_selecionado].append({"role": "user", "content": prompt})
            # Lógica para gerar a próxima pergunta da IA...
            st.rerun()
        
        if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para {candidato_atual['nome_candidato']}"):
            with st.spinner("Analisando entrevista e gerando relatório..."):
                historico_final = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[id_candidato_selecionado]])
                relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final, google_api_key)
                
                if vaga_atual['codigo_vaga'] not in st.session_state.relatorios_finais:
                    st.session_state.relatorios_finais[vaga_atual['codigo_vaga']] = {}
                st.session_state.relatorios_finais[vaga_atual['codigo_vaga']][id_candidato_selecionado] = relatorio
                st.success("Relatório gerado!")
                st.markdown(relatorio)

# --- LÓGICA DA ABA 3: ANÁLISE FINAL ---
with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    if not st.session_state.vaga_selecionada:
        st.info("Nenhuma vaga em processo de análise. Comece pela aba 'Matching'.")
    else:
        codigo_vaga_atual = st.session_state.vaga_selecionada.get('codigo_vaga')
        relatorios_vaga_atual = st.session_state.relatorios_finais.get(codigo_vaga_atual, {})

        if not relatorios_vaga_atual:
            st.info("Nenhum relatório de entrevista foi gerado para esta vaga ainda.")
        else:
            st.subheader(f"Finalistas para a vaga: {st.session_state.vaga_selecionada.get('titulo_vaga')}")
            for id_candidato, relatorio in relatorios_vaga_atual.items():
                # Encontra o nome do candidato na lista de entrevista
                nome_cand = "Candidato"
                for cand_data in st.session_state.candidatos_para_entrevista:
                    if cand_data['codigo_candidato'] == id_candidato:
                        nome_cand = cand_data['nome_candidato']
                        break
                with st.expander(f"Ver relatório de {nome_cand}"):
                    st.markdown(relatorio)

            if len(relatorios_vaga_atual) >= 2:
                if st.button("Gerar Análise Comparativa Final com IA", type="primary"):
                    with st.spinner("IA está analisando todos os finalistas..."):
                        todos_relatorios = "\n\n---\n\n".join(relatorios_vaga_atual.values())
                        analise_final = gerar_analise_comparativa(st.session_state.vaga_selecionada, todos_relatorios, google_api_key)
                        st.subheader("Parecer Final do Assistente de IA")
                        st.markdown(analise_final)
            else:
                st.info("Você precisa finalizar pelo menos duas entrevistas para gerar uma análise comparativa.")
