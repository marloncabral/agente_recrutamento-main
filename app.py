import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import joblib
import utils
import json

# --- FUNÇÃO PARA CARREGAR O MODELO PRÉ-TREINADO ---
@st.cache_resource
def carregar_modelo_treinado():
    try:
        modelo = joblib.load("modelo_recrutamento.joblib")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_recrutamento.joblib' não encontrado.")
        return None

# --- FUNÇÕES DE IA GENERATIVA ---
def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    # (Mantenha sua função gerar_relatorio_final aqui)
    pass

def gerar_proxima_pergunta(vaga, candidato, historico_chat, api_key):
    # (Mantenha sua função gerar_proxima_pergunta aqui)
    pass

def gerar_analise_comparativa(vaga, relatorios, api_key):
    # (Mantenha sua função gerar_analise_comparativa aqui)
    pass

# --- Configuração da Página ---
st.set_page_config(page_title="Decision - Assistente de Recrutamento IA", page_icon="✨", layout="wide")
if not utils.preparar_dados_candidatos():
    st.error("Falha na preparação dos dados. A aplicação será interrompida.")
    st.stop()

# Carrega os dados e o modelo no início
df_vagas_ui = utils.carregar_vagas()
prospects_data_dict = utils.carregar_json(utils.PROSPECTS_FILENAME)
mapa_id_nome = utils.criar_mapeamento_id_nome()
modelo_match = carregar_modelo_treinado()

dados_ok = all(obj is not None for obj in [df_vagas_ui, prospects_data_dict, mapa_id_nome, modelo_match])

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

    # --- LÓGICA COMPLETA DA ABA 1 ---
    with tab1:
        st.header("Matching com Machine Learning")
        opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} ({row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x])

        if st.button("Analisar Candidatos", type="primary"):
            with st.spinner("Analisando candidatos..."):
                prospects = prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])
                if prospects:
                    ids = [str(p['codigo']) for p in prospects]
                    df_detalhes = utils.buscar_detalhes_candidatos(ids)
                    if not df_detalhes.empty:
                        df_detalhes.rename(columns={'nome': 'nome_candidato'}, inplace=True)
                        vaga_texto = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]['perfil_vaga_texto']
                        df_detalhes['texto_completo'] = vaga_texto + ' ' + df_detalhes['candidato_texto_completo'].fillna('')
                        probs = modelo_match.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probs[:, 1] * 100).astype(int)
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)

        if not st.session_state.df_analise_resultado.empty:
            st.subheader("Candidatos Recomendados")
            df_para_editar = st.session_state.df_analise_resultado[['codigo_candidato', 'nome_candidato', 'score']].copy()
            df_para_editar['selecionar'] = False
            df_editado = st.data_editor(df_para_editar, column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar"), "codigo_candidato": "ID", "nome_candidato": "Nome do Candidato", "score": st.column_config.ProgressColumn("Score (%)")}, hide_index=True)
            if st.button("Confirmar para Entrevista"):
                selecionados = df_editado[df_editado['selecionar']]['codigo_candidato'].tolist()
                if selecionados:
                    df_final = st.session_state.df_analise_resultado[st.session_state.df_analise_resultado['codigo_candidato'].isin(selecionados)]
                    st.session_state.candidatos_para_entrevista = df_final.to_dict('records')
                    st.session_state.vaga_selecionada = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                    st.success(f"{len(selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                    time.sleep(1); st.rerun()

    # --- LÓGICA COMPLETA DA ABA 2 ---
    with tab2:
        st.header("Agente 2: Condução das Entrevistas")
        if not st.session_state.candidatos_para_entrevista:
            st.info("Nenhum candidato selecionado.")
        else:
            vaga_atual = st.session_state.vaga_selecionada
            st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
            
            opcoes = {c['codigo_candidato']: f"ID: {c['codigo_candidato']} ({mapa_id_nome.get(c['codigo_candidato'], 'N/A')})" for c in st.session_state.candidatos_para_entrevista}
            id_selecionado = st.selectbox("Selecione o candidato:", options=list(opcoes.keys()), format_func=lambda x: opcoes[x])
            candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_selecionado][0]

            if id_selecionado not in st.session_state.messages:
                st.session_state.messages[id_selecionado] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{mapa_id_nome.get(id_selecionado, 'o candidato')}**. Podemos começar?"}]

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

            if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para {mapa_id_nome.get(id_selecionado, 'este candidato')}"):
                with st.spinner("Gerando relatório..."):
                    historico_final = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[id_selecionado]])
                    relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final, google_api_key)
                    if vaga_atual['codigo_vaga'] not in st.session_state.relatorios_finais:
                        st.session_state.relatorios_finais[vaga_atual['codigo_vaga']] = {}
                    st.session_state.relatorios_finais[vaga_atual['codigo_vaga']][id_selecionado] = relatorio
                    st.success("Relatório gerado! Verifique a aba 'Análise Final'.")
                    st.markdown(relatorio)

    # --- LÓGICA COMPLETA DA ABA 3 ---
    with tab3:
        st.header("Agente 3: Análise Final Comparativa")
        vaga_selecionada = st.session_state.get('vaga_selecionada', {})
        if not vaga_selecionada:
            st.info("Selecione uma vaga e finalize entrevistas na Etapa 2 para fazer a análise comparativa.")
        else:
            codigo_vaga_atual = vaga_selecionada.get('codigo_vaga')
            relatorios_vaga_atual = st.session_state.relatorios_finais.get(codigo_vaga_atual, {})

            if not relatorios_vaga_atual:
                st.info("Nenhum relatório de entrevista foi gerado para esta vaga ainda.")
            else:
                st.subheader(f"Finalistas para a vaga: {vaga_selecionada.get('titulo_vaga')}")
                for id_candidato, relatorio in relatorios_vaga_atual.items():
                    nome_cand = mapa_id_nome.get(id_candidato, f"ID: {id_candidato}")
                    with st.expander(f"Ver relatório de {nome_cand}"):
                        st.markdown(relatorio)
                if len(relatorios_vaga_atual) >= 2:
                    if st.button("Gerar Análise Comparativa Final com IA", type="primary"):
                        with st.spinner("IA está analisando todos os finalistas..."):
                            todos_relatorios = "\n\n---\n\n".join(relatorios_vaga_atual.values())
                            analise_final = gerar_analise_comparativa(vaga_selecionada, todos_relatorios, google_api_key)
                            st.subheader("Parecer Final do Assistente de IA")
                            st.markdown(analise_final)
