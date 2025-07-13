import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import re
import json
import shap
import matplotlib.pyplot as plt

# Importa as funções dos módulos
import utils
import ml_logic

# --- Configuração da Página ---
st.set_page_config(
    page_title="Decision - Assistente de Recrutamento IA",
    page_icon="✨",
    layout="wide"
)

# --- Funções de IA Generativa ---
# Nota: Para um projeto maior, estas funções poderiam ser movidas para um módulo próprio (ex: gemini_logic.py)
def analisar_competencias_vaga(competencias_texto, api_key):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Analise a descrição de competências de uma vaga de TI e extraia as informações em formato JSON. Descrição: "{competencias_texto}"\nSeu objetivo é identificar:\n1. `obrigatorias`: Uma lista das 5 competências técnicas mais essenciais.\n2. `desejaveis`: Uma lista de outras competências.\n3. `sinonimos`: Para cada competência obrigatória, gere uma lista de 2-3 sinônimos ou tecnologias relacionadas.\nRetorne APENAS o objeto JSON."""
        response = model.generate_content(prompt)
        json_response = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_response)
    except Exception as e:
        st.error(f"Erro na IA ao analisar competências: {e}")
        return None

def calcular_score_hibrido(candidato_texto, competencias_analisadas):
    if not competencias_analisadas or not isinstance(candidato_texto, str): return 0
    score = 0
    candidato_texto_lower = candidato_texto.lower()
    for comp, sinonimos in competencias_analisadas.get('sinonimos', {}).items():
        if comp.lower() in candidato_texto_lower: score += 10
        for s in sinonimos:
            if s.lower() in candidato_texto_lower:
                score += 5; break
    for comp in competencias_analisadas.get('desejaveis', []):
        if comp.lower() in candidato_texto_lower: score += 3
    return score

def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    if not api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.\n**Vaga:** {vaga.get('titulo_vaga', 'N/A')}\n**Candidato:** {candidato.get('nome', 'N/A')}\n**Transcrição:**\n{historico_chat}\n**Sua Tarefa:** Gere um relatório estruturado em markdown com: ### Relatório Final de Entrevista, 1. Score Geral (0 a 10), 2. Pontos Fortes, 3. Pontos de Atenção, 4. Recomendação Final ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado")."""
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
    st.error("Falha na preparação dos dados. Verifique os arquivos e a conexão. A aplicação será interrompida.")
    st.stop()

vagas_data_dict = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data_dict = utils.carregar_prospects()
df_vagas_ui = utils.carregar_vagas()

# --- Interface Principal ---
st.title("✨ Assistente de Recrutamento da Decision")
st.markdown("Bem-vindo ao assistente híbrido de IA: **Machine Learning** para matching e **IA Generativa** para entrevistas e análises.")

# --- Sidebar e Treinamento do Modelo ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password", help="Necessária para as funcionalidades de IA Generativa.")
    
    st.markdown("---")
    st.header("Motor de Machine Learning")
    modelo_match, X_train_data = ml_logic.treinar_modelo_matching(vagas_data_dict, prospects_data_dict)
    
    if modelo_match:
        st.session_state.modelo_match = modelo_match
        st.session_state.X_train_data = X_train_data # Salva os dados de treino para o SHAP
        st.session_state.ml_mode_available = True
        if 'model_performance' in st.session_state:
             st.success(st.session_state.model_performance)
        else:
             st.success("Modelo de Matching carregado.")
    else:
        st.session_state.ml_mode_available = False
        st.warning("Fallback: Usando IA Generativa para o matching.")

# --- Inicialização do Session State ---
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = pd.DataFrame()

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    if st.session_state.get('ml_mode_available', False):
        st.header("Matching com Machine Learning")
        opcoes_vagas_ml = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas_ui.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas_ml.keys()), format_func=lambda x: opcoes_vagas_ml[x], key="ml_vaga_select")

        if st.button("Analisar Candidatos com Machine Learning", type="primary"):
            vaga_selecionada_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
            texto_vaga_selecionada = vaga_selecionada_data['perfil_vaga_texto']
            candidatos_prospect_ids = [p['codigo'] for p in prospects_data_dict.get(codigo_vaga_selecionada, {}).get('prospects', [])]

            if not candidatos_prospect_ids:
                st.warning("Nenhum candidato (prospect) encontrado para esta vaga no histórico.")
            else:
                with st.spinner("Analisando candidatos com o modelo de ML..."):
                    df_detalhes = utils.buscar_detalhes_candidatos(candidatos_prospect_ids)
                    if not df_detalhes.empty:
                        # Prepara os dados para o formato que o modelo espera
                        df_detalhes['texto_completo'] = texto_vaga_selecionada + ' ' + df_detalhes['candidato_texto_completo']
                        
                        modelo = st.session_state.modelo_match
                        probabilidades = modelo.predict_proba(df_detalhes[['texto_completo']])
                        df_detalhes['score'] = (probabilidades[:, 1] * 100).astype(int)
                        
                        st.session_state.df_analise_resultado = df_detalhes.sort_values(by='score', ascending=False).head(20)
    else: # Modo Fallback com IA Generativa
        st.header("Matching com IA Generativa (Modo de Fallback)")
        # ... (código do modo fallback omitido para brevidade, pode ser colado da sua versão anterior)

    if not st.session_state.df_analise_resultado.empty:
        st.subheader("Candidatos Recomendados")
        df_para_editar = st.session_state.df_analise_resultado[['nome', 'score']].copy()
        df_para_editar['selecionar'] = False
        
        df_editado = st.data_editor(df_para_editar, column_config={
            "selecionar": st.column_config.CheckboxColumn("Selecionar p/ Entrevista", default=False),
            "score": st.column_config.ProgressColumn("Score de Match (%)", min_value=0, max_value=100)
        }, hide_index=True, use_container_width=True)

        if st.button("Confirmar Seleção para Entrevista"):
            selecionados_df = df_editado[df_editado['selecionar']]
            if not selecionados_df.empty:
                df_selecionados_completo = pd.merge(selecionados_df, st.session_state.df_analise_resultado, on=['nome', 'score'])
                st.session_state.candidatos_para_entrevista = df_selecionados_completo.to_dict('records')
                vaga_data = df_vagas_ui[df_vagas_ui['codigo_vaga'] == codigo_vaga_selecionada].iloc[0].to_dict()
                st.session_state.vaga_selecionada = vaga_data
                st.success(f"{len(selecionados_df)} candidato(s) movido(s) para a aba de entrevistas!")
                time.sleep(1); st.rerun()
            else: st.warning("Nenhum candidato selecionado.")

        # --- IMPLEMENTAÇÃO DO SHAP (XAI) ---
        with st.expander("🔍 Entenda a Pontuação do Melhor Candidato (Análise SHAP)"):
            try:
                modelo = st.session_state.modelo_match
                df_resultado = st.session_state.df_analise_resultado

                # Extrai o pré-processador e o classificador do pipeline
                preprocessor = modelo.named_steps['preprocessor']
                classifier = modelo.named_steps['clf']

                # Usa o SHAP TreeExplainer que é otimizado para modelos de árvore
                explainer = shap.TreeExplainer(classifier)
                
                # Pega o melhor candidato como exemplo
                melhor_candidato_dados = df_resultado.head(1)
                nome_candidato = melhor_candidato_dados['nome'].iloc[0]

                # Transforma os dados do melhor candidato usando o pré-processador treinado
                dados_transformados = preprocessor.transform(melhor_candidato_dados[['texto_completo']])
                
                # Calcula os valores SHAP para a classe positiva (1)
                shap_values = explainer.shap_values(dados_transformados)[1]
                
                st.write(f"Análise para **{nome_candidato}**:")
                st.write("Principais palavras que influenciaram na pontuação (vermelho aumenta, azul diminui).")
                
                # Gera o gráfico SHAP
                fig, ax = plt.subplots()
                shap.force_plot(
                    explainer.expected_value[1], 
                    shap_values, 
                    feature_names=preprocessor.get_feature_names_out(),
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                st.warning(f"Não foi possível gerar a análise SHAP. Erro: {e}")

# As abas 2 e 3 podem ser mantidas como na sua última versão, pois a lógica delas não depende diretamente das alterações de ML.
# O código delas foi omitido aqui para focar nas melhorias implementadas.

with tab2:
    st.header("Agente 2: Condução das Entrevistas")
    st.info("A lógica desta aba foi omitida para brevidade, mas pode ser colada da sua versão anterior.")
    # Cole aqui a lógica da 'tab2' do seu app_v2.py
    
with tab3:
    st.header("Agente 3: Análise Final Comparativa")
    st.info("A lógica desta aba foi omitida para brevidade, mas pode ser colada da sua versão anterior.")
    # Cole aqui a lógica da 'tab3' do seu app_v2.py
