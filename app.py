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
    if not competencias_analisadas: return 0
    score = 0
    candidato_texto = candidato_texto.lower()
    for comp, sinonimos in competencias_analisadas.get('sinonimos', {}).items():
        if comp.lower() in candidato_texto: score += 10
        for s in sinonimos:
            if s.lower() in candidato_texto:
                score += 5
                break
    for comp in competencias_analisadas.get('desejaveis', []):
        if comp.lower() in candidato_texto: score += 3
    return score

# (As outras funções do Gemini permanecem as mesmas)
def gerar_relatorio_final(vaga, candidato, historico_chat, api_key):
    # ... (código da função original)
    pass

def gerar_analise_comparativa(vaga, relatorios, api_key):
    # ... (código da função original)
    pass

# --- Interface Principal ---
st.title("✨ Assistente de Recrutamento da Decision v2.1")
st.markdown("Bem-vindo ao assistente híbrido de IA: **Machine Learning** para matching e **IA Generativa** para entrevistas.")

# --- Sidebar e Configurações ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Chave de API do Google Gemini", type="password", help="Necessária para os Agentes de Entrevista e Análise.")
    
    st.markdown("---")
    st.header("Motor de Machine Learning")
    # O treinamento agora é tentado automaticamente.
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


# --- Download e Preparação dos Dados Iniciais ---
utils.preparar_dados_candidatos()

# Carrega os dados principais para a UI
df_vagas = utils.carregar_vagas()
prospects_data = utils.carregar_prospects()

# Inicializa o session_state
# ... (código de inicialização do session_state original)

# --- Abas da Aplicação ---
tab1, tab2, tab3 = st.tabs(["Agente 1: Matching de Candidatos", "Agente 2: Entrevistas", "Análise Final"])

with tab1:
    # --- LÓGICA DE EXIBIÇÃO CONDICIONAL ---
    if st.session_state.get('ml_mode_available', False):
        st.header("Matching com Machine Learning")
        st.info("O modelo de ML foi treinado com sucesso e está sendo usado para esta análise.")
        
        opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas.iterrows()}
        codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x], key="ml_vaga_select")

        if st.button("Analisar Candidatos com Machine Learning", type="primary", key="ml_button"):
            # ... (Lógica para usar o modelo de ML, como na versão anterior)
            pass # Cole a lógica do botão da versão anterior aqui
    
    else:
        st.header("Matching com IA Generativa (Modo de Fallback)")
        st.warning("Não foi possível treinar o modelo de Machine Learning devido à falta de dados históricos de feedback. O sistema está usando a IA Generativa para analisar as competências.")

        if not google_api_key:
            st.error("Por favor, insira sua chave de API do Google na barra lateral para usar o modo de fallback.")
        else:
            opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for _, row in df_vagas.iterrows()}
            codigo_vaga_selecionada = st.selectbox("Selecione a vaga para análise:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x], key="gemini_vaga_select")

            if st.button("Analisar Candidatos com IA Generativa", type="primary", key="gemini_button"):
                # Lógica original do Agente 1 que usa o Gemini
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
    
    # A lógica de exibição dos resultados é a mesma para ambos os modos
    if 'df_analise_resultado' in st.session_state and not st.session_state.df_analise_resultado.empty:
        # ... (Cole aqui a lógica do st.data_editor para selecionar candidatos)
        pass

# As abas 2 e 3 permanecem as mesmas
with tab2:
    # ... (código original da tab2)
    pass
with tab3:
    # ... (código original da tab3)
    pass
