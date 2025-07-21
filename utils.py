import streamlit as st
import pandas as pd
import json
import os
import requests
from pathlib import Path

# --- Constantes de Arquivos e URLs ---
DATA_DIR = Path("./data")
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"

def preparar_dados_candidatos():
    """Garante que todos os arquivos de dados necessários estejam disponíveis."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baixar_arquivo_se_nao_existir(VAGAS_JSON_URL, VAGAS_FILENAME)
    baixar_arquivo_se_nao_existir(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)
    if not baixar_arquivo_se_nao_existir(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True):
        st.error("Não foi possível baixar o arquivo principal de candidatos.")
        return False

    if not os.path.exists(NDJSON_FILENAME):
        st.info(f"Primeiro uso: Convertendo '{RAW_APPLICANTS_FILENAME.name}' para um formato otimizado...")
        with st.spinner("Isso pode levar um momento..."):
            try:
                with open(RAW_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in: data = json.load(f_in)
                with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
                    for codigo, candidato_data in data.items():
                        candidato_data['codigo_candidato'] = codigo
                        json.dump(candidato_data, f_out)
                        f_out.write('\n')
                st.success("Arquivo de dados otimizado!")
            except Exception as e:
                st.error(f"Falha ao converter o arquivo JSON: {e}")
                return False
    return True

def baixar_arquivo_se_nao_existir(url, nome_arquivo, is_large=False):
    """Baixa um arquivo da URL se ele não existir."""
    if os.path.exists(nome_arquivo): return True
    st.info(f"Arquivo '{nome_arquivo.name}' não encontrado. Baixando...")
    try:
        with st.spinner(f"Baixando {nome_arquivo.name}..."):
            response = requests.get(url, stream=is_large)
            response.raise_for_status()
            with open(nome_arquivo, 'wb') as f:
                if is_large:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                else: f.write(response.content)
            st.success(f"Arquivo '{nome_arquivo.name}' baixado!")
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o arquivo '{nome_arquivo.name}': {e}.")
        return False

@st.cache_data
def carregar_json(caminho_arquivo):
    """Carrega um arquivo JSON de forma segura."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data
def carregar_vagas():
    """Carrega e padroniza os dados das vagas."""
    vagas_data = carregar_json(VAGAS_FILENAME)
    if vagas_data is None: return pd.DataFrame()
    vagas_lista = []
    for codigo, dados in vagas_data.items():
        info_basicas = dados.get('informacoes_basicas', {})
        vaga_info = {'codigo_vaga': codigo, 'titulo_vaga': info_basicas.get('titulo_vaga', 'N/A'), 'cliente': info_basicas.get('cliente', 'N/A'), 'perfil_vaga_texto': json.dumps(dados.get('perfil_vaga', {}))}
        vagas_lista.append(vaga_info)
    return pd.DataFrame(vagas_lista)

# --- NOVA FUNÇÃO OTIMIZADA ---
def buscar_detalhes_candidatos_por_id(ids_necessarios: list):
    """
    Busca detalhes de candidatos específicos no arquivo NDJSON,
    carregando apenas os registros necessários para economizar memória.
    """
    if not os.path.exists(NDJSON_FILENAME):
        return pd.DataFrame()

    ids_necessarios_set = set(ids_necessarios)
    candidatos_encontrados = []
    
    with open(NDJSON_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                candidato = json.loads(line)
                if candidato.get('codigo_candidato') in ids_necessarios_set:
                    candidatos_encontrados.append(candidato)
            except json.JSONDecodeError:
                continue
    
    if not candidatos_encontrados:
        return pd.DataFrame()

    df = pd.json_normalize(candidatos_encontrados, sep='_')
    coluna_nome = 'informacoes_pessoais_dados_pessoais_nome_completo'
    if coluna_nome in df.columns:
        df.rename(columns={coluna_nome: 'nome_candidato'}, inplace=True)
    else:
        df['nome_candidato'] = 'Nome não encontrado'
    df['codigo_candidato'] = df['codigo_candidato'].astype(str)
    return df
