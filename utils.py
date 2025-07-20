import streamlit as st
import pandas as pd
import json
import os
import requests
from pathlib import Path
import duckdb

# --- Constantes de Arquivos e URLs ---
DATA_DIR = Path("./data")
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"

# ... (o resto das funções baixar_arquivo, preparar_dados, etc. permanecem iguais) ...

def preparar_dados_candidatos():
    """Garante que todos os arquivos de dados necessários estejam disponíveis para o app Streamlit."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baixar_arquivo_se_nao_existir(VAGAS_JSON_URL, VAGAS_FILENAME)
    baixar_arquivo_se_nao_existir(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)
    if not baixar_arquivo_se_nao_existir(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True):
        st.error("Não foi possível baixar o arquivo principal de candidatos.")
        return False

    if not os.path.exists(NDJSON_FILENAME):
        st.info(f"Primeiro uso: Convertendo '{RAW_APPLICANTS_FILENAME.name}' para um formato otimizado...")
        with st.spinner("Isso pode levar um momento, mas só acontecerá uma vez."):
            try:
                with open(RAW_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in: data = json.load(f_in)
                with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
                    for codigo, candidato_data in data.items():
                        candidato_data['codigo_candidato'] = codigo
                        json.dump(candidato_data, f_out)
                        f_out.write('\n')
                st.success("Arquivo de dados otimizado com sucesso!")
            except Exception as e:
                st.error(f"Falha ao converter o arquivo JSON: {e}")
                return False
    return True

def baixar_arquivo_se_nao_existir(url, nome_arquivo, is_large=False):
    """Baixa um arquivo da URL especificada se ele não existir localmente."""
    if os.path.exists(nome_arquivo): return True
    st.info(f"Arquivo '{nome_arquivo.name}' não encontrado. Baixando...")
    try:
        with st.spinner(f"Baixando {nome_arquivo.name}..."):
            response = requests.get(url, stream=is_large)
            response.raise_for_status()
            with open(nome_arquivo, 'wb') as f:
                if is_large:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                else:
                    f.write(response.content)
            st.success(f"Arquivo '{nome_arquivo.name}' baixado!")
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o arquivo '{nome_arquivo.name}': {e}.")
        return False

@st.cache_data
def carregar_json(caminho_arquivo):
    """Função genérica e segura para carregar um arquivo JSON."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho_arquivo}"); return None
    except json.JSONDecodeError:
        st.error(f"Erro ao decodificar o JSON do arquivo '{caminho_arquivo}'"); return None

@st.cache_data
def carregar_vagas():
    """Carrega e padroniza os dados das vagas a partir do arquivo JSON local."""
    vagas_data = carregar_json(VAGAS_FILENAME)
    if vagas_data is None: return pd.DataFrame()
    vagas_lista = []
    for codigo, dados in vagas_data.items():
        info_basicas = dados.get('informacoes_basicas', {})
        vaga_info = {'codigo_vaga': codigo, 'titulo_vaga': info_basicas.get('titulo_vaga', 'N/A'), 'cliente': info_basicas.get('cliente', 'N/A'), 'perfil_vaga_texto': json.dumps(dados.get('perfil_vaga', {}))}
        vagas_lista.append(vaga_info)
    return pd.DataFrame(vagas_lista)

def buscar_detalhes_candidatos(codigos_candidatos):
    """Busca detalhes de uma lista de candidatos usando DuckDB com o caminho corrigido para o nome."""
    if not isinstance(codigos_candidatos, list) or not codigos_candidatos:
        return pd.DataFrame()
    
    codigos_str = ", ".join([f"'{str(c)}'" for c in codigos_candidatos])
    
    # --- CORREÇÃO DEFINITIVA DA QUERY ---
    query = f"""
    SELECT
        codigo_candidato,
        informacoes_pessoais -> 'dados_pessoais' ->> 'nome_completo' AS nome,
        CONCAT_WS(' ',
            informacoes_profissionais ->> 'resumo_profissional',
            informacoes_profissionais ->> 'conhecimentos',
            cv_pt,
            cv_en
        ) AS candidato_texto_completo
    FROM read_json_auto('{NDJSON_FILENAME}')
    WHERE codigo_candidato IN ({codigos_str})
    """
    try:
        with duckdb.connect(database=':memory:', read_only=False) as con:
            return con.execute(query).fetchdf()
    except Exception as e:
        print(f"Erro ao consultar o arquivo de candidatos com DuckDB: {e}")
        return pd.DataFrame()
