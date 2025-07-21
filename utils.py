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

def preparar_dados_candidatos():
    """Garante que todos os arquivos de dados necessários estejam disponíveis."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baixar_arquivo_se_nao_existir(VAGAS_JSON_URL, VAGAS_FILENAME)
    baixar_arquivo_se_nao_existir(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)
    if not baixar_arquivo_se_nao_existir(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True):
        st.error("Não foi possível baixar o arquivo principal de candidatos.")
        return False
    if not os.path.exists(NDJSON_FILENAME):
        st.info(f"Primeiro uso: Convertendo '{RAW_APPLICANTS_FILENAME.name}'...")
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

def buscar_detalhes_candidatos(codigos_candidatos):
    """Busca detalhes de candidatos usando DuckDB com a query corrigida."""
    if not isinstance(codigos_candidatos, list) or not codigos_candidatos: return pd.DataFrame()
    codigos_str = ", ".join([f"'{str(c)}'" for c in codigos_candidatos])
    
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
        print(f"Erro ao consultar candidatos com DuckDB: {e}")
        return pd.DataFrame()

@st.cache_data
def criar_mapeamento_id_nome():
    """
    Cria um dicionário que mapeia 'codigo_candidato' para 'nome_candidato'
    para ser usado nas etapas finais do processo.
    """
    mapeamento = {}
    try:
        with open(NDJSON_FILENAME, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    codigo = record.get('codigo_candidato')
                    nome = record.get('informacoes_pessoais', {}).get('dados_pessoais', {}).get('nome_completo')
                    if codigo and nome:
                        mapeamento[str(codigo)] = nome
                except json.JSONDecodeError:
                    continue
        return mapeamento
    except FileNotFoundError:
        st.error(f"Arquivo de mapeamento '{NDJSON_FILENAME}' não encontrado.")
        return {}
