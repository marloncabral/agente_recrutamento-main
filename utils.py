# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import json
import os
import requests
import time
import duckdb

# --- URLs para os arquivos no Hugging Face ---
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/prospects.json"
NDJSON_FILENAME = "applicants_nd.json"
ORIGINAL_APPLICANTS_FILENAME = "applicants.json"
VAGAS_FILENAME = "vagas.json"
PROSPECTS_FILENAME = "prospects.json"

# --- Funções de Preparação e Download de Dados ---

def baixar_arquivo_se_nao_existir(url, nome_arquivo, is_large=False):
    """
    Baixa um arquivo da URL especificada se ele não existir localmente.
    Lida com arquivos grandes (stream) e pequenos.
    """
    if os.path.exists(nome_arquivo):
        return True

    st.info(f"Arquivo '{nome_arquivo}' não encontrado. Baixando do repositório...")
    try:
        with st.spinner(f"Baixando {nome_arquivo}... (Pode levar um momento)"):
            response = requests.get(url, stream=is_large)
            response.raise_for_status()
            with open(nome_arquivo, 'wb') as f:
                if is_large:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                else:
                    f.write(response.content)
        st.success(f"Arquivo '{nome_arquivo}' baixado com sucesso!")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o arquivo '{nome_arquivo}': {e}. Verifique a URL.")
        st.stop()
    return False

def preparar_dados_candidatos():
    """
    Garante que todos os arquivos de dados necessários estejam disponíveis,
    convertendo o 'applicants.json' para o formato otimizado NDJSON na primeira vez.
    """
    # Garante que os arquivos menores (vagas, prospects) existam.
    baixar_arquivo_se_nao_existir(VAGAS_JSON_URL, VAGAS_FILENAME)
    baixar_arquivo_se_nao_existir(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)

    # Lida com o arquivo grande de candidatos
    if os.path.exists(NDJSON_FILENAME):
        return

    # Baixa o arquivo original grande se necessário
    baixar_arquivo_se_nao_existir(APPLICANTS_JSON_URL, ORIGINAL_APPLICANTS_FILENAME, is_large=True)

    # Converte o JSON original para NDJSON para otimizar a leitura
    st.info(f"Primeiro uso: Convertendo '{ORIGINAL_APPLICANTS_FILENAME}' para um formato otimizado...")
    with st.spinner("Isso pode levar um momento, mas só acontecerá uma vez."):
        try:
            with open(ORIGINAL_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)

            with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
                for codigo, candidato_data in data.items():
                    candidato_data['codigo_candidato'] = codigo
                    json.dump(candidato_data, f_out)
                    f_out.write('\n')
            st.success("Arquivo de dados otimizado com sucesso!")
            time.sleep(2)
        except Exception as e:
            st.error(f"Falha ao converter o arquivo JSON: {e}")
            st.stop()

# --- Funções de Carregamento de Dados ---

@st.cache_data
def carregar_vagas():
    """Carrega os dados das vagas a partir do arquivo JSON local."""
    with open(VAGAS_FILENAME, 'r', encoding='utf-8') as f:
        vagas_data = json.load(f)
    vagas_lista = []
    for codigo, dados in vagas_data.items():
        vaga_info = {
            'codigo_vaga': codigo,
            'titulo_vaga': dados.get('informacoes_basicas', {}).get('titulo_vaga', 'N/A'),
            'cliente': dados.get('informacoes_basicas', {}).get('cliente', 'N/A'),
            'perfil_vaga_texto': json.dumps(dados.get('perfil_vaga', {})) # Converte o perfil para texto
        }
        vagas_lista.append(vaga_info)
    return pd.DataFrame(vagas_lista)

@st.cache_data
def carregar_prospects():
    """Carrega os dados dos prospects a partir do arquivo JSON local."""
    with open(PROSPECTS_FILENAME, 'r', encoding='utf-8') as f:
        return json.load(f)

def buscar_detalhes_candidatos(codigos_candidatos):
    """Busca detalhes de uma lista de candidatos usando DuckDB para performance."""
    if not codigos_candidatos:
        return pd.DataFrame()

    codigos_str = ", ".join([f"'{c}'" for c in codigos_candidatos])

    query = f"""
    SELECT
        codigo_candidato,
        informacoes_pessoais ->> 'nome_completo' AS nome,
        (
            (informacoes_profissionais ->> 'resumo_profissional'       ) || ' ' ||
            (informacoes_profissionais ->> 'conhecimentos'             ) || ' ' ||
            (informacoes_profissionais ->> 'area_de_atuacao'           ) || ' ' ||
            (informacoes_profissionais ->> 'nivel_profissional'        ) || ' ' ||
            (formacao_e_idiomas      ->> 'formacao'                    ) || ' ' ||
            (formacao_e_idiomas      ->> 'nivel_ingles'                ) || ' ' ||
            cv_pt || ' ' || cv_en
        ) AS candidato_texto_completo
    FROM read_json_auto('{NDJSON_FILENAME}')
    WHERE codigo_candidato IN ({codigos_str})
    """
    try:
        with duckdb.connect(database=':memory:', read_only=False) as con:
            return con.execute(query).fetchdf()
    except Exception as e:
        st.error(f"Erro ao consultar o arquivo de candidatos com DuckDB: {e}")
        return pd.DataFrame()

@st.cache_data
def carregar_json(caminho_arquivo):
    """Função genérica para carregar um arquivo JSON."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)
