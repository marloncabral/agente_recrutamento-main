import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os
import requests
from pathlib import Path
import sklearn
import utils

# --- SEÇÃO DE CONFIGURAÇÃO E DOWNLOAD AUTÔNOMO ---
print("Iniciando processo de treinamento do modelo...")
print(f"Versão do scikit-learn utilizada para treinamento: {sklearn.__version__}")

DATA_DIR = Path("./data")
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"
MODELO_FILENAME = "modelo_recrutamento.joblib"

def baixar_arquivo(url, nome_arquivo, is_large=False):
    """Função de download que usa 'print' em vez de comandos do Streamlit."""
    if os.path.exists(nome_arquivo):
        print(f"Arquivo '{nome_arquivo.name}' já existe. Pulando download.")
        return True
    
    print(f"Baixando '{nome_arquivo.name}' de {url}...")
    try:
        response = requests.get(url, stream=is_large)
        response.raise_for_status()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(nome_arquivo, 'wb') as f:
            if is_large:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            else:
                f.write(response.content)
