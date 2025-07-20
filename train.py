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

print("Iniciando processo de treinamento do modelo...")
print(f"Versão do scikit-learn utilizada para treinamento: {sklearn.__version__}")

# Define os caminhos e URLs
DATA_DIR = Path("./data")
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
MODELO_FILENAME = "modelo_recrutamento.joblib"
# ... (código de download permanece o mesmo) ...

# Carrega os dados
with open(VAGAS_FILENAME, 'r', encoding='utf-8') as f: vagas_data = json.load(f)
with open(PROSPECTS_FILENAME, 'r', encoding='utf-8') as f: prospects_data = json.load(f)

# Cria DataFrame Mestre com a lógica correta de busca de nomes
lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
df_vagas = pd.json_normalize(lista_vagas, sep='_')
lista_prospects = []
for vaga_id, data in prospects_data.items():
    for p in data.get('prospects', []): lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': p.get('codigo'), 'status_final': p.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)
ids_necessarios = df_prospects['codigo_candidato'].dropna().unique().tolist()
df_applicants_details = utils.buscar_detalhes_candidatos(ids_necessarios) # Agora usa a função corrigida

df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
if not df_applicants_details.empty:
    df_applicants_details['codigo_candidato'] = df_applicants_details['codigo_candidato'].astype(str)
    if 'nome' in df_applicants_details.columns:
        df_applicants_details.rename(columns={'nome': 'nome_candidato'}, inplace=True)

df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
if not df_applicants_details.empty:
    df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')

# O restante do script de treinamento permanece o mesmo...
# Feature Engineering, definição de target, treinamento e salvamento do modelo.
