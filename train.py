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

# Lógica de download simplificada para o script de treino
def baixar(url, nome_arquivo):
    if not os.path.exists(nome_arquivo):
        print(f"Baixando '{nome_arquivo.name}'...")
        r = requests.get(url)
        r.raise_for_status()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(nome_arquivo, 'wb') as f:
            f.write(r.content)
            
print("\n--- Etapa 0: Verificação e Download dos Dados ---")
baixar(VAGAS_JSON_URL, VAGAS_FILENAME)
baixar(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)
baixar(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME)

if not os.path.exists(NDJSON_FILENAME):
    print(f"\nConvertendo '{RAW_APPLICANTS_FILENAME.name}' para NDJSON...")
    with open(RAW_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in: data = json.load(f_in)
    with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
        for c, d in data.items(): d['codigo_candidato'] = c; json.dump(d, f_out); f_out.write('\n')

print("\n--- Etapa 1: Preparando o DataFrame de Treinamento ---")
# Carrega os dados
df_vagas = pd.DataFrame.from_dict(utils.carregar_json(VAGAS_FILENAME), orient='index').reset_index().rename(columns={'index': 'codigo_vaga'})
df_vagas = pd.json_normalize(df_vagas.to_dict('records'), sep='_')
prospects_data = utils.carregar_json(PROSPECTS_FILENAME)
lista_prospects = []
for vaga_id, data in prospects_data.items():
    for p in data.get('prospects', []): lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': p.get('codigo'), 'status_final': p.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)

# Merge
df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
ids_necessarios = df_mestre['codigo_candidato'].dropna().unique().tolist()
df_applicants_details = utils.buscar_detalhes_candidatos(ids_necessarios)
if not df_applicants_details.empty:
    df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')

# Feature Engineering
colunas_vaga = [c for c in df_mestre.columns if c.startswith('perfil_vaga_')]
for col in colunas_vaga: df_mestre[col] = df_mestre[col].fillna('').astype(str)
df_mestre['texto_vaga'] = df_mestre[colunas_vaga].apply(lambda x: ' '.join(x), axis=1)
df_mestre['texto_completo'] = df_mestre['texto_vaga'] + ' ' + df_mestre['candidato_texto_completo'].fillna('')

# Target com as novas regras
positivos_keywords = ['contratado', 'aprovado', 'documentação', 'encaminhado ao requisitante']
df_mestre['status_final_lower'] = df_mestre['status_final'].astype(str).str.lower()
df_mestre['target'] = df_mestre['status_final_lower'].apply(lambda x: 1 if any(k in x for k in positivos_keywords) else 0)
df_treino = df_mestre[df_mestre['status_final'] != 'N/A'].dropna(subset=['texto_completo']).copy()
print(f"Distribuição do Alvo: \n{df_treino['target'].value_counts(normalize=True)}")

print("\n--- Etapa 2: Treinando o Modelo ---")
X = df_treino[['texto_completo']]
y = df_treino['target']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer(transformers=[('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2)), 'texto_completo')], remainder='drop')
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
])
pipeline.fit(X_train, y_train)

print("\n--- Etapa 3: Salvando o Modelo ---")
joblib.dump(pipeline, MODELO_FILENAME)
print(f"\nTreinamento concluído! Modelo salvo como '{MODELO_FILENAME}'. Faça o upload deste arquivo para o GitHub.")
