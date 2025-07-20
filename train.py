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

# --- INÍCIO DA CORREÇÃO ---
# Define as URLs dos arquivos de dados
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
# --- FIM DA CORREÇÃO ---

# Define os caminhos dos arquivos locais
DATA_DIR = Path("./data")
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
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
        print(f"Download de '{nome_arquivo.name}' concluído.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao baixar '{nome_arquivo.name}': {e}")
        return False

# Garante que todos os dados sejam baixados antes de continuar
print("\n--- Etapa 0: Verificação e Download dos Dados ---")
if not all([baixar_arquivo(VAGAS_JSON_URL, VAGAS_FILENAME),
            baixar_arquivo(PROSPECTS_JSON_URL, PROSPECTS_FILENAME),
            baixar_arquivo(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True)]):
    exit("Falha no download dos arquivos. Abortando.")

if not os.path.exists(NDJSON_FILENAME):
    print(f"\nConvertendo '{RAW_APPLICANTS_FILENAME.name}' para formato otimizado NDJSON...")
    try:
        with open(RAW_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
            for codigo, candidato_data in data.items():
                candidato_data['codigo_candidato'] = codigo
                json.dump(candidato_data, f_out)
                f_out.write('\n')
        print("Conversão para NDJSON concluída com sucesso.")
    except Exception as e:
        print(f"ERRO: Falha ao converter arquivo: {e}")
        exit()

# --- INÍCIO DO PROCESSO DE TREINAMENTO ---

print("\n--- Etapa 1: Carregando e Preparando os Dados ---")
# Carrega os dados locais
with open(VAGAS_FILENAME, 'r', encoding='utf-8') as f:
    vagas_data = json.load(f)
with open(PROSPECTS_FILENAME, 'r', encoding='utf-8') as f:
    prospects_data = json.load(f)

# Criar DataFrame Mestre
lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
df_vagas = pd.json_normalize(lista_vagas, sep='_')
lista_prospects = []
for vaga_id, data in prospects_data.items():
    for prospect in data.get('prospects', []):
        lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': prospect.get('codigo'), 'status_final': prospect.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)
ids_necessarios = df_prospects['codigo_candidato'].dropna().unique().tolist()
df_applicants_details = utils.buscar_detalhes_candidatos(ids_necessarios)

df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
if not df_applicants_details.empty:
    df_applicants_details['codigo_candidato'] = df_applicants_details['codigo_candidato'].astype(str)
    if 'nome' in df_applicants_details.columns:
        df_applicants_details.rename(columns={'nome': 'nome_candidato'}, inplace=True)

df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
if not df_applicants_details.empty:
    df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')

# Feature Engineering
colunas_perfil_vaga = [col for col in df_mestre.columns if col.startswith('perfil_vaga_')]
for col in colunas_perfil_vaga: df_mestre[col] = df_mestre[col].fillna('').astype(str)
df_mestre['texto_vaga_combinado'] = df_mestre[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)
df_mestre['texto_candidato_combinado'] = df_mestre['candidato_texto_completo'].fillna('')
df_mestre['texto_completo'] = df_mestre['texto_vaga_combinado'] + ' ' + df_mestre['texto_candidato_combinado']

# Target com as novas regras
positivos_keywords = [
    'contratado', 'aprovado', 'documentação',
    'encaminhado ao requisitante', 
    'contratado pela decision', 
    'contratado como hunting'
]
df_mestre['status_final_lower'] = df_mestre['status_final'].astype(str).str.lower()
df_mestre['target'] = df_mestre['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)
df_treino = df_mestre[df_mestre['status_final'] != 'N/A'].dropna(subset=['texto_completo']).copy()

print(f"Total de {len(df_treino)} registros válidos para o treinamento.")
print(f"Distribuição do Alvo: \n{df_treino['target'].value_counts(normalize=True)}")

# Treinar o Modelo
print("\n--- Etapa 2: Treinando o Modelo de Machine Learning ---")
features = ['texto_completo']
target = 'target'
X = df_treino[features]
y = df_treino[target]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer(transformers=[('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2)), 'texto_completo')], remainder='drop')
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
])
pipeline.fit(X_train, y_train)

# Salvar o Modelo
print("\n--- Etapa 3: Salvando o Modelo Treinado ---")
joblib.dump(pipeline, MODELO_FILENAME)
print(f"\nTreinamento concluído! Modelo salvo como '{MODELO_FILENAME}'.")
print("Faça o upload deste arquivo para o seu repositório no GitHub.")
