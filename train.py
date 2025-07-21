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
import shap # <-- 1. Importação do SHAP

print("Iniciando processo de treinamento do modelo e do explicador SHAP...")
print(f"Versão do scikit-learn: {sklearn.__version__}")
print(f"Versão do SHAP: {shap.__version__}")


# --- Constantes de Arquivos ---
DATA_DIR = Path("./data")
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"
MODELO_FILENAME = "modelo_recrutamento.joblib"
EXPLAINER_FILENAME = "shap_explainer.joblib" # <-- 2. Novo arquivo para o explicador

def baixar_arquivo(url, nome_arquivo, is_large=False):
    if os.path.exists(nome_arquivo):
        print(f"Arquivo '{nome_arquivo.name}' já existe.")
        return True
    print(f"Baixando '{nome_arquivo.name}'...")
    try:
        response = requests.get(url, stream=is_large)
        response.raise_for_status()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(nome_arquivo, 'wb') as f:
            if is_large:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            else: f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERRO ao baixar '{nome_arquivo.name}': {e}")
        return False

print("\n--- Etapa 0: Verificação e Download dos Dados ---")
if not all([baixar_arquivo(VAGAS_JSON_URL, VAGAS_FILENAME),
            baixar_arquivo(PROSPECTS_JSON_URL, PROSPECTS_FILENAME),
            baixar_arquivo(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True)]):
    exit("Falha no download dos arquivos. Abortando.")

if not os.path.exists(NDJSON_FILENAME):
    print(f"\nConvertendo '{RAW_APPLICANTS_FILENAME.name}' para NDJSON...")
    with open(RAW_APPLICANTS_FILENAME, 'r', encoding='utf-8') as f_in: data = json.load(f_in)
    with open(NDJSON_FILENAME, 'w', encoding='utf-8') as f_out:
        for c, d in data.items(): d['codigo_candidato'] = c; json.dump(d, f_out); f_out.write('\n')

print("\n--- Etapa 1: Preparando o DataFrame de Treinamento ---")
# Carregando os dados
with open(VAGAS_FILENAME, 'r', encoding='utf-8') as f: vagas_data = json.load(f)
with open(PROSPECTS_FILENAME, 'r', encoding='utf-8') as f: prospects_data = json.load(f)

# Lendo candidatos do arquivo NDJSON
candidatos_lista = []
with open(NDJSON_FILENAME, 'r', encoding='utf-8') as f:
    for line in f:
        candidatos_lista.append(json.loads(line))
df_applicants = pd.json_normalize(candidatos_lista, sep='_')
df_applicants.rename(columns={'informacoes_pessoais_dados_pessoais_nome_completo': 'nome_candidato'}, inplace=True)

# Processando vagas e prospects
lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
df_vagas = pd.json_normalize(lista_vagas, sep='_')
lista_prospects = []
for vaga_id, data in prospects_data.items():
    for p in data.get('prospects', []): lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': p.get('codigo'), 'status_final': p.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)

# Criando o DataFrame mestre
df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
df_applicants['codigo_candidato'] = df_applicants['codigo_candidato'].astype(str)
df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
df_mestre = pd.merge(df_mestre, df_applicants, on='codigo_candidato', how='left')

# Engenharia de Features de Texto
text_cols_vaga = [c for c in df_mestre.columns if c.startswith('perfil_vaga_')]
for col in text_cols_vaga: df_mestre[col] = df_mestre[col].fillna('').astype(str)
df_mestre['texto_vaga'] = df_mestre[text_cols_vaga].apply(lambda x: ' '.join(x), axis=1)

text_cols_candidato = ['informacoes_profissionais_resumo_profissional', 'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en']
for col in text_cols_candidato:
    if col not in df_mestre.columns: df_mestre[col] = ''
df_mestre['candidato_texto_completo'] = df_mestre[text_cols_candidato].fillna('').astype(str).agg(' '.join, axis=1)

df_mestre['texto_completo'] = df_mestre['texto_vaga'] + ' ' + df_mestre['candidato_texto_completo']

# Criando a variável alvo (target)
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

# --- ETAPA 3: CRIAÇÃO DO EXPLICADOR SHAP ---
print("\n--- Etapa 3: Criando o Explicador SHAP ---")

# 1. Transformar os dados de treino com o pré-processador do pipeline
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)

# 2. Criar o explicador SHAP usando o modelo treinado e os dados transformados
# Usamos o `masker` para obter os nomes das features (palavras do TF-IDF)
masker = shap.maskers.Text(pipeline.named_steps['preprocessor'].named_transformers_['tfidf'].build_analyzer())
explainer = shap.Explainer(pipeline.named_steps['clf'], X_train_transformed, feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out())

print("Explicador SHAP criado com sucesso.")


# --- ETAPA 4: SALVANDO O MODELO E O EXPLICADOR ---
print("\n--- Etapa 4: Salvando os Artefatos ---")
joblib.dump(pipeline, MODELO_FILENAME)
joblib.dump(explainer, EXPLAINER_FILENAME) # <-- Salvando o explicador
print(f"\nTreinamento concluído!")
print(f"Modelo salvo como: '{MODELO_FILENAME}'")
print(f"Explicador SHAP salvo como: '{EXPLAINER_FILENAME}'")
print("\nIMPORTANTE: Faça o upload de AMBOS os arquivos para o seu repositório no GitHub.")

