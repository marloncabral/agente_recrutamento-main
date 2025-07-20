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

# --- SEÇÃO DE CONFIGURAÇÃO E DOWNLOAD AUTÔNOMO ---

print("Iniciando processo de treinamento autônomo...")

# Define os caminhos e URLs
DATA_DIR = Path("./data")
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/Postech7/datathon-fiap/resolve/main/prospects.json"
RAW_APPLICANTS_FILENAME = DATA_DIR / "applicants.raw.json"
NDJSON_FILENAME = DATA_DIR / "applicants.nd.json"
VAGAS_FILENAME = DATA_DIR / "vagas.json"
PROSPECTS_FILENAME = DATA_DIR / "prospects.json"

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
baixar_arquivo(VAGAS_JSON_URL, VAGAS_FILENAME)
baixar_arquivo(PROSPECTS_JSON_URL, PROSPECTS_FILENAME)
baixar_arquivo(APPLICANTS_JSON_URL, RAW_APPLICANTS_FILENAME, is_large=True)

# Converte o JSON de applicants para NDJSON se necessário
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

# --- FIM DA SEÇÃO DE DOWNLOAD ---


# --- INÍCIO DO PROCESSO DE TREINAMENTO ---

# 1. Carregar os dados locais
print("\n--- Etapa 1: Carregando dados para a memória ---")
with open(VAGAS_FILENAME, 'r', encoding='utf-8') as f:
    vagas_data = json.load(f)
with open(PROSPECTS_FILENAME, 'r', encoding='utf-8') as f:
    prospects_data = json.load(f)

# 2. Criar o DataFrame mestre
print("--- Etapa 2: Criando o DataFrame mestre para treinamento ---")
lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
df_vagas = pd.json_normalize(lista_vagas, sep='_')

lista_prospects = []
for vaga_id, data in prospects_data.items():
    for prospect in data.get('prospects', []):
        lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': prospect.get('codigo'), 'status_final': prospect.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)

# Lê o arquivo NDJSON com pandas
df_applicants_details = pd.read_json(NDJSON_FILENAME, lines=True)

# Merge dos dataframes
df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
df_applicants_details['codigo_candidato'] = df_applicants_details['codigo_candidato'].astype(str)
if 'nome' in df_applicants_details.columns:
    df_applicants_details.rename(columns={'nome': 'nome_candidato'}, inplace=True)
df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')

# 3. Preparar os dados para o formato de treino
print("--- Etapa 3: Preparando e limpando os dados ---")
# Combina textos da vaga
colunas_perfil_vaga = [col for col in df_mestre.columns if col.startswith('perfil_vaga_')]
for col in colunas_perfil_vaga:
    df_mestre[col] = df_mestre[col].fillna('').astype(str)
df_mestre['texto_vaga_combinado'] = df_mestre[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)

# Combina textos do candidato
# Para extrair o texto de cv_pt e cv_en, precisamos normalizar o df de applicants
df_applicants_normalized = pd.json_normalize(df_applicants_details.to_dict('records'), sep='_')
df_applicants_normalized['codigo_candidato'] = df_applicants_normalized['codigo_candidato'].astype(str)
text_cols = ['informacoes_profissionais_resumo_profissional', 'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en']
for col in text_cols:
    if col not in df_applicants_normalized.columns:
        df_applicants_normalized[col] = ''
    df_applicants_normalized[col] = df_applicants_normalized[col].fillna('')
df_applicants_normalized['candidato_texto_completo'] = df_applicants_normalized[text_cols].apply(lambda x: ' '.join(x), axis=1)
df_mestre = pd.merge(df_mestre.drop(columns=['candidato_texto_completo'], errors='ignore'), df_applicants_normalized[['codigo_candidato', 'candidato_texto_completo']], on='codigo_candidato', how='left')


df_mestre['texto_completo'] = df_mestre['texto_vaga_combinado'] + ' ' + df_mestre['candidato_texto_completo'].fillna('')
positivos_keywords = ['contratado', 'aprovado', 'documentação']
df_mestre['status_final_lower'] = df_mestre['status_final'].astype(str).str.lower()
df_mestre['target'] = df_mestre['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)
df_treino = df_mestre[df_mestre['status_final'] != 'N/A'].copy()

print(f"Total de {len(df_treino)} registros válidos para o treinamento.")

# 4. Treinar o modelo
print("--- Etapa 4: Treinando o modelo de Machine Learning ---")
features = ['texto_completo']
target = 'target'
X = df_treino[features]
y = df_treino[target]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer(transformers=[('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2)), 'texto_completo')], remainder='drop')
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
])
pipeline.fit(X_train, y_train)

# 5. Salvar o pipeline treinado em um arquivo
print("--- Etapa 5: Salvando o modelo treinado em 'modelo_recrutamento.joblib' ---")
joblib.dump(pipeline, 'modelo_recrutamento.joblib')

print("\nTreinamento concluído com sucesso! O arquivo 'modelo_recrutamento.joblib' foi criado.")
print("Agora, faça o download deste arquivo no painel à esquerda do Colab e envie-o para o seu repositório no GitHub.")
