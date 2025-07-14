import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib # Biblioteca para salvar o modelo
import utils # Reutilizamos nosso módulo de utilidades

# Mensagem inicial
print("Iniciando o processo de treinamento do modelo...")

# 1. Carregar os dados originais
print("Etapa 1: Carregando dados de vagas e prospects...")
vagas_data = utils.carregar_json(utils.VAGAS_FILENAME)
prospects_data = utils.carregar_prospects()

# 2. Criar o DataFrame mestre otimizado
print("Etapa 2: Criando o DataFrame mestre para treinamento...")
# Reutilizamos a lógica robusta que já criamos
# (Esta é uma versão simplificada da função de ml_logic.py)
lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
df_vagas = pd.json_normalize(lista_vagas, sep='_')
lista_prospects = []
for vaga_id, data in prospects_data.items():
    for prospect in data.get('prospects', []):
        lista_prospects.append({'codigo_vaga': vaga_id, 'codigo_candidato': prospect.get('codigo'), 'status_final': prospect.get('situacao_candidado', 'N/A')})
df_prospects = pd.DataFrame(lista_prospects)
ids_necessarios = df_prospects['codigo_candidato'].unique().tolist()
df_applicants_details = utils.buscar_detalhes_candidatos(ids_necessarios)
df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
if not df_applicants_details.empty:
    df_applicants_details['codigo_candidato'] = df_applicants_details['codigo_candidato'].astype(str)
    if 'nome' in df_applicants_details.columns:
        df_applicants_details.rename(columns={'nome': 'nome_candidato'}, inplace=True)
df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
if not df_applicants_details.empty:
    df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')
else:
    df_mestre['nome_candidato'] = None
    df_mestre['candidato_texto_completo'] = ''

# 3. Preparar os dados para o formato de treino
print("Etapa 3: Preparando e limpando os dados...")
colunas_perfil_vaga = [col for col in df_mestre.columns if col.startswith('perfil_vaga_')]
for col in colunas_perfil_vaga:
    df_mestre[col] = df_mestre[col].fillna('').astype(str)
df_mestre['texto_vaga_combinado'] = df_mestre[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)
df_mestre['texto_candidato_combinado'] = df_mestre['candidato_texto_completo'].fillna('').astype(str)
df_mestre['texto_completo'] = df_mestre['texto_vaga_combinado'] + ' ' + df_mestre['texto_candidato_combinado']
positivos_keywords = ['contratado', 'aprovado', 'documentação']
df_mestre['status_final_lower'] = df_mestre['status_final'].astype(str).str.lower()
df_mestre['target'] = df_mestre['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)
df_treino = df_mestre[df_mestre['status_final'] != 'N/A'].copy()

print(f"Total de {len(df_treino)} registros válidos para o treinamento.")

# 4. Treinar o modelo
print("Etapa 4: Treinando o modelo de Machine Learning...")
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
print("Etapa 5: Salvando o modelo treinado em 'modelo_recrutamento.joblib'...")
joblib.dump(pipeline, 'modelo_recrutamento.joblib')

print("\nTreinamento concluído com sucesso! O arquivo 'modelo_recrutamento.joblib' foi criado.")
print("Agora, faça o upload deste arquivo para o seu repositório no GitHub.")
