import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import utils

# --- ETAPA 1: PREPARAÇÃO DOS DADOS ---
def etapa_1_preparar_dados(vagas_data, prospects_data):
    """
    Executa a primeira etapa: carregar, unir e limpar os dados para treinamento,
    preservando todas as colunas necessárias.
    """
    # Função interna para criar o DataFrame mestre
    def criar_dataframe_mestre(vagas_data, prospects_data):
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
            # A correção crucial é manter todas as colunas de df_applicants_details
            df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')
        else:
            df_mestre['nome_candidato'] = None
            df_mestre['candidato_texto_completo'] = ''
        return df_mestre

    # Função interna para preparar o DataFrame para o modelo
    def preparar_para_treino(df):
        colunas_perfil_vaga = [col for col in df.columns if col.startswith('perfil_vaga_')]
        for col in colunas_perfil_vaga:
            df[col] = df[col].fillna('').astype(str)
        df['texto_vaga_combinado'] = df[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)
        
        df['texto_candidato_combinado'] = df['candidato_texto_completo'].fillna('').astype(str)
        df['texto_completo'] = df['texto_vaga_combinado'] + ' ' + df['texto_candidato_combinado']
        
        positivos_keywords = ['contratado', 'aprovado', 'documentação']
        df['status_final_lower'] = df['status_final'].astype(str).str.lower()
        df['target'] = df['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)
        
        df_modelo = df[df['status_final'] != 'N/A'].copy()
        df_modelo.dropna(subset=['texto_completo'], inplace=True)

        if df_modelo.empty or df_modelo['target'].nunique() < 2:
            return None
        return df_modelo

    df_mestre = criar_dataframe_mestre(vagas_data, prospects_data)
    if df_mestre.empty: return None
    
    df_treino = preparar_para_treino(df_mestre)
    return df_treino

# --- ETAPA 2: TREINAMENTO DO MODELO ---
def etapa_2_treinar_modelo(df_treino):
    """Executa o treinamento do pipeline de Machine Learning."""
    if df_treino is None or df_treino.empty: return None, None, None

    # Retorna o DataFrame completo para que o app possa usar as colunas de nome e id
    return df_treino

# --- ETAPA 3: AVALIAÇÃO E FINALIZAÇÃO ---
def etapa_3_avaliar_e_finalizar(df_treino):
    """Treina e avalia o modelo, retornando o pipeline treinado e a performance."""
    if df_treino is None: return None, "Modelo não pôde ser treinado."

    features = ['texto_completo']
    target = 'target'
    X = df_treino[features]
    y = df_treino[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(transformers=[('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2)), 'texto_completo')], remainder='drop')
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)
    
    try:
        y_pred_test = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred_test)
        performance_string = f"Modelo treinado! Performance (F1-Score): {f1:.2f}"
    except:
        performance_string = "Modelo treinado! (Não foi possível calcular a performance)"
        
    return pipeline, performance_string
