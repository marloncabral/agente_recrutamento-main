import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import utils 

def criar_dataframe_mestre_otimizado(vagas_data, prospects_data):
    """Cria um DataFrame mestre de forma otimizada, garantindo a retenção de todas as colunas."""
    lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
    df_vagas = pd.json_normalize(lista_vagas, sep='_')

    lista_prospects = []
    for vaga_id, data in prospects_data.items():
        for prospect in data.get('prospects', []):
            lista_prospects.append({
                'codigo_vaga': vaga_id,
                'codigo_candidato': prospect.get('codigo'),
                'status_final': prospect.get('situacao_candidado', 'N/A')
            })
    df_prospects = pd.DataFrame(lista_prospects)

    ids_necessarios = df_prospects['codigo_candidato'].unique().tolist()
    df_applicants_details = utils.buscar_detalhes_candidatos(ids_necessarios)
    
    # Assegura que a coluna de ID do candidato seja string em ambos os dataframes
    df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
    if not df_applicants_details.empty:
        df_applicants_details['codigo_candidato'] = df_applicants_details['codigo_candidato'].astype(str)
        if 'nome' in df_applicants_details.columns:
            df_applicants_details.rename(columns={'nome': 'nome_candidato'}, inplace=True)

    df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
    # Se df_applicants_details não estiver vazio, faz o merge
    if not df_applicants_details.empty:
        df_mestre = pd.merge(df_mestre, df_applicants_details, on='codigo_candidato', how='left')
    # Se estiver vazio, garante que as colunas existam, preenchidas com nulos
    else:
        df_mestre['nome_candidato'] = None
        df_mestre['candidato_texto_completo'] = ''

    return df_mestre

def preparar_dados_para_treino(df):
    """Prepara o DataFrame mestre para o treinamento do modelo, tratando dados faltantes de forma robusta."""
    
    # --- INÍCIO DA CORREÇÃO DE RESILIÊNCIA ---
    colunas_perfil_vaga = [col for col in df.columns if col.startswith('perfil_vaga_')]
    for col in colunas_perfil_vaga:
        df[col] = df[col].fillna('').astype(str)
    
    df['texto_vaga_combinado'] = df[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)
    
    # Trata de forma segura o texto do candidato, que pode ser nulo após o merge
    df['texto_candidato_combinado'] = df['candidato_texto_completo'].fillna('').astype(str)
    
    # Combina os textos para a análise do modelo
    df['texto_completo'] = df['texto_vaga_combinado'] + ' ' + df['texto_candidato_combinado']
    
    positivos_keywords = ['contratado', 'aprovado', 'documentação']
    df['status_final_lower'] = df['status_final'].astype(str).str.lower()
    df['target'] = df['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)

    # Filtra apenas os registros que possuem um status final definido
    df_modelo = df[df['status_final'] != 'N/A'].copy()
    # Remove o dropna() agressivo que estava causando o problema
    # --- FIM DA CORREÇÃO DE RESILIÊNCIA ---

    if df_modelo.empty or df_modelo['target'].nunique() < 2:
        st.error("Falha Crítica no Treinamento: Após filtrar, não há dados históricos suficientes com exemplos de sucesso e fracasso.")
        return None
        
    return df_modelo

@st.cache_resource(show_spinner="Treinando modelo de Machine Learning...")
def treinar_modelo_matching(vagas_data, prospects_data):
    """Orquestra a preparação dos dados e o treinamento do modelo de ML."""
    df_mestre = criar_dataframe_mestre_otimizado(vagas_data, prospects_data)
    if df_mestre.empty: return None, None

    df_treino = preparar_dados_para_treino(df_mestre)
    if df_treino is None: return None, None

    features = ['texto_completo']
    target = 'target'
    X = df_treino[features]
    y = df_treino[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = ColumnTransformer(transformers=[('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2)), 'texto_completo')], remainder='drop')
    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
    pipeline.fit(X_train, y_train)

    try:
        y_pred_test = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred_test)
        st.session_state.model_performance = f"Modelo treinado! Performance (F1-Score): {f1:.2f}"
    except Exception:
        st.session_state.model_performance = "Modelo treinado!"
    return pipeline, X_train
