import streamlit as st
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import utils # Importa as funções do seu arquivo utils.py

# --- Funções de Preparação de Dados para ML ---

def criar_dataframe_mestre(vagas_data, prospects_data):
    """
    Usa os dados brutos (passados como argumento) para criar um DataFrame mestre,
    unindo vagas, prospects e candidatos para o treinamento.
    """
    lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
    df_vagas = pd.json_normalize(lista_vagas, sep='_')

    lista_prospects = []
    for vaga_id, data in prospects_data.items():
        for prospect in data.get('prospects', []):
            lista_prospects.append({
                'codigo_vaga': vaga_id,
                'codigo_candidato': prospect.get('codigo'),
                'status_final': prospect.get('feedback', 'N/A') 
            })
    df_prospects = pd.DataFrame(lista_prospects)

    df_applicants = pd.read_json(utils.NDJSON_FILENAME, lines=True)

    # Garante que as chaves de merge tenham o mesmo tipo de dado (string) para evitar erros.
    df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
    df_applicants['codigo_candidato'] = df_applicants['codigo_candidato'].astype(str)

    # Merge para criar o DataFrame mestre
    df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
    df_mestre = pd.merge(df_mestre, df_applicants, on='codigo_candidato', how='left')
    
    return df_mestre

def preparar_dados_para_treino(df):
    """Prepara o DataFrame mestre para o treinamento do modelo."""
    
    colunas_texto_vaga = ['perfil_vaga_competencia_tecnicas_e_comportamentais']
    colunas_texto_candidato = [
        'informacoes_profissionais_resumo_profissional',
        'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en'
    ]
    
    for col in colunas_texto_vaga + colunas_texto_candidato:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = ''

    df['texto_vaga'] = df[colunas_texto_vaga].apply(lambda x: ' '.join(x), axis=1)
    df['texto_candidato'] = df[colunas_texto_candidato].apply(lambda x: ' '.join(x), axis=1)
    
    positivos_keywords = ['contratado', 'feedback positivo', 'aprovado', 'contratada', 'aprovada']
    
    df['status_final_lower'] = df['status_final'].astype(str).str.lower()
    
    df['target'] = df['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)

    df_modelo = df[df['status_final'] != 'N/A'].copy()
    
    if df_modelo.empty or df_modelo['target'].nunique() < 2:
        st.error("Falha Crítica no Treinamento: Não foi possível encontrar dados de feedback histórico (sucesso/fracasso) para treinar o modelo de Machine Learning.")
        unique_feedbacks = df_modelo['status_final'].unique()
        st.warning("A IA precisa aprender com o passado. Para isso, a coluna 'feedback' no arquivo `prospects.json` deve conter exemplos de sucesso (ex: 'Contratado') e de fracasso.")
        st.info(f"Valores de feedback encontrados (após filtrar 'N/A'): **{list(unique_feedbacks)}**")
        return None
        
    return df_modelo[['texto_vaga', 'texto_candidato', 'target']]

# --- Função de Treinamento ---

@st.cache_resource(show_spinner="Treinando modelo de Machine Learning...")
def treinar_modelo_matching(vagas_data, prospects_data):
    """
    Orquestra a preparação dos dados e o treinamento do modelo de ML.
    Recebe os dados como argumentos para garantir a ordem de execução.
    """
    df_mestre = criar_dataframe_mestre(vagas_data, prospects_data)
    df_treino = preparar_dados_para_treino(df_mestre)
    
    if df_treino is None:
        return None

    X = df_treino['texto_vaga'] + ' ' + df_treino['texto_candidato']
    y = df_treino['target']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X, y)
    
    try:
        y_pred_test = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred_test)
        st.session_state.model_performance = f"Modelo treinado! Performance (F1-Score no teste): {f1:.2f}"
    except Exception:
        st.session_state.model_performance = "Modelo treinado! (Não foi possível calcular F1-Score no teste - poucos dados)"
    
    return pipeline
