import streamlit as st
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import utils 

# --- Funções de Preparação de Dados para ML ---

def criar_dataframe_mestre_otimizado(vagas_data, prospects_data):
    """
    Usa os dados brutos para criar um DataFrame mestre de forma otimizada,
    evitando carregar todos os candidatos em memória.
    """
    # 1. Carrega vagas e normaliza (operação rápida)
    lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
    df_vagas = pd.json_normalize(lista_vagas, sep='_')

    # 2. Carrega prospects e cria o DataFrame inicial (operação rápida)
    lista_prospects = []
    for vaga_id, data in prospects_data.items():
        for prospect in data.get('prospects', []):
            lista_prospects.append({
                'codigo_vaga': vaga_id,
                'codigo_candidato': prospect.get('codigo'),
                'status_final': prospect.get('situacao_candidado', 'N/A') 
            })
    df_prospects = pd.DataFrame(lista_prospects)

    # 3. OTIMIZAÇÃO PRINCIPAL: Pega apenas os IDs dos candidatos que precisamos
    ids_necessarios = df_prospects['codigo_candidato'].unique().tolist()
    
    # 4. Usa a função eficiente de utils para buscar APENAS esses candidatos
    df_applicants = utils.buscar_detalhes_candidatos(ids_necessarios)
    if df_applicants.empty:
        st.error("Não foi possível buscar os detalhes dos candidatos necessários.")
        return pd.DataFrame()

    # Renomeia a coluna 'nome' para evitar conflitos no merge
    df_applicants.rename(columns={'nome': 'nome_candidato'}, inplace=True)

    # 5. Merge final em dataframes muito menores
    df_prospects['codigo_candidato'] = df_prospects['codigo_candidato'].astype(str)
    df_applicants['codigo_candidato'] = df_applicants['codigo_candidato'].astype(str)

    df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
    df_mestre = pd.merge(df_mestre, df_applicants, on='codigo_candidato', how='left')
    
    return df_mestre

def preparar_dados_para_treino(df):
    """Prepara o DataFrame mestre para o treinamento do modelo."""
    
    # --- INÍCIO DA CORREÇÃO ---
    # Identifica dinamicamente todas as colunas que foram criadas a partir do 'perfil_vaga'
    colunas_perfil_vaga = [col for col in df.columns if col.startswith('perfil_vaga_')]
    
    # Garante que todas essas colunas sejam texto e preenche valores nulos
    for col in colunas_perfil_vaga:
        df[col] = df[col].fillna('').astype(str)

    # Combina todas as informações de perfil da vaga em uma única string de texto
    df['texto_vaga_combinado'] = df[colunas_perfil_vaga].apply(lambda x: ' '.join(x), axis=1)
    # --- FIM DA CORREÇÃO ---

    df['texto_candidato_combinado'] = df['candidato_texto_completo'].fillna('').astype(str)
    df['texto_completo'] = df['texto_vaga_combinado'] + ' ' + df['texto_candidato_combinado']
    
    positivos_keywords = ['contratado', 'aprovado', 'documentação']
    
    df['status_final_lower'] = df['status_final'].astype(str).str.lower()
    
    df['target'] = df['status_final_lower'].apply(lambda x: 1 if any(keyword in x for keyword in positivos_keywords) else 0)

    df_modelo = df[df['status_final'] != 'N/A'].copy()
    
    if df_modelo.empty or df_modelo['target'].nunique() < 2:
        st.error("Falha Crítica no Treinamento: Não foi possível encontrar dados de feedback histórico (sucesso/fracasso) para treinar o modelo.")
        unique_feedbacks = df_modelo['status_final'].unique()
        st.warning("A IA precisa aprender com o passado. Para isso, a coluna 'situacao_candidado' no arquivo `prospects.json` deve conter exemplos de sucesso (ex: 'Contratado') e de fracasso.")
        st.info(f"Valores de feedback encontrados (após filtrar 'N/A'): **{list(unique_feedbacks)}**")
        return None
        
    return df_modelo

# --- Função de Treinamento ---

@st.cache_resource(show_spinner="Treinando modelo de Machine Learning...")
def treinar_modelo_matching(vagas_data, prospects_data):
    """
    Orquestra a preparação dos dados e o treinamento do modelo de ML.
    """
    # Usa a função otimizada
    df_mestre = criar_dataframe_mestre_otimizado(vagas_data, prospects_data)
    if df_mestre.empty:
        return None, None
        
    df_treino = preparar_dados_para_treino(df_mestre)
    
    if df_treino is None:
        return None, None

    features = ['texto_completo']
    target = 'target'
    X = df_treino[features]
    y = df_treino[target]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2)), 'texto_completo')
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    try:
        y_pred_test = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred_test)
        st.session_state.model_performance = f"Modelo treinado! Performance (F1-Score no teste): {f1:.2f}"
    except Exception:
        st.session_state.model_performance = "Modelo treinado! (Não foi possível calcular F1-Score no teste - poucos dados)"
    
    return pipeline, X_train
