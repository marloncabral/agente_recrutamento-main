# -*- coding: utf-8 -*-

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

def criar_dataframe_mestre():
    """
    Usa os dados brutos para criar um DataFrame mestre, unindo vagas,
    prospects e candidatos para o treinamento.
    """
    # Carrega todos os dados necessários usando as funções do utils
    vagas_data = utils.carregar_json(utils.VAGAS_FILENAME)
    prospects_data = utils.carregar_json(utils.PROSPECTS_FILENAME)

    # Transforma os dicionários em listas de dicionários
    lista_vagas = [{'codigo_vaga': k, **v} for k, v in vagas_data.items()]
    df_vagas = pd.json_normalize(lista_vagas, sep='_')

    # Cria a tabela de relação (tabela fato)
    lista_prospects = []
    for vaga_id, data in prospects_data.items():
        for prospect in data.get('prospects', []):
            lista_prospects.append({
                'codigo_vaga': vaga_id,
                'codigo_candidato': prospect.get('codigo'),
                # Assumindo que o status está aqui. ESSA É A CHAVE DO PROJETO!
                'status_final': prospect.get('feedback', 'N/A')
            })
    df_prospects = pd.DataFrame(lista_prospects)

    # Carrega TODOS os candidatos do arquivo otimizado de uma vez
    df_applicants = pd.read_json(utils.NDJSON_FILENAME, lines=True)

    # Merge para criar o DataFrame mestre
    df_mestre = pd.merge(df_prospects, df_vagas, on='codigo_vaga', how='left')
    df_mestre = pd.merge(df_mestre, df_applicants, on='codigo_candidato', how='left')

    return df_mestre

def preparar_dados_para_treino(df):
    """Prepara o DataFrame mestre para o treinamento do modelo."""

    # Seleciona e limpa colunas de texto relevantes de vagas e candidatos
    colunas_texto_vaga = ['perfil_vaga_competencia_tecnicas_e_comportamentais']
    colunas_texto_candidato = [
        'informacoes_profissionais_resumo_profissional',
        'informacoes_profissionais_conhecimentos', 'cv_pt', 'cv_en'
    ]

    for col in colunas_texto_vaga + colunas_texto_candidato:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = '' # Adiciona a coluna vazia se não existir

    # Combina as colunas de texto em uma única feature para o modelo
    df['texto_vaga'] = df[colunas_texto_vaga].apply(lambda x: ' '.join(x), axis=1)
    df['texto_candidato'] = df[colunas_texto_candidato].apply(lambda x: ' '.join(x), axis=1)

    # Cria a variável alvo (target)
    # A hipótese é que um feedback positivo ou contratação indica sucesso.
    # Adapte essa lista conforme os valores reais nos seus dados.
    positivos = ['Contratado', 'Feedback Positivo', 'Aprovado', 'Contratada']
    df['target'] = df['status_final'].apply(lambda x: 1 if x in positivos else 0)

    # Filtra apenas os dados úteis para o treinamento
    df_modelo = df[df['status_final'] != 'N/A'].copy()

    if df_modelo['target'].nunique() < 2:
        st.error("Não foi possível criar uma variável alvo com duas classes (0 e 1). Verifique os valores na coluna 'feedback' do arquivo prospects.json. O treinamento não pode continuar.")
        st.stop()

    return df_modelo[['texto_vaga', 'texto_candidato', 'target']]

# --- Função de Treinamento ---

@st.cache_resource(show_spinner="Treinando modelo de Machine Learning...")
def treinar_modelo_matching():
    """
    Orquestra a preparação dos dados e o treinamento do modelo de ML.
    Usa @st.cache_resource para que o modelo treinado seja mantido em cache,
    evitando o retreinamento a cada interação do usuário.
    """
    df_mestre = criar_dataframe_mestre()
    df_treino = preparar_dados_para_treino(df_mestre)

    # Combina texto da vaga e do candidato para o TF-IDF
    X = df_treino['texto_vaga'] + ' ' + df_treino['texto_candidato']
    y = df_treino['target']

    # Divide os dados (apenas para logar a performance, o modelo final usa todos os dados)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Treina o modelo com TODOS os dados disponíveis para máxima performance no app
    pipeline.fit(X, y)

    # Loga a performance (opcional, mas bom para debug)
    try:
        y_pred_test = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred_test)
        st.session_state.model_performance = f"Modelo treinado! Performance (F1-Score no teste): {f1:.2f}"
    except Exception:
        st.session_state.model_performance = "Modelo treinado! (Não foi possível calcular F1-Score no teste - poucos dados)"

    return pipeline
