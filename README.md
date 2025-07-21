# Decision Match: Assistente de Recrutamento com IA

Projeto desenvolvido para o Datathon da Pós-Tech Data Analytics.

**Link para a Aplicação Streamlit:** [Inserir o link do seu app deployado aqui]

## 1. O Problema

A empresa de recrutamento "Decision" enfrenta desafios na padronização de entrevistas e na identificação rápida de candidatos qualificados, o que afeta a agilidade e a qualidade do processo seletivo.

## 2. A Solução

O "Decision Match" é uma plataforma multiagente baseada em IA para otimizar o recrutamento:

* **Agente 1 (Matching):** Utiliza um modelo de Machine Learning (Regressão Logística) para analisar o perfil de candidatos e vagas, gerando um score de compatibilidade para ranquear os melhores talentos.
* **Agente 2 (Entrevistas):** Conduz entrevistas preliminares de forma automatizada usando IA Generativa (Google Gemini), aprofundando a análise dos candidatos selecionados.
* **Agente 3 (Análise Final):** Gera relatórios individuais e uma análise comparativa final, fornecendo um parecer completo para auxiliar na tomada de decisão.

## 3. Como Executar o Projeto Localmente

1.  Clone o repositório: `git clone [seu-link-do-repo]`
2.  Instale as dependências do sistema: `sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx`
3.  Instale as bibliotecas Python: `pip install -r requirements.txt`
4.  Execute a aplicação Streamlit: `streamlit run app.py`

## 4. Estrutura do Projeto

- `app.py`: Código principal da aplicação Streamlit.
- `train.py`: Script para treinamento do modelo de Machine Learning.
- `utils.py`: Funções de suporte (download, processamento de dados).
- `modelo_recrutamento.joblib`: Modelo treinado.
- `requirements.txt`: Dependências Python.
- `packages.txt`: Dependências de sistema (para deploy).
