# ✨ Assistente de Recrutamento IA - Projeto Datathon Decision

Projeto desenvolvido para o Datathon da Pós-Tech Data Analytics, com o objetivo de solucionar os desafios de recrutamento da empresa "Decision" utilizando uma abordagem multiagente baseada em Inteligência Artificial.

**[>> Link para a Aplicação Web <<](https://insira-o-link-do-seu-app-streamlit-aqui.streamlit.app/)**

---

## 1. O Desafio de Negócio

A "Decision", empresa especializada em recrutamento e bodyshop de TI, enfrenta desafios para escalar seu processo seletivo mantendo a alta qualidade. As principais dores identificadas foram:

* **Falta de Padronização:** Dificuldade em manter um padrão objetivo na avaliação inicial dos candidatos.
* **Demanda de Tempo:** O processo de triagem manual consome um tempo valioso dos recrutadores ("hunters").
* **Engajamento Incerto:** Dificuldade em avaliar o real interesse e engajamento dos candidatos nas fases iniciais.

O desafio proposto foi desenvolver um MVP (Mínimo Produto Viável) de uma solução de IA que otimizasse o processo de recrutamento, tornando-o mais ágil, preciso e baseado em dados.

---

## 2. A Solução: Plataforma Multiagente

Para resolver o problema, foi desenvolvida uma aplicação web completa em Streamlit que opera com um sistema de três agentes inteligentes, cada um responsável por uma etapa crucial do funil de recrutamento:

### Agente 1: Matching e Análise com IA Explicável (XAI)
* **O que faz?** Utiliza um modelo de Machine Learning (Regressão Logística) treinado com dados históricos para analisar o perfil de centenas de candidatos e gerar um **score de compatibilidade** com a vaga selecionada.
* **Diferencial:** Além de ranquear, a ferramenta utiliza a biblioteca **SHAP** para fornecer **explicabilidade (XAI)**. O recrutador pode ver exatamente quais palavras-chave no perfil do candidato mais contribuíram (positiva ou negativamente) para o seu score, aumentando a confiança e a transparência do processo.

### Agente 2: Entrevistas com IA Generativa
* **O que faz?** Após a seleção dos melhores candidatos pelo Agente 1, o Agente 2, alimentado pelo **Google Gemini**, conduz uma entrevista preliminar via chat. Ele formula perguntas contextuais com base no perfil do candidato e da vaga.
* **Valor:** Automatiza a primeira rodada de entrevistas, padroniza a coleta de informações e gera uma transcrição estruturada para análise posterior.

### Agente 3: Análise Final e Comparativa
* **O que faz?** Após a conclusão das entrevistas, o Agente 3 utiliza a IA Generativa para analisar as transcrições e gerar dois tipos de relatórios:
    1.  **Relatório Individual:** Um resumo para cada candidato, com pontos fortes, pontos de atenção e uma recomendação.
    2.  **Parecer Comparativo Final:** Uma análise consolidada de todos os finalistas, com um ranking e uma justificativa para a recomendação do candidato ideal, pronta para ser apresentada ao cliente final.

---

## 3. Tecnologias Utilizadas

* **Frontend:** Streamlit
* **Análise e Manipulação de Dados:** Pandas, NumPy
* **Machine Learning (Matching):** Scikit-learn (TfidfVectorizer, LogisticRegression)
* **Explicabilidade (XAI):** SHAP, Matplotlib
* **IA Generativa (Entrevistas e Relatórios):** Google Generative AI (Gemini)
* **Deployment:** Streamlit Community Cloud

---

## 4. Estrutura do Repositório

.├── 📄 .streamlit/config.toml  # Arquivo de configuração do Streamlit├── 🐍 app.py                  # Código principal da aplicação Streamlit (UI e lógica dos agentes)├── 📦 data/                    # Diretório para os dados (criado dinamicamente)├── 📄 packages.txt            # Dependências de sistema para o deploy├── 📄 README.md                # Esta documentação├── 📄 requirements.txt         # Dependências Python do projeto├── 🤖 train.py                 # Script para treinar o modelo de ML e o explicador SHAP├── 🛠️ utils.py                 # Funções de suporte (download, processamento de dados)├── 🧠 modelo_recrutamento.joblib # Artefato do modelo de ML treinado└── 📊 shap_explainer.joblib    # Artefato do explicador SHAP treinado
---

## 5. Como Executar o Projeto Localmente

**Pré-requisitos:**
* Python 3.9+
* Git

**Passos:**

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure sua chave de API:**
    * Crie um arquivo `.streamlit/secrets.toml`.
    * Dentro dele, adicione sua chave do Google Gemini:
        ```toml
        GOOGLE_API_KEY = "SUA_CHAVE_DE_API_AQUI"
        ```

5.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```

A aplicação estará disponível no seu navegador em `http://localhost:8501`.
