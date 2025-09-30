# 🤖 Credit Agent — Análise Automática de CSV

Agente inteligente para **Análise Exploratória de Dados (EDA)** de arquivos CSV, com foco no dataset de fraude em cartão de crédito do Kaggle.  
Permite que o usuário faça perguntas em linguagem natural e receba respostas interpretativas, com suporte a gráficos interativos.

---

## 🚀 Framework escolhida
A solução foi desenvolvida em **Python** usando o **Streamlit** como framework para a interface interativa.  
O agente (em `agent.py`) integra bibliotecas de ciência de dados como **Pandas, Scikit-learn, Seaborn, Matplotlib e Plotly**.

---

## 🏗️ Estrutura da solução
- `agent.py` → contém a classe `EvolvedAgent`, responsável por carregar os dados, processar perguntas, gerar análises e gráficos, além de produzir respostas interpretativas.  
- `app.py` → aplicação Streamlit que integra a interface gráfica, permite upload de CSV/ZIP e envio de perguntas ao agente.  
- `requirements.txt` → dependências para rodar o projeto.  
- `README.md` → este arquivo, com instruções de uso.  

---

## ❓ Exemplos de perguntas e respostas

### 1. Tipos de dados
**Pergunta:**  
> Quais são os tipos de dados (numéricos, categóricos)?

**Resposta:**  
Todos os atributos do dataset `creditcard.csv` são numéricos (resultados de transformações PCA), com exceção de `Class`, que indica fraude (1) ou transação legítima (0).  
Isso significa que todas as análises são feitas sobre variáveis contínuas ou discretas.

---

### 2. Distribuição de variáveis (com gráfico)
**Pergunta:**  
> Qual a distribuição de cada variável (histogramas, distribuições)?

**Resposta:**  
Foram gerados histogramas para variáveis como `Time`, `V1`, `V2`, `Amount`, entre outras.  
Os gráficos mostram que a maioria das variáveis transformadas segue distribuição próxima de normal centrada em zero, enquanto `Amount` apresenta forte assimetria (maioria das transações de baixo valor).

📊 **Exemplo de gráfico gerado:**
*(ao rodar localmente com `streamlit run app.py`, o gráfico aparece interativo no app).*

---

### 3. Correlações
**Pergunta:**  
> Existe correlação entre as variáveis?

**Resposta:**  
As variáveis mais associadas à ocorrência de fraude (`Class`) são:  
- `V17` (0.326)  
- `V14` (0.303)  
- `V12` (0.261)  
- `V10` (0.217)  
- `V16` (0.197)  

📖 **Interpretação:**  
Isso indica que alguns componentes transformados por PCA carregam padrões fortemente relacionados a transações fraudulentas.  
No gráfico de calor, áreas em vermelho indicam forte correlação positiva e azul negativa.

---

### 4. Clusters
**Pergunta:**  
> Existem agrupamentos (clusters) nos dados?

**Resposta:**  
Foram aplicados **KMeans (k=3)**, identificando três agrupamentos distintos.  
A maioria dos clusters concentrou transações legítimas, mas um pequeno grupo mostrou proporção maior de fraudes.

📖 **Interpretação:**  
Clustering pode ajudar a segmentar transações suspeitas, servindo como etapa preliminar para modelos de detecção.

---

### 5. Conclusão geral
**Pergunta:**  
> Quais conclusões gerais podemos tirar da análise?

**Resposta:**  
- O dataset é altamente desbalanceado (menos de 0,2% de fraudes).  
- Certas variáveis (`V17`, `V14`, `V12`) são fortemente indicativas de fraude.  
- A maior parte das transações legítimas tem valores baixos, enquanto fraudes tendem a concentrar-se em valores médios.  
- A aplicação de clusters e correlações mostra que padrões de fraude podem ser isolados, mas exigem técnicas de balanceamento para modelos preditivos.

---

## 🛠️ Como rodar o projeto
Crie e ative o ambiente virtual:
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell

Instale as dependências:
pip install -r requirements.txt

Rode o app:
streamlit run app.py

Carregue os dados:
Faça upload de creditcard.csv (dataset do Kaggle).
Ou use a opção de upload ZIP com CSV dentro.
Ou clique em Kaggle (automatico)

### 1. Clone o repositório
```bash
git clone https://github.com/VaniceGomes/credit-agent/
cd credit-agent
