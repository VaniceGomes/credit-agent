import streamlit as st
import pandas as pd
import zipfile
import os
from agent import EvolvedAgent

# Inicializa o agente
agent = EvolvedAgent()
os.makedirs("outputs", exist_ok=True)

st.set_page_config(page_title="Agente Evolved — EDA Automática", layout="wide")
st.title("🤖 Agente Evolved — Análise Automática de CSV")

# ---------------- Sidebar ----------------
st.sidebar.header("📂 Fonte de Dados")
input_type = st.sidebar.radio(
    "Escolha o tipo de entrada",
    ["CSV", "ZIP", "Kaggle (automático)"]
)

df = None
msg = None

if input_type == "CSV":
    uploaded = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        msg = agent.load_dataframe(df, uploaded.name)
        st.sidebar.success(msg)

elif input_type == "ZIP":
    uploaded = st.sidebar.file_uploader("Envie um arquivo ZIP com CSV dentro", type=["zip"])
    if uploaded:
        with zipfile.ZipFile(uploaded, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if csv_files:
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    msg = agent.load_dataframe(df, csv_files[0])
                    st.sidebar.success(msg)
            else:
                st.sidebar.error("Nenhum CSV encontrado no ZIP.")

elif input_type == "Kaggle (automático)":
    import kagglehub
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    file_path = os.path.join(dataset_path, "creditcard.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        msg = agent.load_dataframe(df, "creditcard.csv")
        st.sidebar.success(msg)
    else:
        st.sidebar.error("Não foi possível baixar o dataset Kaggle.")

# ---------------- Main ----------------
if df is not None:
    st.markdown("### 🔎 Status do Conjunto de Dados")
    st.markdown(f"**Conjunto de dados carregado:** `{agent.filename}` — **{df.shape[0]} linhas x {df.shape[1]} colunas**")
    st.dataframe(df.head())

    st.markdown("---")
    st.markdown("### 💬 Pergunte ao Agente")
    user_q = st.text_input("Digite sua pergunta (ex: 'quais são os tipos de dados?', 'existem outliers em Amount?', 'importância das variáveis?')")

    if user_q:
        answer, conclusion, interpretation, visuals = agent.answer_question(user_q)

        st.markdown("### 🤖 Resposta")
        st.write(answer)

        if conclusion:
            st.markdown("### 📝 Conclusão (interpretativa)")
            st.markdown(f"**{conclusion}**")

        if interpretation:
            st.markdown("### 📖 Interpretação adicional")
            st.markdown(f"**{interpretation}**")

        if visuals:
            st.markdown("### 📊 Visualização gerada(s)")
            for i, path in enumerate(visuals, 1):
                if os.path.exists(path):
                    st.image(path, caption=f"Visualização {i}", use_container_width=True)

    # Histórico
    if agent.history:
        st.markdown("---")
        st.markdown("### 📚 Histórico de Conclusões do Agente")
        for h in agent.history:
            st.markdown(f"- [{h['time']}] {h['text']}")

