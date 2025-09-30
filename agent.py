import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import plotly.express as px
from datetime import datetime


class EvolvedAgent:
    def __init__(self):
        self.df = None
        self.filename = None
        self.history = []

    # ---------------- Loaders ----------------
    def load_dataframe(self, df: pd.DataFrame, name="dataset.csv"):
        """Carrega o DataFrame e inicializa"""
        self.df = df
        self.filename = name
        msg = f"✅ Arquivo {name} carregado com {df.shape[0]} linhas e {df.shape[1]} colunas."
        self._add_history(msg)
        return msg

    # ---------------- Memory ----------------
    def _add_history(self, text: str):
        self.history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             "text": text})

    # ---------------- Q&A ----------------
    def answer_question(self, question: str):
        if self.df is None:
            return "⚠️ Nenhum dataset carregado.", "", "", []

        q = question.lower()
        visuals = []
        answer, conclusion, interpretation = "", "", ""

        try:
            # 1) Tipos de dados
            if "tipo" in q and "dado" in q:
                types = self.df.dtypes.value_counts().to_dict()
                answer = f"Tipos de dados detectados: {types}"
                conclusion = f"O dataset contém {types}."
                interpretation = "Isso indica a quantidade de variáveis numéricas e categóricas disponíveis."

            # 2) Distribuição
            elif "distribui" in q or "histograma" in q:
                num_cols = self.df.select_dtypes(include=np.number).columns[:6]
                for col in num_cols:
                    fig = px.histogram(self.df, x=col, nbins=50, title=f"Distribuição de {col}")
                    path = f"outputs/dist_{col}.png"
                    fig.write_image(path)
                    visuals.append(path)
                answer = f"Histogramas gerados para: {', '.join(num_cols)}"
                conclusion = "Distribuições numéricas calculadas."
                interpretation = "Os histogramas mostram como os valores estão concentrados ou dispersos."

            # 3) Intervalo
            elif "intervalo" in q or ("mínimo" in q and "máximo" in q):
                desc = self.df.describe().T[["min", "max"]]
                answer = desc.to_dict()
                conclusion = "Intervalos calculados para variáveis numéricas."
                interpretation = "Os valores mínimos e máximos ajudam a entender a escala de cada variável."

            # 4) Tendência central
            elif "média" in q or "mediana" in q:
                desc = self.df.describe().T[["mean"]]
                desc["median"] = self.df.median(numeric_only=True)
                answer = desc.to_dict()
                conclusion = "Médias e medianas calculadas."
                interpretation = "A média mostra a tendência central, enquanto a mediana reduz efeito de outliers."

            # 5) Variabilidade
            elif "variância" in q or "desvio" in q:
                var = self.df.var(numeric_only=True)
                std = self.df.std(numeric_only=True)
                answer = {"variância": var.to_dict(), "desvio": std.to_dict()}
                conclusion = "Variabilidade (variância e desvio padrão) calculada."
                interpretation = "Alta variância/desvio indica grande dispersão dos dados."

            # 6) Padrões temporais
            elif "tempo" in q or "tendên" in q:
                if "time" in self.df.columns or "Time" in self.df.columns:
                    col = "Time" if "Time" in self.df.columns else "time"
                    fig = px.line(self.df.head(1000), y=col, title="Padrões Temporais (amostra)")
                    path = "outputs/padrao_temporal.png"
                    fig.write_image(path)
                    visuals.append(path)
                    answer = f"Gráfico temporal gerado para {col}"
                    conclusion = "Tendências temporais identificadas."
                    interpretation = "É possível observar padrões de variação ao longo do tempo."
                else:
                    answer = "Nenhuma coluna temporal encontrada."

            # 7) Frequência
            elif "frequente" in q:
                cat_cols = self.df.select_dtypes(exclude=np.number).columns
                if len(cat_cols) > 0:
                    col = cat_cols[0]
                    freq = self.df[col].value_counts().head(5).to_dict()
                    answer = f"Top frequências em {col}: {freq}"
                    conclusion = f"As categorias mais frequentes foram identificadas na coluna {col}."
                    interpretation = "As frequências mostram quais valores aparecem mais ou menos vezes."
                else:
                    answer = "Nenhuma coluna categórica encontrada."

            # 8) Clusters
            elif "cluster" in q or "agrupamento" in q:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                cols = self.df.select_dtypes(include=np.number).iloc[:, :5]
                self.df["cluster"] = km.fit_predict(cols)
                table = self.df.groupby("cluster")["Class"].value_counts(normalize=True).unstack().fillna(0).round(3)
                answer = table.to_dict()
                conclusion = f"Clusters KMeans (k=3) identificados."
                interpretation = "Os clusters mostram grupos de observações semelhantes. Alguns podem concentrar fraudes."
                fig = px.scatter_matrix(cols.assign(cluster=self.df["cluster"]),
                                        dimensions=cols.columns,
                                        color=self.df["cluster"].astype(str),
                                        title="Clusters")
                path = "outputs/clusters.png"
                fig.write_image(path)
                visuals.append(path)

            # 9) Outliers
            elif "outlier" in q or "atípic" in q:
                iso = IsolationForest(contamination=0.001, random_state=42)
                preds = iso.fit_predict(self.df.select_dtypes(include=np.number).sample(1000))
                outliers = (preds == -1).sum()
                answer = f"Foram detectados {outliers} outliers em 1000 amostras."
                conclusion = "Outliers detectados com IsolationForest."
                interpretation = "Outliers são pontos fora do padrão; podem indicar erros ou casos especiais."

            # 10) Correlação
            elif "correla" in q or "relacion" in q:
                corr = self.df.corr(numeric_only=True)["Class"].sort_values(ascending=False).head(6)
                answer = f"Principais correlações com fraudes: {corr.to_dict()}"
                conclusion = "Variáveis mais associadas às fraudes calculadas."
                interpretation = "Correlação mostra como variáveis variam juntas. Valores próximos de 1 ou -1 indicam forte relação."
                fig = px.imshow(self.df.corr(numeric_only=True),
                                color_continuous_scale="RdBu_r",
                                title="Mapa de Correlação")
                path = "outputs/correlacao.png"
                fig.write_image(path)
                visuals.append(path)

            # 11) Importância variáveis
            elif "importânc" in q or "influênc" in q:
                X = self.df.drop(columns=["Class"])
                y = self.df["Class"]
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
                answer = f"Principais variáveis: {imp.to_dict()}"
                conclusion = "Importância das variáveis calculada com RandomForest."
                interpretation = "Mostra quais variáveis mais ajudam a prever fraudes."

            else:
                answer = "🤖 Pergunta não reconhecida. Reformule."

        except Exception as e:
            answer = f"Erro na análise: {e}"

        # salvar histórico
        self._add_history(conclusion if conclusion else answer)

        return answer, conclusion, interpretation, visuals
