"""Benchmark DGA — protocolo único e reprodutível para todos os classificadores.

Reproduz e compara, sob avaliação idêntica, os métodos estudados na dissertação de
Otacílio Rodrigues Oliveira Filho (UFPA, 2024): métodos convencionais de análise DGA
(Duval, Rogers, Gás Chave, Doernenburg, IEC 60599) e métodos de aprendizado de máquina
(Regressão Logística, SVM, KNN, ANN), mais um baseline de boosting.

Ver ResultadosNovos/README.md para a lista de problemas metodológicos da implementação
anterior (rascunho_v1/) que este script corrige por construção.

Stack: apenas scikit-learn, scipy, pandas, numpy, matplotlib.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# "indeterminado" é uma 4a classe de predição válida (métodos convencionais),
# que nunca aparece em y_true por construção — o aviso do sklearn sobre
# "y_pred contains classes not in y_true" é esperado e não indica um bug.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================

SEMENTE = 42
N_SPLITS, N_REPEATS = 10, 5
GASES = ["H2", "CH4", "C2H4", "C2H6", "C2H2"]
CAMINHO_DADOS = Path(__file__).resolve().parent.parent / "DataSet" / "DataSetDGA.xlsx"
COLUNA_ROTULO = "Rótulo"
SAIDA = Path(__file__).resolve().parent / "resultados"

# NO -> normal; T1/T2/T3 -> T; D1/D2/PD -> D (mesmo agrupamento da dissertação,
# Capítulo 6, e idêntico à coluna NTE já presente no dataset).
MAPA_CLASSE3 = {
    "NO": "normal",
    "T1": "T",
    "T2": "T",
    "T3": "T",
    "D1": "D",
    "D2": "D",
    "PD": "D",
}

# O dataset já chega com o piso de 0,01 ppm aplicado (não há nenhum zero literal:
# min() de todo gás é exatamente 0.01). Não há como recuperar os zeros originais,
# então tratamos o valor 0.01 como sentinela de "não detectado" em vez de 0.0.
SENTINELA_ZERO = 0.01


def carregar_dados(caminho: Path = CAMINHO_DADOS) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retorna (X com os 5 gases em ppm, y em {normal,T,D}, grupo = composição de gases)."""
    df = pd.read_excel(caminho)
    y = df[COLUNA_ROTULO].map(MAPA_CLASSE3)
    if y.isna().any():
        rotulos_invalidos = sorted(df.loc[y.isna(), COLUNA_ROTULO].unique())
        raise ValueError(f"Rótulos fora do mapa esperado: {rotulos_invalidos}")
    X = df[GASES].copy()
    # Grupo de deduplicação para StratifiedGroupKFold: linhas com composição de
    # gases idêntica (13,2% do dataset, ver auditoria) nunca podem ficar uma em
    # treino e outra em teste no mesmo fold.
    grupo = X.apply(lambda linha: hash(tuple(linha)), axis=1)
    return X, y, grupo


# ============================================================================
# 2. PRÉ-PROCESSAMENTO COMO TRANSFORMADORES DE PIPELINE
# ============================================================================


class TrataZeros(BaseEstimator, TransformerMixin):
    """Trata o piso de detecção (sentinela 0,01 ppm) antes do log.

    estrategia:
      "log1p"      (padrão) — log1p aplicado direto; o piso vira log1p(0.01)≈0.00995,
                    um valor pequeno mas não patológico.
      "lod_metade" — substitui o sentinela por metade do limite de detecção por gás
                    (usa os limites L1 da Tabela-5/Doernenburg como proxy de LOD),
                    depois aplica log.
      "original"   — reproduz o artefato do trabalho anterior: mantém 0,01 como valor
                    de concentração (sem tratamento) e aplica log. Existe só para a
                    ablação que demonstra o cluster espúrio; não usar em MODELOS.
    """

    LOD_METADE = {"H2": 50.0, "CH4": 60.0, "C2H4": 25.0, "C2H6": 32.5, "C2H2": 0.5}

    def __init__(self, estrategia: str = "log1p"):
        self.estrategia = estrategia

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=GASES).copy()
        if self.estrategia == "log1p":
            return np.log1p(X.to_numpy())
        if self.estrategia == "lod_metade":
            for gas in GASES:
                eh_sentinela = np.isclose(X[gas], SENTINELA_ZERO)
                X.loc[eh_sentinela, gas] = self.LOD_METADE[gas]
            return np.log1p(X.to_numpy())
        if self.estrategia == "original":
            return np.log(X.to_numpy())
        raise ValueError(f"estrategia desconhecida: {self.estrategia!r}")


class RelacoesGases(BaseEstimator, TransformerMixin):
    """Razões de Rogers como atributos derivados (sobre ppm bruto, antes do log)."""

    NOMES = ["CH4_H2", "C2H2_C2H4", "C2H4_C2H6", "C2H2_CH4"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=GASES)

        def razao(num, den):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = num / den
            return np.where(den == 0, np.where(num == 0, 0.0, np.inf), r)

        return np.column_stack(
            [
                razao(X["CH4"], X["H2"]),
                razao(X["C2H2"], X["C2H4"]),
                razao(X["C2H4"], X["C2H6"]),
                razao(X["C2H2"], X["CH4"]),
            ]
        )


class PercentuaisDuval(BaseEstimator, TransformerMixin):
    """%CH4, %C2H4, %C2H2 relativos à soma dos três (Equações 2.1-2.3 da dissertação)."""

    NOMES = ["pct_CH4", "pct_C2H4", "pct_C2H2"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=GASES)
        total = X["CH4"] + X["C2H4"] + X["C2H2"]
        total = total.replace(0, np.nan)
        pct = pd.DataFrame(
            {
                "pct_CH4": X["CH4"] / total * 100,
                "pct_C2H4": X["C2H4"] / total * 100,
                "pct_C2H2": X["C2H2"] / total * 100,
            }
        ).fillna(0.0)
        return pct.to_numpy()


# ============================================================================
# 3. MÉTODOS CONVENCIONAIS COMO ESTIMADORES SKLEARN
# ============================================================================
# Todos recebem ppm bruto (5 colunas de GASES, nesta ordem) e não aprendem nada
# de fit(): são determinísticos, implementando regras normativas fixas.
# O dataset não possui a coluna CO — Doernenburg e Gás Chave usam apenas os 5
# gases disponíveis, exatamente como a dissertação também fez para Doernenburg
# ("o CO... não foi levado em consideração no diagnóstico").


class _MetodoConvencional(BaseEstimator, ClassifierMixin):
    suporta_normal: bool = True

    def fit(self, X, y=None):
        self.classes_ = np.array(["normal", "T", "D", "indeterminado"])
        return self

    def predict(self, X):
        X = pd.DataFrame(X, columns=GASES)
        return np.array([self._diagnostico(linha) for _, linha in X.iterrows()])

    def _diagnostico(self, linha) -> str:  # pragma: no cover - implementado nas subclasses
        raise NotImplementedError


def _razao(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return num / den


class Rogers(_MetodoConvencional):
    """Tabela-3: R1=C2H2/C2H4, R2=CH4/H2, R3=C2H4/C2H6 (IEEE C57.104:2019, ap. Ikeshoji 2020)."""

    suporta_normal = True

    def _diagnostico(self, linha) -> str:
        h2, ch4, c2h4, c2h6, c2h2 = linha["H2"], linha["CH4"], linha["C2H4"], linha["C2H6"], linha["C2H2"]
        r1, r2, r3 = _razao(c2h2, c2h4), _razao(ch4, h2), _razao(c2h4, c2h6)

        if r1 < 0.1 and 0.1 <= r2 <= 1.0 and r3 < 0.1:
            return "normal"
        if r1 < 0.1 and r2 < 0.1 and r3 < 0.1:
            return "D"  # Descarga de baixa energia (DP)
        if 0.1 <= r1 <= 3.0 and 0.1 <= r2 <= 1.0 and r3 > 3.0:
            return "D"  # Descarga de alta energia (Arco)
        if r1 < 0.1 and 0.1 <= r2 <= 1.0 and 0.1 <= r3 <= 3.0:
            return "T"  # Falta térmica de baixa temperatura
        if r1 < 0.1 and r2 > 1.0 and 0.1 <= r3 <= 3.0:
            return "T"  # Falta térmica T<700°C
        if r1 < 0.1 and r2 > 1.0 and r3 > 3.0:
            return "T"  # Falta térmica T>700°C
        return "indeterminado"


class IEC60599(_MetodoConvencional):
    """Tabela-7: razões C2H2/C2H4, CH4/H2, C2H4/C2H6 (IEC 60599:2015). Sem diagnóstico normal."""

    suporta_normal = False

    def _diagnostico(self, linha) -> str:
        h2, ch4, c2h4, c2h6, c2h2 = linha["H2"], linha["CH4"], linha["C2H4"], linha["C2H6"], linha["C2H2"]
        r1, r2, r3 = _razao(c2h2, c2h4), _razao(ch4, h2), _razao(c2h4, c2h6)

        if r2 < 0.1 and r3 < 0.2:
            return "D"  # Descarga parcial
        if r1 > 1.0 and 0.1 <= r2 <= 0.5 and r3 > 1.0:
            return "D"  # Baixa energia
        if 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1.0 and r3 > 2.0:
            return "D"  # Alta energia
        if r2 > 1.0 and r3 < 1.0:
            return "T"  # Térmica T<300°C (r1 = NS)
        if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 <= 4.0:
            return "T"  # Térmica T<700°C
        if r1 < 0.2 and r2 > 1.0 and r3 > 4.0:
            return "T"  # Térmica T>700°C
        return "indeterminado"


class Doernenburg(_MetodoConvencional):
    """Tabelas 5 e 6: três pré-condições + 4 razões (IEEE C57.104:2019, ap. Ikeshoji 2020).

    Pré-condição 1: ao menos 1 gás > 2×L1 (senão: normal).
    Pré-condição 3: em cada razão, ao menos 1 dos 2 gases envolvidos > L1
    ("ao menos um dos gases de cada relação exceda a concentração limite", texto da
    dissertação) — senão: indeterminado (recomenda nova amostra).
    """

    suporta_normal = True
    L1 = {"H2": 100, "CH4": 120, "C2H4": 50, "C2H6": 65, "C2H2": 1}

    def _diagnostico(self, linha) -> str:
        gases = {g: linha[g] for g in GASES}

        if not any(gases[g] > 2 * self.L1[g] for g in self.L1):
            return "normal"

        sig = {g: gases[g] > self.L1[g] for g in self.L1}
        pares = {
            "R1": ("CH4", "H2"),
            "R2": ("C2H2", "C2H4"),
            "R3": ("C2H2", "CH4"),
            "R4": ("C2H6", "C2H2"),
        }
        if not all(sig[a] or sig[b] for a, b in pares.values()):
            return "indeterminado"

        r1 = _razao(gases["CH4"], gases["H2"])
        r2 = _razao(gases["C2H2"], gases["C2H4"])
        r3 = _razao(gases["C2H2"], gases["CH4"])
        r4 = _razao(gases["C2H6"], gases["C2H2"])

        if r1 > 1.0 and r2 < 0.75 and r3 < 0.3 and r4 > 0.4:
            return "T"  # Decomposição térmica
        if r1 < 0.1 and r3 < 0.3 and r4 > 0.4:
            return "D"  # Descarga de baixa energia (R2 = NS na Tabela-6)
        if 0.1 <= r1 <= 1.0 and r2 > 0.75 and r3 > 0.3 and r4 < 0.4:
            return "D"  # Descarga de alta energia
        return "indeterminado"


class GasChave(_MetodoConvencional):
    """Tabela-4 (IEEE C57.104:2019). Sem valores-limite numéricos publicados para
    automação — o próprio texto da dissertação registra que "o Gás chave tende a
    apresentar baixa acurácia quando aplicado de forma automática". Implementação:
    gás de maior concentração relativa entre os 3 disponíveis (C2H4, H2, C2H2) define
    o diagnóstico; se o gás dominante entre os 5 é CH4 ou C2H6 (não listados como
    "gás chave"), não há indicação de falha -> normal.
    """

    suporta_normal = True
    CHAVE_PARA_CLASSE = {"C2H4": "T", "H2": "D", "C2H2": "D"}

    def _diagnostico(self, linha) -> str:
        valores = {g: linha[g] for g in GASES}
        if sum(valores.values()) == 0:
            return "indeterminado"
        dominante = max(valores, key=valores.get)
        return self.CHAVE_PARA_CLASSE.get(dominante, "normal")


class Duval1(_MetodoConvencional):
    """Triângulo de Duval 1 (Figura-8, Equações 2.1-2.3). Sem diagnóstico normal.

    Vértices das 7 regiões (em %C2H4, %CH4) conferidos visualmente contra a
    Figura-8 do PDF da dissertação (Publicacoes/Dissertacao_Otacilio.pdf, p. 17):
    topo = 100% CH4, vértice inferior-esquerdo = 100% C2H2, inferior-direito = 100% C2H4.
    A região DT (falha mista) não mapeia para {T,D} e sai como "indeterminado".
    """

    suporta_normal = False

    REGIOES = {
        "PD": [(0, 98), (0, 100), (2, 98)],
        "T1": [(0, 96), (0, 98), (2, 98), (20, 80), (20, 76)],
        "T2": [(20, 76), (20, 80), (50, 50), (50, 46)],
        "T3": [(50, 35), (50, 50), (100, 0), (85, 0)],
        "D1": [(0, 0), (0, 87), (23, 64), (23, 0)],
        "D2": [(23, 0), (23, 64), (40, 47), (40, 31), (71, 0)],
        "DT": [(0, 87), (0, 96), (50, 46), (50, 35), (85, 0), (71, 0), (40, 31), (40, 47)],
    }
    REGIAO_PARA_CLASSE = {
        "PD": "D", "D1": "D", "D2": "D",
        "T1": "T", "T2": "T", "T3": "T",
        "DT": "indeterminado",
    }

    @staticmethod
    def _ponto_no_poligono(x: float, y: float, poligono: list) -> bool:
        dentro = False
        n = len(poligono)
        for i in range(n):
            x1, y1 = poligono[i]
            x2, y2 = poligono[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_intersecao = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < x_intersecao:
                    dentro = not dentro
        return dentro

    def _diagnostico(self, linha) -> str:
        ch4, c2h4, c2h2 = linha["CH4"], linha["C2H4"], linha["C2H2"]
        total = ch4 + c2h4 + c2h2
        if total == 0:
            return "indeterminado"
        pct_ch4 = ch4 / total * 100
        pct_c2h4 = c2h4 / total * 100
        for regiao, poligono in self.REGIOES.items():
            if self._ponto_no_poligono(pct_c2h4, pct_ch4, poligono):
                return self.REGIAO_PARA_CLASSE[regiao]
        if pct_c2h4 <= 0 and pct_ch4 >= 96:
            return self.REGIAO_PARA_CLASSE["T1"]
        return "indeterminado"


# ============================================================================
# TESTE COM CASOS CANÔNICOS (Tabela-2 da dissertação, base IEC TC10:1999)
# ============================================================================

CASOS_CANONICOS = pd.DataFrame(
    [
        {"caso": "Normal (Tabela-2)", "H2": 95, "CH4": 280, "C2H4": 150, "C2H6": 250, "C2H2": 10,
         "esperado": "Duval indica falha mesmo sendo normal; Rogers/Doernenburg: térmica; Gás chave: normal"},
        {"caso": "Falha T3 (Tabela-2)", "H2": 2500, "CH4": 10500, "C2H4": 13500, "C2H6": 4790, "C2H2": 6,
         "esperado": "Duval: T3; Rogers: térmica T<700 (subestima); Doernenburg: térmica; "
                     "IEC 60599: térmica T<700 (300-700°C); Gás chave: térmica"},
    ]
)


def testar_casos_canonicos() -> None:
    metodos = {
        "Duval1": Duval1(),
        "Rogers": Rogers(),
        "GasChave": GasChave(),
        "Doernenburg": Doernenburg(),
        "IEC60599": IEC60599(),
    }
    X = CASOS_CANONICOS[GASES]
    print(f"{'Caso':<22}" + "".join(f"{nome:<14}" for nome in metodos))
    for i, caso in CASOS_CANONICOS["caso"].items():
        linha = X.iloc[[i]]
        preds = []
        for nome, modelo in metodos.items():
            modelo.fit(linha)
            preds.append(modelo.predict(linha)[0])
        print(f"{caso:<22}" + "".join(f"{p:<14}" for p in preds))
        print(f"  esperado: {CASOS_CANONICOS.loc[i, 'esperado']}")


# ============================================================================
# 4. PRÉ-PROCESSADORES E DICIONÁRIO CENTRAL DE MODELOS
# ============================================================================

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PRE_NULO = "passthrough"

PRE_BASICO = Pipeline([
    ("zeros", TrataZeros(estrategia="log1p")),
    ("escala", StandardScaler()),
])

PRE_ESTENDIDO = Pipeline([
    ("uniao", FeatureUnion([
        ("zeros_log", TrataZeros(estrategia="log1p")),
        ("relacoes", RelacoesGases()),
        ("duval_pct", PercentuaisDuval()),
    ])),
    ("escala", StandardScaler()),
])


def _pipeline_convencional(modelo: _MetodoConvencional) -> Pipeline:
    return Pipeline([("pre", PRE_NULO), ("clf", modelo)])


# nota sobre a ANN: MLPClassifier não tem ativação "selu" nem "class_weight"
# (limitação do sklearn frente à arquitetura Keras original da dissertação,
# documentada no README). O desbalanceamento é compensado via class_weight nos
# demais modelos e via early_stopping aqui.
MODELOS = {
    # --- convencionais (ppm bruto, sem pré-processamento, sem sintonia) ---
    "Duval": _pipeline_convencional(Duval1()),
    "Rogers": _pipeline_convencional(Rogers()),
    "GasChave": _pipeline_convencional(GasChave()),
    "Doernenburg": _pipeline_convencional(Doernenburg()),
    "IEC60599": _pipeline_convencional(IEC60599()),
    # --- aprendizado de máquina ---
    "LR": Pipeline([("pre", PRE_BASICO), ("clf", LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEMENTE))]),
    "SVM": Pipeline([("pre", PRE_ESTENDIDO), ("clf", SVC(
        kernel="rbf", class_weight="balanced", random_state=SEMENTE))]),
    "KNN": Pipeline([("pre", PRE_BASICO), ("clf", KNeighborsClassifier())]),
    "ANN": Pipeline([("pre", PRE_BASICO), ("clf", MLPClassifier(
        hidden_layer_sizes=(12, 12, 8, 6), activation="relu", solver="adam",
        early_stopping=True, max_iter=500, random_state=SEMENTE))]),
    "HistGB": Pipeline([("pre", PRE_BASICO), ("clf", HistGradientBoostingClassifier(
        random_state=SEMENTE))]),
    # --- variantes para isolar o efeito da representação de entrada ---
    "SVM_5gases": Pipeline([("pre", PRE_BASICO), ("clf", SVC(
        kernel="rbf", class_weight="balanced", random_state=SEMENTE))]),
    "KNN_estend": Pipeline([("pre", PRE_ESTENDIDO), ("clf", KNeighborsClassifier())]),
}

# Pares (nome_basico, nome_estendido) com o MESMO classificador, usados na
# ablação de representação (item 3 dos experimentos opcionais).
PARES_REPRESENTACAO = [("KNN", "KNN_estend"), ("SVM_5gases", "SVM")]


# ============================================================================
# 5. SINTONIA DE HIPERPARÂMETROS (busca no laço interno de uma CV aninhada)
# ============================================================================

from scipy.stats import loguniform, randint

N_ITER_BUSCA = 10
N_SPLITS_INTERNO = 3

ESPACOS = {
    "LR": {"clf__C": loguniform(1e-2, 1e2)},
    "SVM": {
        "clf__C": loguniform(1e-1, 1e3),
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__kernel": ["rbf", "linear", "poly", "sigmoid"],
    },
    "KNN": {"clf__n_neighbors": randint(3, 30), "clf__weights": ["uniform", "distance"]},
    "ANN": {
        "clf__alpha": loguniform(1e-5, 1e-1),
        "clf__learning_rate_init": loguniform(1e-4, 1e-1),
    },
    "HistGB": {
        "clf__max_iter": randint(50, 300),
        "clf__max_depth": randint(2, 10),
        "clf__learning_rate": loguniform(1e-2, 3e-1),
    },
}
ESPACOS["SVM_5gases"] = ESPACOS["SVM"]
ESPACOS["KNN_estend"] = ESPACOS["KNN"]
# Duval, Rogers, GasChave, Doernenburg, IEC60599: sem entrada em ESPACOS -> sem sintonia
# (são determinísticos, não têm hiperparâmetros).

# Limitação documentada: a busca interna usa StratifiedKFold simples (SEM
# agrupamento por composição de gases), diferente da CV externa. Isso pode
# enviesar um pouco a escolha de hiperparâmetros (uma duplicata pode cair em
# treino-interno e validação-interna da mesma busca), mas NUNCA vaza para a
# avaliação externa reportada — o StratifiedGroupKFold externo garante que
# cada grupo de composição idêntica fica inteiro em um único fold de teste.
# Implementar agrupamento também na busca interna exigiria roteamento de
# metadados (groups) do sklearn, o que aumentaria muito a complexidade deste
# script único; ver README para essa limitação.


# ============================================================================
# 6. LAÇO DE AVALIAÇÃO — CV EXTERNA IDÊNTICA PARA TODOS OS MODELOS
# ============================================================================

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

CLASSES3 = ["normal", "T", "D"]


def avaliar_todos(
    modelos: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    grupo: pd.Series,
    espacos: dict | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    """Avalia cada modelo em MODELOS sob a mesma sequência de CVs externas.

    Retorna (predicoes_por_repeticao, predicoes_repeticao_0). A repetição 0 é
    a usada para os testes pareados (McNemar, Kappa) e para os CSVs de
    predições exportados — precisa ser a mesma partição para todos os
    modelos, o que só é garantido porque o mesmo objeto `cv_externo` (mesma
    semente) é usado para todos eles dentro de cada repetição.
    """
    espacos = espacos or {}
    pasta_pred = SAIDA / "predicoes"
    pasta_pred.mkdir(parents=True, exist_ok=True)

    predicoes_por_repeticao: dict[str, list[np.ndarray]] = {nome: [] for nome in modelos}
    predicoes_repeticao_0: dict[str, np.ndarray] = {}

    for r in range(n_repeats):
        cv_externo = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEMENTE + r)
        for nome, pipeline in modelos.items():
            estimador = pipeline
            paralelo_externo = -1
            if nome in espacos:
                cv_interno = StratifiedKFold(n_splits=N_SPLITS_INTERNO, shuffle=True, random_state=SEMENTE + r)
                estimador = RandomizedSearchCV(
                    pipeline,
                    espacos[nome],
                    n_iter=N_ITER_BUSCA,
                    cv=cv_interno,
                    scoring="balanced_accuracy",
                    random_state=SEMENTE + r,
                    n_jobs=-1,
                )
                paralelo_externo = 1  # evita paralelismo aninhado (busca já usa -1)

            pred = cross_val_predict(
                estimador, X, y, groups=grupo, cv=cv_externo, n_jobs=paralelo_externo
            )
            assert len(pred) == len(X), (
                f"{nome} (repetição {r}): total de predições {len(pred)} != {len(X)} amostras "
                "— toda avaliação deve cobrir 100% do dataset, nunca um subconjunto."
            )
            predicoes_por_repeticao[nome].append(pred)
            if r == 0:
                predicoes_repeticao_0[nome] = pred
                pd.Series(pred, name="predicao").to_csv(pasta_pred / f"{nome}.csv", index=False)
            print(f"  [repeticao {r + 1}/{n_repeats}] {nome}: concluido")

    # Verificação de regressão obrigatória: todo modelo deve ter sido avaliado
    # sobre o dataset inteiro, em toda repetição — nunca subconjuntos distintos.
    totais = {
        nome: [len(pred) for pred in preds] for nome, preds in predicoes_por_repeticao.items()
    }
    for nome, lista in totais.items():
        assert lista == [len(X)] * n_repeats, f"Totais inconsistentes em {nome}: {lista}"

    return predicoes_por_repeticao, predicoes_repeticao_0


def metricas_uma_repeticao(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cobertos = y_pred != "indeterminado"

    resultado = {
        "acuracia": accuracy_score(y_true, y_pred),
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASSES3, average="macro", zero_division=0),
        "kappa_vs_verdade": cohen_kappa_score(y_true, y_pred, labels=CLASSES3 + ["indeterminado"]),
        "taxa_indeterminacao": float((~cobertos).mean()),
        "acuracia_condicional_cobertura": (
            accuracy_score(y_true[cobertos], y_pred[cobertos]) if cobertos.any() else float("nan")
        ),
    }
    for classe in CLASSES3:
        resultado[f"precisao_{classe}"] = precision_score(
            y_true, y_pred, labels=[classe], average="macro", zero_division=0
        )
        resultado[f"revocacao_{classe}"] = recall_score(
            y_true, y_pred, labels=[classe], average="macro", zero_division=0
        )
    return resultado


def tabela_metricas(
    predicoes_por_repeticao: dict[str, list[np.ndarray]], y: pd.Series
) -> pd.DataFrame:
    y_arr = y.to_numpy()
    linhas = []
    for nome, repeticoes in predicoes_por_repeticao.items():
        df_rep = pd.DataFrame([metricas_uma_repeticao(y_arr, pred) for pred in repeticoes])
        media = df_rep.mean().add_suffix("_media")
        desvio = df_rep.std().add_suffix("_desvio")
        linha = pd.concat([media, desvio]).to_dict()
        linha["modelo"] = nome
        linhas.append(linha)
    df = pd.DataFrame(linhas).set_index("modelo")
    colunas_ordenadas = sorted(df.columns, key=lambda c: (c.rsplit("_", 1)[0], c.rsplit("_", 1)[1]))
    return df[colunas_ordenadas]


# ============================================================================
# 7. TESTES ESTATÍSTICOS PAREADOS (repetição 0)
# ============================================================================

from itertools import combinations

from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score


def _mcnemar_par(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, int, int]:
    correta_a = pred_a == y_true
    correta_b = pred_b == y_true
    b = int(np.sum(correta_a & ~correta_b))
    c = int(np.sum(~correta_a & correta_b))
    if b + c == 0:
        return 1.0, b, c
    p = binomtest(min(b, c), b + c, p=0.5).pvalue
    return p, b, c


def _holm(pvalores: list[float]) -> list[float]:
    n = len(pvalores)
    ordem = np.argsort(pvalores)
    ajustado = np.empty(n)
    maior_ate_agora = 0.0
    for rank, idx in enumerate(ordem):
        valor = (n - rank) * pvalores[idx]
        maior_ate_agora = max(maior_ate_agora, valor)
        ajustado[idx] = min(maior_ate_agora, 1.0)
    return ajustado.tolist()


def testes_pareados(predicoes_repeticao_0: dict[str, np.ndarray], y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar (com correção de Holm) e Kappa de Cohen par a par, sobre a
    mesma repetição-0 (mesmas partições de teste para todos os modelos)."""
    y_arr = y.to_numpy()
    nomes = list(predicoes_repeticao_0.keys())
    pares = list(combinations(nomes, 2))

    linhas_mcnemar = []
    for a, b in pares:
        p, disc_a, disc_b = _mcnemar_par(y_arr, predicoes_repeticao_0[a], predicoes_repeticao_0[b])
        linhas_mcnemar.append({"modelo_a": a, "modelo_b": b, "p_valor": p, "so_a_acerta": disc_a, "so_b_acerta": disc_b})
    df_mcnemar = pd.DataFrame(linhas_mcnemar)
    df_mcnemar["p_valor_holm"] = _holm(df_mcnemar["p_valor"].tolist())

    kappa_matriz = pd.DataFrame(index=nomes, columns=nomes, dtype=float)
    for a in nomes:
        for b in nomes:
            kappa_matriz.loc[a, b] = cohen_kappa_score(
                predicoes_repeticao_0[a], predicoes_repeticao_0[b], weights="linear"
            )

    return df_mcnemar, kappa_matriz


def intervalo_wilson(acertos: int, total: int) -> tuple[float, float]:
    ic = binomtest(acertos, total).proportion_ci(method="wilson")
    return ic.low, ic.high


# ============================================================================
# 8. EXPORTAÇÃO — TABELAS, LATEX E FIGURAS
# ============================================================================


def exportar_tudo(
    df_metricas: pd.DataFrame,
    df_mcnemar: pd.DataFrame,
    kappa_matriz: pd.DataFrame,
    predicoes_repeticao_0: dict[str, np.ndarray],
    y: pd.Series,
) -> None:
    pasta_tabelas = SAIDA / "tabelas"
    pasta_figuras = SAIDA / "figuras"
    pasta_tabelas.mkdir(parents=True, exist_ok=True)
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    df_metricas.to_csv(pasta_tabelas / "metricas_globais.csv")
    df_mcnemar.to_csv(pasta_tabelas / "mcnemar_holm.csv", index=False)
    kappa_matriz.to_csv(pasta_tabelas / "kappa_pares.csv")

    # Tabela I — desempenho global (formato próximo ao da Tabela-17 da dissertação)
    colunas_i = [
        "acuracia_media", "acuracia_desvio",
        "acuracia_balanceada_media", "acuracia_balanceada_desvio",
        "f1_macro_media", "f1_macro_desvio",
        "taxa_indeterminacao_media",
    ]
    tabela_i = df_metricas[colunas_i].copy()
    with open(pasta_tabelas / "tabela_I.tex", "w") as f:
        f.write(_para_latex(
            tabela_i,
            legenda="Desempenho global por método (média $\\pm$ desvio-padrão entre 5 repetições "
            "de CV externa de 10 folds, agrupada por composição de gases).",
            notas=[
                "Acurácia balanceada: média da revocação por classe (compensa o desbalanceamento 65\\%/22\\%/13\\%).",
                "Taxa de indeterminação: fração de amostras sem diagnóstico definido (0 para os métodos de AM).",
            ],
        ))

    # Tabela II — precisão/revocação por classe
    colunas_ii = [c for c in df_metricas.columns if c.startswith(("precisao_", "revocacao_")) and c.endswith("_media")]
    tabela_ii = df_metricas[colunas_ii].copy()
    with open(pasta_tabelas / "tabela_II.tex", "w") as f:
        f.write(_para_latex(
            tabela_ii,
            legenda="Precisão e revocação por classe (normal/T/D), média entre 5 repetições.",
            notas=["Revocação aqui equivale à sensibilidade por classe usada na dissertação original."],
        ))

    _figura_matrizes_confusao(predicoes_repeticao_0, y, pasta_figuras / "matrizes_confusao.pdf")
    _figura_boxplot(df_metricas, pasta_figuras / "boxplot_acuracia.pdf")


def _para_latex(df: pd.DataFrame, legenda: str, notas: list[str]) -> str:
    corpo = df.round(4).to_latex(escape=True)
    notas_txt = "\n".join(f"% {n}" for n in notas)
    return (
        "% Gerado automaticamente por benchmark_dga.py — não editar à mão.\n"
        f"{notas_txt}\n"
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{legenda}}}\n"
        f"{corpo}\n"
        "\\end{table}\n"
    )


def _figura_matrizes_confusao(predicoes_repeticao_0: dict[str, np.ndarray], y: pd.Series, caminho: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    rotulos = CLASSES3 + ["indeterminado"]
    nomes = list(predicoes_repeticao_0.keys())
    n = len(nomes)
    cols = 4
    linhas_fig = -(-n // cols)
    fig, eixos = plt.subplots(linhas_fig, cols, figsize=(4 * cols, 3.5 * linhas_fig))
    eixos = np.array(eixos).reshape(-1)
    for i, nome in enumerate(nomes):
        m = confusion_matrix(y, predicoes_repeticao_0[nome], labels=rotulos)
        ax = eixos[i]
        ax.imshow(m, cmap="Blues")
        ax.set_title(nome, fontsize=10)
        ax.set_xticks(range(len(rotulos)))
        ax.set_xticklabels(rotulos, rotation=45, fontsize=7)
        ax.set_yticks(range(len(rotulos)))
        ax.set_yticklabels(rotulos, fontsize=7)
        for (r, c), v in np.ndenumerate(m):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=6)
    for j in range(len(nomes), len(eixos)):
        eixos[j].axis("off")
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def _figura_boxplot(df_metricas: pd.DataFrame, caminho: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    modelos = df_metricas.index.tolist()
    medias = df_metricas["acuracia_media"]
    desvios = df_metricas["acuracia_desvio"]
    ax.bar(modelos, medias, yerr=desvios, capsize=3)
    ax.set_ylabel("Acurácia (média ± desvio entre repetições)")
    ax.set_xticklabels(modelos, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


# ============================================================================
# ABLAÇÕES
# ============================================================================


def ablacao_zeros(X: pd.DataFrame, y: pd.Series, grupo: pd.Series) -> pd.DataFrame:
    """Compara as 3 estratégias de TrataZeros com o mesmo classificador (LR)."""
    linhas = []
    for estrategia in ["log1p", "lod_metade", "original"]:
        pre = Pipeline([("zeros", TrataZeros(estrategia=estrategia)), ("escala", StandardScaler())])
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMENTE))])
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEMENTE)
        pred = cross_val_predict(pipe, X, y, groups=grupo, cv=cv)
        m = metricas_uma_repeticao(y.to_numpy(), pred)
        m["estrategia_zeros"] = estrategia
        linhas.append(m)
    df = pd.DataFrame(linhas).set_index("estrategia_zeros")
    print("\n=== Ablação: tratamento do piso de 0,01 ppm (classificador LR) ===")
    print(df[["acuracia", "acuracia_balanceada", "f1_macro"]].round(4))
    return df


def ablacao_outliers_kmeans(
    X: pd.DataFrame, y: pd.Series, grupo: pd.Series, quantil_outlier: float = 0.90
) -> pd.DataFrame:
    """Reproduz de forma aproximada a remoção de outliers por K-means entre as
    amostras normais (crítica 4) e mede o impacto na acurácia de um modelo
    representativo (LR), com e sem essa remoção.

    Não temos a receita exata do K-means original (nº de clusters, critério).
    Usamos K-means(k=4) sobre log1p dos 5 gases das amostras 'normal' e
    marcamos como outlier os `1-quantil_outlier` pontos mais distantes do seu
    próprio centroide (distância euclidiana) — um critério padrão de detecção
    de outlier por K-means. Testamos "cluster minoritário do k=2" antes: gerou
    43,5% da classe normal marcada como outlier, um número implausível para
    ser chamado de "outlier" (é só bisseção da classe, não detecção de
    anomalia) — por isso trocamos para o critério de distância ao centroide.
    """
    from sklearn.cluster import KMeans

    normal_mask = (y == "normal").to_numpy()
    X_log_normal = np.log1p(X.loc[normal_mask, GASES].to_numpy())
    km = KMeans(n_clusters=4, n_init=10, random_state=SEMENTE).fit(X_log_normal)
    centroides = km.cluster_centers_[km.labels_]
    distancias = np.linalg.norm(X_log_normal - centroides, axis=1)
    limiar = np.quantile(distancias, quantil_outlier)
    outlier_local = distancias > limiar

    outlier_mask = np.zeros(len(X), dtype=bool)
    outlier_mask[np.where(normal_mask)[0][outlier_local]] = True
    print(f"\n=== Ablação: outliers K-means entre amostras normais (quantil={quantil_outlier}) ===")
    print(f"{outlier_mask.sum()} de {normal_mask.sum()} amostras normais marcadas como outlier "
          f"({outlier_mask.sum() / normal_mask.sum():.1%}) "
          f"— dissertação original removeu 181/1295 (14,0%)")

    linhas = []
    for descricao, manter in [("com_outliers", np.ones(len(X), dtype=bool)), ("sem_outliers", ~outlier_mask)]:
        Xs, ys, gs = X[manter], y[manter], grupo[manter]
        pipe = Pipeline([("pre", PRE_BASICO), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMENTE))])
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEMENTE)
        pred = cross_val_predict(pipe, Xs, ys, groups=gs, cv=cv)
        m = metricas_uma_repeticao(ys.to_numpy(), pred)
        m["versao"] = descricao
        m["n_amostras"] = int(manter.sum())
        linhas.append(m)
    df = pd.DataFrame(linhas).set_index("versao")
    print(df[["n_amostras", "acuracia", "acuracia_balanceada", "f1_macro"]].round(4))
    return df


def ablacao_representacao(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """PRE_BASICO vs PRE_ESTENDIDO para o mesmo classificador — reaproveita os
    resultados já calculados no laço principal (KNN/KNN_estend, SVM_5gases/SVM)."""
    linhas = []
    for basico, estendido in PARES_REPRESENTACAO:
        for nome in (basico, estendido):
            linha = df_metricas.loc[nome, ["acuracia_media", "acuracia_balanceada_media", "f1_macro_media"]].to_dict()
            linha["modelo"] = nome
            linha["par"] = f"{basico} vs {estendido}"
            linhas.append(linha)
    df = pd.DataFrame(linhas).set_index("modelo")
    print("\n=== Ablação: representação básica (5 gases) vs estendida (+razões +percentuais) ===")
    print(df.round(4))
    return df


# ============================================================================
# 9. EXECUÇÃO
# ============================================================================


def executar_benchmark_completo() -> None:
    print("=== Carregando dados ===")
    X, y, grupo = carregar_dados()
    print(f"{len(X)} amostras | classes: {y.value_counts().to_dict()}")

    print("\n=== Avaliando todos os modelos (CV externa: "
          f"{N_SPLITS} folds x {N_REPEATS} repetições, agrupada por composição de gases) ===")
    predicoes_por_repeticao, predicoes_repeticao_0 = avaliar_todos(MODELOS, X, y, grupo, espacos=ESPACOS)

    print("\n=== Calculando métricas agregadas ===")
    df_metricas = tabela_metricas(predicoes_por_repeticao, y)
    print(df_metricas[["acuracia_media", "acuracia_desvio", "acuracia_balanceada_media", "f1_macro_media"]].round(4))

    print("\n=== Testes estatísticos pareados (repetição 0) ===")
    df_mcnemar, kappa_matriz = testes_pareados(predicoes_repeticao_0, y)
    print(df_mcnemar.sort_values("p_valor_holm").to_string(index=False))

    print("\n=== Exportando tabelas e figuras ===")
    exportar_tudo(df_metricas, df_mcnemar, kappa_matriz, predicoes_repeticao_0, y)
    print(f"Saída em: {SAIDA}")

    ablacao_representacao(df_metricas)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablacao", choices=["zeros", "outliers", "representacao"], default=None)
    args = parser.parse_args()

    print("=== Teste com casos canônicos (Tabela-2 da dissertação) ===\n")
    testar_casos_canonicos()
    print()

    if args.ablacao == "zeros":
        X, y, grupo = carregar_dados()
        ablacao_zeros(X, y, grupo)
    elif args.ablacao == "outliers":
        X, y, grupo = carregar_dados()
        ablacao_outliers_kmeans(X, y, grupo)
    elif args.ablacao == "representacao":
        X, y, grupo = carregar_dados()
        _, predicoes_repeticao_0 = avaliar_todos(MODELOS, X, y, grupo, espacos=ESPACOS, n_repeats=1)
        df_metricas = tabela_metricas({k: [v] for k, v in predicoes_repeticao_0.items()}, y)
        ablacao_representacao(df_metricas)
    else:
        executar_benchmark_completo()
