"""Triângulo de Duval 1 (Figura-8 e Equações 2.1-2.3 da dissertação).

%CH4 = CH4/(CH4+C2H4+C2H2)*100
%C2H4 = C2H4/(CH4+C2H4+C2H2)*100
%C2H2 = C2H2/(CH4+C2H4+C2H2)*100

Vértices das 7 regiões (em %C2H4, %CH4 — o %C2H2 é o complemento a 100)
conferidos visualmente contra a Figura-8 do PDF da dissertação: topo =
100% CH4, vértice inferior-esquerdo = 100% C2H2, vértice inferior-direito =
100% C2H4.

A região DT (falha mista térmica+elétrica) não corresponde a nenhuma das 3
classes do dataset (Normal/Térmica/Elétrica) e é reportada como
"Indeterminado" na avaliação — o método não possui diagnóstico Normal
(nenhuma região cobre valores baixos/normais de gás, conforme observado na
dissertação: "o triângulo de Duval sempre aponta algum tipo de falha, mesmo
para amostras normais").
"""

import pandas as pd

from dados import carregar_dataset

REGIOES = {
    "PD": [(0, 98), (0, 100), (2, 98)],
    "T1": [(0, 96), (0, 98), (2, 98), (20, 80), (20, 76)],
    "T2": [(20, 76), (20, 80), (50, 50), (50, 46)],
    "T3": [(50, 35), (50, 50), (100, 0), (85, 0)],
    "D1": [(0, 0), (0, 87), (23, 64), (23, 0)],
    "D2": [(23, 0), (23, 64), (40, 47), (40, 31), (71, 0)],
    "DT": [(0, 87), (0, 96), (50, 46), (50, 35), (85, 0), (71, 0), (40, 31), (40, 47)],
}

REGIAO_PARA_CLASSE3 = {
    "PD": "Eletrica",
    "D1": "Eletrica",
    "D2": "Eletrica",
    "T1": "Termica",
    "T2": "Termica",
    "T3": "Termica",
    "DT": "Indeterminado",
}


def _ponto_no_poligono(x: float, y: float, poligono: list) -> bool:
    dentro = False
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            dentro = not dentro
    return dentro


def diagnostico(ch4: float, c2h4: float, c2h2: float) -> str:
    total = ch4 + c2h4 + c2h2
    if total == 0:
        return "Indeterminado"
    pct_ch4 = ch4 / total * 100
    pct_c2h4 = c2h4 / total * 100
    pct_c2h2 = c2h2 / total * 100

    for regiao, poligono in REGIOES.items():
        if _ponto_no_poligono(pct_c2h4, pct_ch4, poligono):
            return regiao
    # Pontos exatamente sobre uma aresta (ex.: nos vértices do triângulo).
    if pct_c2h2 <= 0 and pct_c2h4 <= 0:
        return "T1"
    return "Indeterminado"


def classificar(df: pd.DataFrame) -> pd.Series:
    regiao = df.apply(lambda linha: diagnostico(linha["CH4"], linha["C2H4"], linha["C2H2"]), axis=1)
    return regiao.map(REGIAO_PARA_CLASSE3).fillna("Indeterminado")


if __name__ == "__main__":
    from avaliacao import Resultado, imprimir_resultado

    df = carregar_dataset()
    pred = classificar(df)
    imprimir_resultado(Resultado("Triangulo de Duval 1", df["classe3"].tolist(), pred.tolist()))
