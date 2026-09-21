"""Método de Doernenburg (Tabelas 5 e 6 da dissertação).

O dataset não possui a coluna CO, portanto o monóxido de carbono não é usado
nas pré-condições nem nas razões, exatamente como na dissertação: "o CO ...
não foi levado em consideração no diagnóstico".

R1 = CH4/H2, R2 = C2H2/C2H4, R3 = C2H2/CH4, R4 = C2H6/C2H2
"""

import pandas as pd

from dados import carregar_dataset

LIMITES_L1 = {"H2": 100, "CH4": 120, "C2H4": 50, "C2H6": 65, "C2H2": 1}

FAIXA_PARA_CLASSE3 = {
    "Decomposicao termica": "Termica",
    "Descarga de baixa energia": "Eletrica",
    "Descarga de alta energia": "Eletrica",
}


def _razao(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return num / den


def diagnostico(h2: float, ch4: float, c2h4: float, c2h6: float, c2h2: float) -> str:
    gases = {"H2": h2, "CH4": ch4, "C2H4": c2h4, "C2H6": c2h6, "C2H2": c2h2}

    significativo = any(gases[g] > 2 * LIMITES_L1[g] for g in LIMITES_L1)
    if not significativo:
        return "Normal"

    r1 = _razao(ch4, h2)
    r2 = _razao(c2h2, c2h4)
    r3 = _razao(c2h2, ch4)
    r4 = _razao(c2h6, c2h2)

    if r1 > 1.0 and r2 < 0.75 and r3 < 0.3 and r4 > 0.4:
        return "Decomposicao termica"
    if r1 < 0.1 and r3 < 0.3 and r4 > 0.4:
        # R2 é "NS" (não significante) nesta linha da Tabela-6.
        return "Descarga de baixa energia"
    if 0.1 <= r1 <= 1.0 and r2 > 0.75 and r3 > 0.3 and r4 < 0.4:
        return "Descarga de alta energia"
    return "Indeterminado"


def classificar(df: pd.DataFrame) -> pd.Series:
    diag = df.apply(
        lambda linha: diagnostico(
            linha["H2"], linha["CH4"], linha["C2H4"], linha["C2H6"], linha["C2H2"]
        ),
        axis=1,
    )
    return diag.map(FAIXA_PARA_CLASSE3).fillna("Indeterminado")


if __name__ == "__main__":
    from avaliacao import Resultado, imprimir_resultado

    df = carregar_dataset()
    pred = classificar(df)
    imprimir_resultado(Resultado("Doernenburg", df["classe3"].tolist(), pred.tolist()))
