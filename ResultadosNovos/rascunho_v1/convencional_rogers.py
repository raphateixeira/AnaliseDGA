"""Relação de Rogers (Tabela-3 da dissertação, adaptada da IEEE C57.104:2019).

R1 = C2H2/C2H4, R2 = CH4/H2, R3 = C2H4/C2H6
"""

import pandas as pd

from dados import carregar_dataset

FAIXA_PARA_CLASSE3 = {
    "Normal": "Normal",
    "Descarga de baixa energia - DP": "Eletrica",
    "Descarga de alta energia - Arco": "Eletrica",
    "Falta termica de baixa temperatura": "Termica",
    "Falta termica T<700C": "Termica",
    "Falta termica T>700C": "Termica",
}


def _razao(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return num / den


def diagnostico(h2: float, ch4: float, c2h4: float, c2h6: float, c2h2: float) -> str:
    r1 = _razao(c2h2, c2h4)
    r2 = _razao(ch4, h2)
    r3 = _razao(c2h4, c2h6)

    if r1 < 0.1 and 0.1 <= r2 <= 1.0 and r3 < 0.1:
        return "Normal"
    if r1 < 0.1 and r2 < 0.1 and r3 < 0.1:
        return "Descarga de baixa energia - DP"
    if 0.1 <= r1 <= 3.0 and 0.1 <= r2 <= 1.0 and r3 > 3.0:
        return "Descarga de alta energia - Arco"
    if r1 < 0.1 and 0.1 <= r2 <= 1.0 and 0.1 <= r3 <= 3.0:
        return "Falta termica de baixa temperatura"
    if r1 < 0.1 and r2 > 1.0 and 0.1 <= r3 <= 3.0:
        return "Falta termica T<700C"
    if r1 < 0.1 and r2 > 1.0 and r3 > 3.0:
        return "Falta termica T>700C"
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
    imprimir_resultado(Resultado("Rogers", df["classe3"].tolist(), pred.tolist()))
