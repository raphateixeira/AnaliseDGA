"""Métricas de avaliação no mesmo padrão da dissertação: matriz de confusão,
precisão por classe, acurácia e sensibilidade (recall) geral."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from dados import ORDEM_3_CLASSES


@dataclass
class Resultado:
    metodo: str
    y_true: list
    y_pred: list
    indeterminado_label: str = "Indeterminado"

    def matriz_confusao(self) -> pd.DataFrame:
        rotulos = ORDEM_3_CLASSES + (
            [self.indeterminado_label] if self.indeterminado_label in self.y_pred else []
        )
        m = confusion_matrix(self.y_true, self.y_pred, labels=rotulos)
        return pd.DataFrame(m, index=[f"Real={r}" for r in rotulos], columns=rotulos)

    def resumo(self) -> dict:
        certos = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)
        determinados = [
            (t, p) for t, p in zip(self.y_true, self.y_pred) if p != self.indeterminado_label
        ]
        n = len(self.y_true)
        acuracia = certos / n if n else float("nan")
        precisoes = {}
        for classe in ORDEM_3_CLASSES:
            preditos_classe = [p == classe for _, p in determinados]
            corretos_classe = [t == classe and p == classe for t, p in determinados]
            total_preditos = sum(preditos_classe)
            precisoes[classe] = (
                sum(corretos_classe) / total_preditos if total_preditos else float("nan")
            )
        acertos_reais = sum(1 for t, p in determinados if t == p)
        sensibilidade = acertos_reais / len(determinados) if determinados else float("nan")
        return {
            "metodo": self.metodo,
            "acuracia": acuracia,
            "sensibilidade": sensibilidade,
            "indeterminados": n - len(determinados),
            **{f"precisao_{c}": v for c, v in precisoes.items()},
        }


def kappa_cohen(y_pred_a: list, y_pred_b: list) -> float:
    return cohen_kappa_score(y_pred_a, y_pred_b, weights="linear")


def imprimir_resultado(res: Resultado) -> None:
    print(f"\n=== {res.metodo} ===")
    print(res.matriz_confusao())
    resumo = res.resumo()
    print(
        f"Acurácia: {resumo['acuracia']:.2%}  "
        f"Sensibilidade: {resumo['sensibilidade']:.2%}  "
        f"Indeterminados: {resumo['indeterminados']}"
    )
    for classe in ORDEM_3_CLASSES:
        print(f"  Precisão {classe}: {resumo[f'precisao_{classe}']:.2%}")
