"""Etude du conditionnement et de l'effet de la penalite.

Le conditionnement mesure a quel point la surface d'erreur est allongee.
La penalite lambda releve chaque valeur propre, ce qui le rapproche de un
et stabilise l'apprentissage.
"""

import numpy as np


def valeurs_propres(X):
    """Valeurs propres de (1/n) X^T X, triees de la plus grande a la plus petite."""
    n = X.shape[0]
    A = (1 / n) * X.T @ X
    return np.sort(np.linalg.eigvalsh(A))[::-1]


def conditionnement(vp, lam=0.0):
    """Conditionnement kappa pour une penalite donnee."""
    return (vp[0] + lam) / (vp[-1] + lam)


def courbe_conditionnement(vp, lambdas):
    """Conditionnement en fonction de la penalite, pour tracer la courbe."""
    return [(float(l), float(conditionnement(vp, l))) for l in lambdas]
