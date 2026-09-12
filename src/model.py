"""Modele de regression penalisee : cout, gradient et estimation des poids.

La fonction de cout est J(beta) = (1/n) ||X beta moins y||^2 + lambda ||beta||^2.
Son gradient, demontre a la main dans le rapport, vaut
    (2/n) X^T (X beta moins y) + 2 lambda beta.
"""

import numpy as np


def cout(X, y, beta, lam):
    """Valeur de la fonction de cout."""
    n = X.shape[0]
    ecarts = X @ beta - y
    return (1 / n) * np.sum(ecarts ** 2) + lam * np.sum(beta ** 2)


def gradient(X, y, beta, lam):
    """Gradient de la fonction de cout."""
    n = X.shape[0]
    ecarts = X @ beta - y
    return (2 / n) * X.T @ ecarts + 2 * lam * beta


def equations_normales(X, y, lam):
    """Solution directe des poids (formule fermee)."""
    n, p = X.shape
    A = X.T @ X + n * lam * np.eye(p)
    return np.linalg.solve(A, X.T @ y)


def descente_gradient(X, y, lam, alpha, max_iter=5000, tol=1e-8):
    """Estimation des poids par descente de gradient.

    Retourne les poids et l'historique du cout a chaque iteration.
    """
    p = X.shape[1]
    beta = np.zeros(p)
    historique = [cout(X, y, beta, lam)]
    for _ in range(max_iter):
        g = gradient(X, y, beta, lam)
        if np.linalg.norm(g) < tol:
            break
        beta = beta - alpha * g
        historique.append(cout(X, y, beta, lam))
    return beta, np.array(historique)


def intervalles_confiance(X, y):
    """Poids sans penalite, erreurs standard et intervalles a 95 pour cent."""
    n, p = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    residus = y - X @ beta
    variance = np.sum(residus ** 2) / (n - p)
    covariance = variance * np.linalg.inv(X.T @ X)
    erreurs = np.sqrt(np.diag(covariance))
    bas = beta - 1.96 * erreurs
    haut = beta + 1.96 * erreurs
    return beta, erreurs, bas, haut
