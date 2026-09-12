"""Script principal.

Execute la chaine complete : donnees, estimation, conditionnement, graphe,
puis affiche les resultats. Lancer avec : python src/main.py
"""

import numpy as np

from data import generer_donnees
from model import (
    equations_normales,
    descente_gradient,
    intervalles_confiance,
)
from conditioning import valeurs_propres, conditionnement
from graph import plan_optimal


def main():
    d = generer_donnees()
    X, y = d["X"], d["y"]
    lam = 0.1
    alpha = 0.1

    # Estimation par les deux methodes.
    beta_exact = equations_normales(X, y, lam)
    beta_dg, historique = descente_gradient(X, y, lam, alpha)
    ecart = np.linalg.norm(beta_exact - beta_dg)

    print("Poids (equations normales) :", beta_exact.round(3))
    print("Poids (descente de gradient) :", beta_dg.round(3))
    print("Ecart entre les deux methodes :", ecart)

    # Conditionnement.
    vp = valeurs_propres(X)
    print("\nConditionnement sans penalite :", round(conditionnement(vp, 0.0), 2))
    print("Conditionnement avec penalite  :", round(conditionnement(vp, lam), 2))

    # Intervalles de confiance.
    beta_ols, se, bas, haut = intervalles_confiance(X, y)
    noms = ["heure", "temperature", "occupants"]
    print("\nIntervalles de confiance a 95 pour cent :")
    for j, nom in enumerate(noms):
        print(f"  {nom:12s} : {beta_ols[j]:.3f}  [{bas[j]:.3f} ; {haut[j]:.3f}]")

    # Recommandation.
    resultat = plan_optimal(
        beta_exact, d["moyennes"], d["ecarts"], d["consommation_moyenne"]
    )
    print("\nPlan recommande :")
    for h, mode, conso in resultat["plan"]:
        print(f"  {h:2d}h : mode {mode:9s} ({conso} kWh)")
    print("Consommation optimale :", resultat["total_optimal"], "kWh")
    print("Consommation de reference :", resultat["total_norm"] if False else resultat["total_normal"], "kWh")
    print("Economie :", resultat["economie"], "kWh")


if __name__ == "__main__":
    main()
