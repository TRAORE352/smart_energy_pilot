"""Generation des donnees du projet.

Le sujet ne fournit aucune mesure reelle. Nous fabriquons donc des donnees
realistes : pour chaque heure, une temperature liee a l'heure, un nombre
d'occupants, et une consommation calculee avec des poids connus, plus un bruit.
Comme nous connaissons les vrais poids, nous pouvons verifier plus tard que le
modele les retrouve.
"""

import numpy as np

# Poids reels ayant servi a fabriquer la consommation (heure, temperature, occupants).
VRAIS_POIDS = np.array([4.0, 12.0, 8.0])
CONSOMMATION_BASE = 50.0


def generer_donnees(n=200, graine=42):
    """Retourne X standardisee, y centree, et les parametres de standardisation.

    Renvoie un dictionnaire contenant :
    X, y, moyennes, ecarts, consommation_moyenne, n.
    """
    np.random.seed(graine)

    heure = np.random.uniform(8, 18, n)
    temperature = 22 + 0.9 * (heure - 8) + np.random.normal(0, 1.5, n)
    occupants = np.random.uniform(0, 40, n)

    X_brut = np.column_stack([heure, temperature, occupants])

    # Standardisation : chaque colonne a moyenne 0 et ecart type 1.
    moyennes = X_brut.mean(axis=0)
    ecarts = X_brut.std(axis=0)
    X = (X_brut - moyennes) / ecarts

    # Consommation reelle en kWh, puis centrage pour l'entrainement.
    consommation_reelle = CONSOMMATION_BASE + X @ VRAIS_POIDS + np.random.normal(0, 2.0, n)
    consommation_moyenne = consommation_reelle.mean()
    y = consommation_reelle - consommation_moyenne

    return {
        "X": X,
        "y": y,
        "moyennes": moyennes,
        "ecarts": ecarts,
        "consommation_moyenne": consommation_moyenne,
        "n": n,
    }
