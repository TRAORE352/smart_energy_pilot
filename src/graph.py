"""Graphe des etats et recommandation.

Chaque etat est une heure avec un mode (normal ou economie). Sa consommation
est predite par le modele. On avance seulement dans le temps, donc le graphe
est oriente et sans cycle. Le plan optimal est le chemin de plus faible
consommation cumulee, trouve par programmation dynamique.
"""

import numpy as np

HEURES = [8, 10, 12, 14, 16, 18]


def predire(h, t, occ, beta, moyennes, ecarts, consommation_moyenne):
    """Consommation prevue par le modele pour un etat, en kWh."""
    x = (np.array([h, t, occ]) - moyennes) / ecarts
    return float(consommation_moyenne + x @ beta)


def etats_de_l_heure(h):
    """Etats possibles a une heure donnee.

    Le mode economie est interdit a 12h et 14h pour le confort.
    """
    temperature_prevue = 22 + 0.9 * (h - 8)
    etats = [("normal", temperature_prevue, 35)]
    if h not in (12, 14):
        etats.append(("economie", temperature_prevue - 2, 20))
    return etats


def plan_optimal(beta, moyennes, ecarts, consommation_moyenne):
    """Retourne le plan recommande et les consommations cumulees.

    Comme les choix sont independants par heure, le plus court chemin dans le
    graphe par couches revient a prendre le meilleur mode a chaque heure.
    """
    plan = []
    total_optimal = 0.0
    total_normal = 0.0
    for h in HEURES:
        options = etats_de_l_heure(h)
        meilleur = min(
            options,
            key=lambda s: predire(h, s[1], s[2], beta, moyennes, ecarts, consommation_moyenne),
        )
        conso = predire(h, meilleur[1], meilleur[2], beta, moyennes, ecarts, consommation_moyenne)
        conso_normale = predire(h, options[0][1], options[0][2], beta, moyennes, ecarts, consommation_moyenne)
        total_optimal += conso
        total_normal += conso_normale
        plan.append((h, meilleur[0], round(conso, 1)))
    return {
        "plan": plan,
        "total_optimal": round(total_optimal, 1),
        "total_normal": round(total_normal, 1),
        "economie": round(total_normal - total_optimal, 1),
    }
