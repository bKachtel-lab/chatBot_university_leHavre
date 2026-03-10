import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from context_builder import build_context

# Exemple de documents simulés
documents = [
    {"text": "Inscription en licence informatique à l'université Le Havre",
     "title": "Licence Info",
     "category": "formation",
     "url": "https://www.univ-lehavre.fr",
     "distance": 1.35},
    {"text": "Informations sur le logement étudiant à Le Havre",
     "title": "Logement",
     "category": "vie_etudiante",
     "url": "https://www.univ-lehavre.fr/logement",
     "distance": 1.59},
]

context = build_context(documents)
print("Contexte généré :\n")
print(context)