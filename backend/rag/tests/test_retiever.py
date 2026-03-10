"""
Test du module retriever
"""
import sys
import os

#Permet d'importer les modules du dossier parent 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retriever import retrieve_documents

#question test
question = "formation informatique"

#recherche
results = retrieve_documents(question)

#Affichage
print("\nDocuments trouvés :\n")

for doc in results:
    print("Titre :", doc["title"])
    print("Texte :", doc["text"])
    print("URL :", doc["url"])
    print("Distances :", doc["distance"])
    print("-" * 40)
    