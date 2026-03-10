import sys
import os

#ajoute le dossier parent (rag/) au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag_pipeline import load_pipeline

def get_answer_minimal(question):
    model, collection = load_pipeline()

    #ajouter un document test
    collection.add(
        documents=[
            "Inscription en licence informatique à l'université Le Havre",
            "Informations sur le logement étudiant à Le Havre"
        ],
        metadatas=[
            {"title": "Licence Info", "category": "formation", "source_url": "https://www.univ-lehavre.fr"},
            {"title": "Logement", "category": "vie_etudiante", "source_url": "https://www.univ-lehavre.fr/logement"}    
        ],
        
        ids=["doc1","doc2"]
    )

    #tester la recherche
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print("Resultats :", results)

if __name__ == "__main__":
    get_answer_minimal("Formations en informatique ?")