"""
retriever.py

Ce module gère la récupération des documents pertinents
dans la base vectorielle ChromaDB à partir d'une question utilisateur.
"""

#Import du pipeline RAG (modèle + collection chromaDB)
from rag_pipeline import load_pipeline

def retrieve_documents(question, top_k=3):
    """
    Fonction principale de recherche .

    Parameters
     ----------
    question : str
        Question posée par l'utilisateur

    top_k : int
        Nombre de documents les plus proches à récupérer

    Returns
    -------
    list
        Liste des documents pertinents avec leurs métadonnées
    """ 

    #Charger le modèle d'embedding et la collection ChromaDB
    model, collection = load_pipeline()

    #Transformer la question en vecteur numérique
    query_embedding = model.encode(question).tolist()

    #Recherche dans la base vectorielle
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k,
        include = ["documents", "metadatas", "distances"]
    )

    #Liste pour stocker les resultats formatés
    documents = []

    #parcours des résultats retournés
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        documents.append({
            "text" : doc,
            "title" : meta.get("title"),
            "category": meta.get("category"),
            "url": meta.get("source_url"),
            "distance" : dist
        })

    return documents