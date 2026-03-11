"""
context_builder.py

Transforme les documents récupérés par le retriever
en un contexte structuré utilisable par le LLM.
"""
def build_context(documents, max_docs=5, min_score=0.0):
    """
    Construit un contexte à partir des documents récupérés.

    Parametres
    ----
    documents : list of dict
        Liste des documents récupérés par le retriever 
        Chaque doc doit contenir :
        - text 
        - title
        - category
        - url
        - distance (score de similarité)
    
    max_docs : int 
        Nombre maximal de documents à inclure

    min_score : float
        Score minimal de documents à inclure

    Returns
    -----
    str
        Texte combiné à fournir au LLM
    """

    #Filtrer les documents avec un score suffisant
    filtered_docs = [doc for doc in documents if doc["distance"] >= min_score]

    #Trier par distance décroissante (plus pertinent en premier)
    sorted_docs = sorted(filtered_docs, key=lambda x: x["distance"], reverse=True)

    #Limiter au nombre max de documents
    top_docs = sorted_docs[:max_docs]

    #Construire le contexte sous forme de texte structuré
    context_parts = []
    for i, doc in enumerate(top_docs, start=1):
        part = (
            f"[Source {i}] {doc['title']} (Catégorie: {doc['category']})\n"
            f"URL: {doc['url']}\n"
            f"{doc['text']}"
        )
        context_parts.append(part)
    
    #Joindre tous les morceaux avec des séparateurs
    context = "\n---\n".join(context_parts)

    return context