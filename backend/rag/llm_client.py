"""
llm_client.py

Ce module envoie la question + contexte au LLM
et récupère la réponse générée.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

#configuration du serveur localAI
LOCALAI_URL     = os.getenv("LOCALAI_URL",    "http://localhost:8080/v1")
LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY", "not-needed")
LOCALAI_MODEL   = os.getenv("LOCALAI_MODEL",   "mistral")

#charger dotenv
load_dotenv()

#parametres de generation 
MAX_TOKENS = 1024
TEMPERATURE = 0.2

def get_client() -> OpenAI:
    """ Retourne un client OpenAI pointe vers LocalAI."""
    return OpenAI(
        base_url=LOCALAI_URL,
        api_key=LOCALAI_API_KEY
    )

def ask_llm(system_prompt: str, user_message: str, history: list = None) -> str:
    """
    Envoie un message au LLM et retourne la reponse
    Args: 
        system_prompt : instructions systeme (contexte RAG injecte ici)
        user_message  : question de l'etudiant
        history       : historique de conversation [{"role":..., "content":...}]

    Returns:
        Texte de la reponse du LLM
    """
    client = get_client()

    #Liste des messages envoyés au modèle
    messages = [{"role": "system", "content": system_prompt}]

    #Ajouter l'historique de conversation si present
    if history:
        messages.extend(history)
    
    #Ajouter la question actuelle
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model = LOCALAI_MODEL,
            messages=messages,
            max_tokens = MAX_TOKENS,
            temperature = TEMPERATURE,
            stream=False,
        )
        #Retourne la réponse générée
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"[Erreur LLM] Impossible de contacter le serveur LocalAI : {e}\n" \
               f"Verifie que LocalAI tourne sur {LOCALAI_URL}"


def ask_llm_stream(system_prompt: str, user_message: str,
                   history: list = None):
    """
    Version streaming : retourne un generateur de tokens.
    Permet d'afficher la reponse mot par mot dans Streamlit.
    """
    client = get_client()

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=LOCALAI_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        #parcourir les tokens générés
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        yield f"[Erreur LLM] {e}"


def test_connection() -> bool:
    """Verifie que LocalAI repond correctement."""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=LOCALAI_MODEL,
            messages=[{"role": "user", "content": "Reponds juste 'OK'"}],
            max_tokens=10,
        )
        print(f"Connexion LocalAI OK - Modele : {LOCALAI_MODEL}")
        print(f"Reponse test : {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Echec connexion LocalAI ({LOCALAI_URL}) : {e}")
        return False