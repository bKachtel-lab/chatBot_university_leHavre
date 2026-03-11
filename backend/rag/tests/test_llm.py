import sys
import os

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from llm_client import ask_llm, test_connection


def test_llm():

    context = """
    [Source 1] Licence Informatique
    Inscription en licence informatique à l'université Le Havre.
    """

    question = "Comment s'inscrire en licence informatique ?"

    answer = ask_llm(context, question)

    print("\nRéponse du LLM :\n")
    print(answer)


if __name__ == "__main__":

    print("Test connexion LLM...")
    test_connection()

    print("\nTest génération réponse...")
    test_llm()