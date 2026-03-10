import os
#definir le dossier cache
os.environ["TRANSFORMERS_CACHE"] = r"C:\Users\Kachtel\hf_cache"
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_PATH #créer ce fichier avec le chemin vers la DB

def load_pipeline():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   
    client = chromadb.PersistentClient(
        path = VECTOR_DB_PATH,
        settings = Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("campusgpt")
    return model, collection

if __name__ == "__main__":
    model, collection = load_pipeline()
    print("Pipeline RAG chargé !")