from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# On importe  moteur RAG !
from rag.pipeline import answer

app = FastAPI(
    title="CampusGPT API",
    description="API de recherche sémantique pour l'Université Le Havre Normandie",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En prod on mettra l'URL précise, là on autorise tout pour le dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Modèle de données pour la question (Validation Pro)
class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []
    category: Optional[str] = None

@app.get("/")
def home():
    return {"status": "online", "message": "CampusGPT API is running"}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        # On appelle ta fonction answer avec top_k=10 par défaut
        result = answer(
            question=req.question,
            history=req.history,
            top_k=10,
            category_filter=req.category
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)