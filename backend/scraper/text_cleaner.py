"""
CampusGPT — Nettoyage et extraction du texte HTML
Transforme le HTML brut en texte propre, prêt pour les embeddings.
"""

import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from config import ALLOWED_DOMAINS, URL_EXCLUDE_PATTERNS, CONTENT_CATEGORIES


def clean_html(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # 1. Nettoyage des balises INUTILES uniquement
    # On retire "footer" de cette liste pour ne JAMAIS le supprimer
    tags_to_remove = [
        "script", "style", "nav", "header", "aside",
        "iframe", "noscript", "form", "button", "input", "select"
    ]

    for tag in soup(tags_to_remove):
        tag.decompose()

    # 2. Extraire le titre
    title = soup.title.get_text(strip=True) if soup.title else ""
    if not title and soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    title = title.split("|")[0].split("–")[0].strip()

    # 3. Définir le conteneur de recherche
    # On utilise 'body' pour être certain d'englober le footer 
    # car il est souvent à l'extérieur de la balise <main>
    container = soup.find("body") or soup

    # 4. Extraire le texte structuré
    text_parts = []
    # On s'assure que "footer" est dans les balises à extraire
    tags_to_extract = ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "footer", "address"]

    for element in container.find_all(tags_to_extract):
        # Séparateur visuel pour aider le modèle à identifier le bloc de contact
        if element.name in ["footer", "address"]:
            text_parts.append("\n--- INFORMATIONS DE CONTACT ET LOCALISATION ---")

        text = element.get_text(separator=" ", strip=True)
        
        # Sécurité : on ignore les textes trop courts
        if not text or len(text) < 3:
            continue

        # Formatage selon la balise
        if element.name in ("h1", "h2"):
            text_parts.append(f"\n## {text}\n")
        elif element.name in ("h3", "h4"):
            text_parts.append(f"\n### {text}\n")
        elif element.name == "li":
            text_parts.append(f"• {text}")
        else:
            text_parts.append(text)

    content = "\n".join(text_parts)
    content = normalize_whitespace(content)

    # 5. Extraction des liens 
    # ... 

    return {
        "title": title,
        "content": content,
        "links": list(set(links)),
        "pdf_links": list(set(pdf_links)),
    }

def normalize_whitespace(text: str) -> str:
    """Supprime les espaces/lignes multiples inutiles."""
    # Supprimer les caractères spéciaux parasites
    text = text.replace("\xa0", " ")        # Non-breaking space
    text = text.replace("\u200b", "")       # Zero-width space
    text = text.replace("\t", " ")

    # Réduire les espaces multiples sur une même ligne
    text = re.sub(r"[ ]{2,}", " ", text)

    # Réduire les sauts de ligne multiples (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_internal_url(url: str) -> bool:
    """Vérifie si une URL appartient aux domaines autorisés."""
    try:
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in ALLOWED_DOMAINS)
    except Exception:
        return False


def is_valid_url(url: str) -> bool:
    """Filtre les URLs à exclure (images, login, etc.)."""
    url_lower = url.lower()
    return not any(pattern in url_lower for pattern in URL_EXCLUDE_PATTERNS)


def detect_category(url: str) -> str:
    """Détermine la catégorie d'une page à partir de son URL."""
    url_lower = url.lower()
    for pattern, category in CONTENT_CATEGORIES.items():
        if pattern in url_lower:
            return category
    return "general"


# ── EXTRACTION PDF ───────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extrait le texte d'un fichier PDF local.
    Utilise pdfplumber (meilleur que PyPDF2 pour les PDFs complexes).
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"\n--- Page {page_num} ---\n{text}")

        full_text = "\n".join(text_parts)
        return normalize_whitespace(full_text)

    except ImportError:
        # Fallback avec PyPDF2
        try:
            import PyPDF2
            text_parts = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return normalize_whitespace("\n".join(text_parts))
        except Exception as e:
            return f"[Erreur extraction PDF: {e}]"

    except Exception as e:
        return f"[Erreur extraction PDF: {e}]"


# ── CHUNKING POUR RAG ─────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Stratégie intelligente :
      1. Découpe d'abord par paragraphes
      2. Si un paragraphe est trop long, découpe par phrases
      3. Si encore trop long, découpe par caractères

    Retourne une liste de chunks propres.
    """
    if not text or len(text.strip()) < 50:
        return []

    # Séparer par paragraphes
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_words = para.split()
        para_size = len(para_words)

        # Le paragraphe dépasse la taille max → découper par phrases
        if para_size > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sent_words = sentence.split()
                sent_size = len(sent_words)

                if current_size + sent_size > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Chevauchement : garder les derniers mots
                    overlap_words = current_chunk[-overlap:] if overlap else []
                    current_chunk = overlap_words + sent_words
                    current_size = len(current_chunk)
                else:
                    current_chunk.extend(sent_words)
                    current_size += sent_size

        # Le paragraphe rentre dans le chunk courant
        elif current_size + para_size <= chunk_size:
            current_chunk.extend(para_words)
            current_size += para_size

        # Le paragraphe ne rentre pas → nouveau chunk
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_words = current_chunk[-overlap:] if overlap else []
                current_chunk = overlap_words + para_words
                current_size = len(current_chunk)
            else:
                current_chunk = para_words
                current_size = para_size

    # Ne pas oublier le dernier chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Filtrer les chunks trop courts (< 30 mots = peu utile pour RAG)
    return [c for c in chunks if len(c.split()) >= 30]