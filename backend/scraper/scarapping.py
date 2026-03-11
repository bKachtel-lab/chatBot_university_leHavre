"""
CampusGPT — Crawler asynchrone pour l'Universite Le Havre Normandie
"""



import asyncio  # Pour gérer les tâches en parallèle (asynchronisme)
import aiofiles # Pour écrire les PDF sur le disque sans bloquer le programme
import argparse
import aiohttp
import os
import time
from urllib.parse import urlparse

from config import (
    SEED_URLS, MAX_PAGES, MAX_DEPTH, DELAY_BETWEEN_REQUESTS,
    TIMEOUT, MAX_RETRIES, CONCURRENT_REQUESTS, HEADERS, PDF_DOWNLOAD_DIR
)
from database import (
    init_db, enqueue_url, get_next_urls, update_queue_status,
    save_page, save_pdf, log_error, get_stats
)
from text_cleaner import clean_html, detect_category, extract_pdf_text


def log(msg, level="info"):
    icons = {"info": "i", "ok": "OK", "warn": "!", "error": "X", "pdf": "PDF"}
    print(f"[{icons.get(level,'·')}] {msg}", flush=True)


urls_en_cours: set = set()
pages_scraped: int = 0
semaphore = None


async def fetch_url(session, url):
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.get(
                    url, headers=HEADERS,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT),
                    allow_redirects=True, ssl=False,
                ) as response:
                    status = response.status
                    ct = response.content_type or ""
                    if status == 200:
                        if "text/html" in ct:
                            return status, "html", await response.text(errors="replace")
                        elif "application/pdf" in ct:
                            return status, "pdf", await response.read()
                        else:
                            return status, "other", ""
                    else:
                        return status, "error", ""
        except asyncio.TimeoutError:
            log(f"Timeout ({attempt+1}/{MAX_RETRIES}) : {url}", "warn")
        except aiohttp.ClientError as e:
            log(f"Erreur reseau : {url} - {e}", "warn")
        except Exception as e:
            log(f"Erreur : {url} - {e}", "error")
            log_error(url, str(e))
            break
        await asyncio.sleep(2 ** attempt)
    return 0, "error", ""


async def download_pdf(session, url, category):
    os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)
    filename = urlparse(url).path.replace("/", "_").lstrip("_")
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    local_path = os.path.join(PDF_DOWNLOAD_DIR, filename)
    if os.path.exists(local_path):
        return
    status, ct, body = await fetch_url(session, url)
    if status == 200 and ct == "pdf":
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(body)
        text = extract_pdf_text(local_path)
        save_pdf(url=url, local_path=local_path,
                 title=filename.replace("_", " ").replace(".pdf", ""),
                 content=text, category=category)
        log(f"PDF sauvegarde : {filename}", "pdf")


async def process_page(session, url, depth, queue_id, max_pages):
    global pages_scraped, urls_en_cours

    if pages_scraped >= max_pages:
        update_queue_status(queue_id, "skip")
        return

    log(f"[{pages_scraped + 1}/{max_pages}] depth={depth} -> {url}")
    await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    status, ct, body = await fetch_url(session, url)

    if ct == "html" and status == 200:
        parsed   = clean_html(body, url)
        category = detect_category(url)

        if len(parsed["content"]) > 100:
            save_page(url=url, title=parsed["title"], content=parsed["content"],
                      category=category, depth=depth, http_status=status,
                      html_raw=body[:50000])
            pages_scraped += 1
            log(f"  -> '{parsed['title'][:60]}' | {len(parsed['content'])} chars "
                f"| {len(parsed['links'])} liens | cat:{category}", "ok")

            if depth < MAX_DEPTH:
                for link in parsed["links"]:
                    enqueue_url(link, depth + 1)  # SQLite ignore les doublons

            for pdf_url in parsed["pdf_links"]:
                enqueue_url(pdf_url, depth + 1)
                await download_pdf(session, pdf_url, category)
        else:
            log("  -> Page vide, ignoree", "warn")

        update_queue_status(queue_id, "done")

    elif ct == "pdf" and status == 200:
        await download_pdf(session, url, detect_category(url))
        update_queue_status(queue_id, "done")

    elif status in (403, 404, 410):
        log(f"  -> HTTP {status}", "warn")
        update_queue_status(queue_id, "error")

    else:
        log(f"  -> Statut {status} ignore", "warn")
        update_queue_status(queue_id, "skip")

    urls_en_cours.discard(url)


async def run_crawler(max_pages=MAX_PAGES):
    global semaphore, pages_scraped, urls_en_cours

    pages_scraped = 0
    urls_en_cours = set()
    semaphore     = asyncio.Semaphore(CONCURRENT_REQUESTS)

    init_db()
    for url in SEED_URLS:
        enqueue_url(url, depth=0)

    log(f"Demarrage du crawler CampusGPT - ULHN")
    log(f"   Max pages : {max_pages} | Max depth : {MAX_DEPTH} | Delai : {DELAY_BETWEEN_REQUESTS}s")
    print()

    start_time = time.time()
    connector  = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS, ssl=False, limit_per_host=2)

    async with aiohttp.ClientSession(connector=connector) as session:
        while pages_scraped < max_pages:
            batch = get_next_urls(batch_size=CONCURRENT_REQUESTS * 2)

            if not batch:
                log("File d'attente vide. Crawl termine !", "ok")
                break

            # Marquer en processing pour eviter la re-servitude par get_next_urls
            for item in batch:
                update_queue_status(item["id"], "processing")

            # Filtrer uniquement les URLs deja EN COURS de traitement (async)
            to_process = [item for item in batch if item["url"] not in urls_en_cours]
            for item in to_process:
                urls_en_cours.add(item["url"])

            # Marquer skip les vrais doublons
            ids_ok = {item["id"] for item in to_process}
            for item in batch:
                if item["id"] not in ids_ok:
                    update_queue_status(item["id"], "skip")

            if not to_process:
                await asyncio.sleep(0.1)
                continue

            await asyncio.gather(*[
                process_page(session, item["url"], item["depth"], item["id"], max_pages)
                for item in to_process
            ], return_exceptions=True)

    elapsed = time.time() - start_time
    stats   = get_stats()
    print(f"\n{'='*52}")
    print(f"Crawl termine en {elapsed:.1f}s")
    print(f"   Pages scrapees   : {stats['total_pages']}")
    print(f"   PDFs telecharges : {stats['total_pdfs']}")
    print(f"   Erreurs          : {stats['errors']}")
    if stats.get("by_category"):
        print(f"\n   Repartition par categorie :")
        for cat, n in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"     {cat:<22} : {n} pages")
    print(f"{'='*52}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        init_db()
        stats = get_stats()
        for k, v in stats.items():
            print(f"   {k:<20} : {v}")
        return

    asyncio.run(run_crawler(max_pages=args.max_pages))


if __name__ == "__main__":
    main()