import os
import pickle
import glob
import frontmatter
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import re
import argparse
import logging
import uuid  # Added for UUID generation
from dateutil import parser as dateutil_parser  # Moved dateutil import to top level
from typing import Optional
import hashlib
from copy import deepcopy

# Optional: .env laden, damit Keys lokal bleiben
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load sensitive data and configurations from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_CLOUD_URL = os.environ.get("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Check if essential environment variables are set
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set.")
    exit(1)
if not QDRANT_CLOUD_URL:
    logging.error("QDRANT_CLOUD_URL environment variable not set.")
    exit(1)
# QDRANT_API_KEY can be optional if your Qdrant instance doesn't require it,
# but it's good practice to check if it's expected.
if not QDRANT_API_KEY:
    logging.warning(
        "QDRANT_API_KEY environment variable not set. Proceeding without it if not required by your Qdrant setup.")

# --- Constants and Default Values ---
DEFAULT_COLLECTION_NAME = "LinkedInPosts"
DEFAULT_DOCUMENT_TYPE = "note"
DEFAULT_EMBEDDINGS_FILE = "embeddings_cache.pickle"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
TEXT_SPLITTER_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
TEXT_SPLITTER_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
QDRANT_UPLOAD_BATCH_SIZE = int(os.environ.get("UPLOAD_BATCH_SIZE", "256"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "128"))


# --- Helper Functions ---
def convert_date_format(date_str):
    """
    Converts a date string from various common formats
    to 'YYYY-MM-DDTHH:MM:SSZ'.
    Handles ordinal suffixes (st, nd, rd, th) in the day.
    Returns None if the input is empty, "Unknown", or parsing fails.
    Note: The 'Z' in the output format implies UTC. This function currently
    formats the naive datetime object. If the source time is local and needs
    to be converted to UTC, timezone handling would be required.
    """
    if not date_str or str(date_str).lower() == "unknown":
        return None

    date_str_cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', str(date_str))

    input_formats = [
        "%A, %B %d %Y, %I:%M:%S %p",  # Full format with time e.g. "Monday, May 13 2024, 5:00:50 PM"
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone
        "%Y-%m-%d %H:%M:%S%z",  # ISO-like with timezone
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without timezone (assumed naive or UTC)
        "%Y-%m-%d %H:%M:%S",  # ISO-like without T/Z
        "%Y-%m-%d",  # Date only
        "%B %d, %Y",  # Month Day, Year e.g. "May 13, 2024"
        "%d %B %Y",  # Day Month Year e.g. "13 May 2024"
        "%m/%d/%Y"  # MM/DD/YYYY
    ]

    dt = None
    for fmt in input_formats:
        try:
            dt = datetime.strptime(date_str_cleaned, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        # Try parsing with dateutil if standard formats fail, as it's more flexible
        try:
            # dateutil_parser is now imported at the top level
            dt = dateutil_parser.parse(date_str_cleaned)
        except (ValueError, TypeError) as e:  # Catch parsing errors from dateutil
            logging.warning(f"Failed to parse date: '{date_str}' with standard formats and dateutil. Error: {e}")
            return None
        except Exception as e:  # Catch any other unexpected error from dateutil
            logging.warning(f"An unexpected error occurred while parsing date '{date_str}' with dateutil: {e}")
            return None

    output_format = "%Y-%m-%dT%H:%M:%SZ"
    return dt.strftime(output_format)


def ensure_payload_indexes(qdrant_client: QdrantClient, collection_name: str):
    """
    Legt die benötigten Payload-Indizes an (idempotent).
    """
    try:
        logging.info(
            "Creating payload indexes (if not exist): "
            "date_created(datetime), tag(keyword), note_type(keyword), file_hash(keyword), source(keyword)"
        )
        # Siehe Qdrant Indexing Doku: https://qdrant.tech/documentation/concepts/indexing/
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="date_created",
            field_schema="datetime",
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="tag",
            field_schema="keyword",
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="note_type",
            field_schema="keyword",
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="file_hash",
            field_schema="keyword",
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema="keyword",
        )
    except Exception as e:
        logging.warning(f"Failed to create one or more payload indexes: {e}")


# --- Core Processing ---
def run_loader(
    md_folder,
    collection_name=DEFAULT_COLLECTION_NAME,
    document_type=DEFAULT_DOCUMENT_TYPE,
    embeddings_file=DEFAULT_EMBEDDINGS_FILE,
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    upload_batch_size=QDRANT_UPLOAD_BATCH_SIZE,
    embedding_batch_size=EMBEDDING_BATCH_SIZE,
    create_indexes=False,
    create_collection_if_missing=False,
    vector_size: Optional[int] = 1536,
    dedupe_mode: str = "skip",  # skip | overwrite | off
    only_indexes: bool = False,  # wenn True: nur Indizes anlegen/aktualisieren, keine Embeddings/Uploads
):

    # Pfad-Validierung nur, wenn wir wirklich Dateien verarbeiten
    if not only_indexes:
        if not os.path.isdir(md_folder) and not os.path.isfile(md_folder):
            logging.error(f"Markdown folder or file not found: {md_folder}")
            return

    # 🔹 Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=60)
        logging.info(f"Successfully connected to Qdrant at {QDRANT_CLOUD_URL}.")
        # Optional: Collection anlegen, falls nicht vorhanden
        try:
            qdrant_client.get_collection(collection_name)
            logging.info(f"Collection '{collection_name}' exists.")
        except Exception:
            if create_collection_if_missing:
                vs = int(vector_size or 1536)
                logging.info(f"Collection '{collection_name}' not found. Creating with size={vs}, distance=COSINE...")
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vs, distance=models.Distance.COSINE),
                )
                logging.info(f"Collection '{collection_name}' created.")
            else:
                logging.warning(f"Collection '{collection_name}' not found and auto-create disabled.")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {e}")
        return

    # Nur Indizes neu bauen – keine Embeddings, kein Upload
    if only_indexes:
        if create_indexes:
            ensure_payload_indexes(qdrant_client, collection_name)
        else:
            logging.info("only_indexes=True, aber create_indexes=False – nichts zu tun.")
        return

    # 🔹 Initialize OpenAI embeddings
    try:
        embeddings_model = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        logging.info(f"OpenAI embeddings model '{DEFAULT_EMBEDDING_MODEL}' initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI embeddings: {e}")
        return

    # 🔹 Load all Markdown files from the folder or a single file
    try:
        md_files = []
        if os.path.isdir(md_folder):
            md_files = glob.glob(os.path.join(md_folder, "*.md"))
        elif os.path.isfile(md_folder):
            if md_folder.lower().endswith(".md"):
                md_files = [md_folder]
            else:
                logging.warning(f"Specified file is not a Markdown file: {md_folder}")
                return
        if not md_files:
            logging.warning(f"No Markdown files found in {md_folder}.")
            return
        logging.info(f"Found {len(md_files)} Markdown files to process.")
    except Exception as e:
        logging.error(f"Error accessing Markdown files in {md_folder}: {e}")
        return

    # 🔹 Process Markdown files
    documents = []
    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            md_data = frontmatter.loads(md_content)
            date_created_raw = md_data.get("date_created",
                                           md_data.get("created", md_data.get("creation_date", "Unknown")))
            date_created_formatted = convert_date_format(date_created_raw)
            if date_created_raw != "Unknown" and date_created_formatted is None:
                logging.warning(
                    f"Could not parse date '{date_created_raw}' from frontmatter in {os.path.basename(file_path)}. Storing as None.")

            # Parse note_type and tags from frontmatter
            note_type = md_data.get("note_type", md_data.get("type", document_type))
            raw_tags = md_data.get("tags", md_data.get("tag", []))
            if isinstance(raw_tags, str):
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif isinstance(raw_tags, list):
                # normalize to list of strings
                tags = [str(t).strip() for t in raw_tags if str(t).strip()]
            else:
                tags = []
            # Optional: source aus Frontmatter
            source = md_data.get("source", None)

            text_content = md_data.content.strip()
            if not text_content:
                logging.warning(
                    f"No content found in {os.path.basename(file_path)} after removing frontmatter. Skipping.")
                continue

            # Compute file hash for dedupe (hash of full file content)
            try:
                with open(file_path, "rb") as rf:
                    content_bytes = rf.read()
                file_hash = hashlib.sha256(content_bytes).hexdigest()
            except Exception as e:
                logging.warning(f"Could not compute hash for {file_path}: {e}")
                file_hash = None

            documents.append({
                "text": text_content,
                "file_name": os.path.basename(file_path),
                "date_created": date_created_formatted,
                "note_type": note_type,
                "tag": tags,
                "file_hash": file_hash,
                "source": source
            })
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

    if not documents:
        logging.info("No documents were successfully processed.")
        return

    # 🔹 Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = []  # This will store dicts with text and metadata for each chunk
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for idx, chunk_text in enumerate(chunks):
            # Hash des Chunk-Texts (für stabile IDs wenn kein file_hash existiert)
            try:
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            except Exception:
                chunk_hash = None
            split_docs.append({
                "text": chunk_text,  # This is the actual text chunk
                "file_name": doc["file_name"],
                "date_created": doc["date_created"],
                "note_type": doc["note_type"],
                "tag": doc["tag"],
                "file_hash": doc["file_hash"],
                "chunk_index": idx,
                "chunk_hash": chunk_hash,
                "source": doc.get("source")
            })

    logging.info(f"Created {len(split_docs)} text chunks from {len(documents)} processed Markdown files.")

    # 🔹 Dedupe pre-checks: build set of existing file_hashes in collection (if provided)
    existing_hashes = set()
    try:
        # Only check if any file has hash and dedupe isn't off
        hashes_to_check = {d["file_hash"] for d in documents if d.get("file_hash")}
        if hashes_to_check and dedupe_mode in ("skip", "overwrite"):
            for fh in hashes_to_check:
                flt = models.Filter(
                    must=[models.FieldCondition(key="file_hash", match=models.MatchValue(value=fh))]
                )
                # Fetch a single point to know if exists
                res = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=flt,
                    with_payload=False,
                    with_vectors=False,
                    limit=1
                )
                if res and res[0]:
                    existing_hashes.add(fh)
    except Exception as e:
        logging.warning(f"Failed to pre-check existing file hashes: {e}")

    # If overwrite: delete old points for files that already exist
    if dedupe_mode == "overwrite" and existing_hashes:
        try:
            logging.info(f"Overwrite mode: deleting existing points for {len(existing_hashes)} files")
            for fh in existing_hashes:
                flt = models.Filter(
                    must=[models.FieldCondition(key="file_hash", match=models.MatchValue(value=fh))]
                )
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=flt),
                    wait=True
                )
        except Exception as e:
            logging.warning(f"Failed to delete existing points in overwrite mode: {e}")

    # If skip: filter out docs belonging to existing hashes
    if dedupe_mode == "skip" and existing_hashes:
        before = len(split_docs)
        split_docs = [d for d in split_docs if not d.get("file_hash") or d["file_hash"] not in existing_hashes]
        logging.info(f"Skip mode: filtered chunks from {before} to {len(split_docs)} (skipped existing files).")

    # 🔹 Stable ID pro Chunk bestimmen (für Cache & Point-ID)
    def compute_stable_id(payload: dict) -> str:
        if payload.get("file_hash") is not None and payload.get("chunk_index") is not None:
            base = f"{payload['file_hash']}:{payload['chunk_index']}"
        elif payload.get("chunk_hash") is not None and payload.get("chunk_index") is not None:
            base = f"{payload['chunk_hash']}:{payload['chunk_index']}"
        else:
            base = f"{payload.get('file_name','unknown')}:{payload.get('chunk_index','0')}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    for item in split_docs:
        item["stable_id"] = compute_stable_id(item)

    # 🔹 Generate or Load Embeddings - Cache per stable_id
    cache: dict = {}
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache = loaded
                    logging.info(f"Loaded embeddings cache with {len(cache)} entries.")
                elif isinstance(loaded, list):
                    # Backward-compat: try to convert list[(vector, payload)] to dict by computing stable_id
                    temp = {}
                    for vec, payload in loaded:
                        if isinstance(payload, dict):
                            sid = compute_stable_id(payload)
                            temp[sid] = (vec, payload)
                    cache = temp
                    logging.info(f"Migrated legacy cache to dict with {len(cache)} entries.")
                else:
                    logging.warning("Unknown cache format; starting fresh.")
        except Exception as e:
            logging.warning(f"Failed to load cache from {embeddings_file}: {e}. Starting with empty cache.")

    # Finde fehlende Stable-IDs
    missing = [d for d in split_docs if d["stable_id"] not in cache]
    logging.info(f"Embeddings to compute: {len(missing)} (from total {len(split_docs)})")

    if missing:
        try:
            texts = [d["text"] for d in missing]
            for start in range(0, len(texts), embedding_batch_size):
                batch_docs = missing[start:start + embedding_batch_size]
                batch_texts = [d["text"] for d in batch_docs]
                batch_vectors = embeddings_model.embed_documents(batch_texts)
                for d, vector in zip(batch_docs, batch_vectors):
                    # Payload in Cache ohne tiefe Kopie verändern?
                    cache[d["stable_id"]] = (vector, deepcopy(d))
                logging.info(f"Computed embeddings for {min(start + len(batch_texts), len(texts))}/{len(texts)} missing chunks.")
            # Cache speichern
            with open(embeddings_file, "wb") as f:
                pickle.dump(cache, f)
            logging.info(f"Embeddings cache updated with {len(missing)} new entries; total {len(cache)}.")
        except Exception as e:
            logging.error(f"Failed to generate or save embeddings: {e}")
            return

    # Baue die Uploadliste aus Cache in der Reihenfolge von split_docs
    embedded_docs_with_payload = []
    for d in split_docs:
        vector, payload = cache[d["stable_id"]]
        embedded_docs_with_payload.append((vector, payload))

    # 🔹 Prepare points for Qdrant using deterministic UUIDs per file_hash+chunk_index
    points_to_upload = []
    for embedding_vector, doc_payload in embedded_docs_with_payload:
        # Deterministische ID: uuid5 über file_hash:chunk_index, fallback auf random falls kein hash
        if doc_payload.get("file_hash") is not None and doc_payload.get("chunk_index") is not None:
            stable_key = f"{doc_payload['file_hash']}:{doc_payload['chunk_index']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        elif doc_payload.get("chunk_hash") is not None and doc_payload.get("chunk_index") is not None:
            stable_key = f"{doc_payload['chunk_hash']}:{doc_payload['chunk_index']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        else:
            point_id = str(uuid.uuid4())
        points_to_upload.append(
            models.PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=doc_payload  # doc_payload contains text, file_name, date_created, note_type, tag
            )
        )

    # 🔹 Optionally create payload indices (date_created, tag, note_type, file_hash, source)
    if create_indexes:
        ensure_payload_indexes(qdrant_client, collection_name)

    # 🔹 Upsert Data to Qdrant
    if not points_to_upload:
        logging.info("No points to upload to Qdrant.")
        return

    logging.info(
        f"Starting upload of {len(points_to_upload)} points in batches to Qdrant collection '{collection_name}'...")

    # Define the batch size from parameters
    batch_size = upload_batch_size

    try:
        # Loop through the points in chunks of batch_size
        for i in range(0, len(points_to_upload), batch_size):
            # Get the current batch of points
            batch = points_to_upload[i:i + batch_size]

            # Upsert the batch to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            logging.info(
                f"Uploaded batch {i // batch_size + 1}/{(len(points_to_upload) + batch_size - 1) // batch_size}")

        logging.info(
            f"✅ All {len(points_to_upload)} points successfully stored in Qdrant collection '{collection_name}'!")

    except Exception as e:
        logging.error(f"Failed to upload data to Qdrant: {e}")



def main():
    # 🔹 Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description="Process Markdown files, generate embeddings, and upload to Qdrant.")
    parser.add_argument("md_folder", help="Path to the folder containing Markdown files.")
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
                        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME} or QDRANT_COLLECTION_NAME env var).")
    parser.add_argument("--doctype", default=os.environ.get("DOCUMENT_TYPE", DEFAULT_DOCUMENT_TYPE),
                        help=f"Document type for metadata (default: {DEFAULT_DOCUMENT_TYPE} or DOCUMENT_TYPE env var).")
    parser.add_argument("--embedfile", default=DEFAULT_EMBEDDINGS_FILE,
                        help=f"Path to store/load cached embeddings (default: {DEFAULT_EMBEDDINGS_FILE}).")
    parser.add_argument("--chunk-size", type=int, default=TEXT_SPLITTER_CHUNK_SIZE, help="Text splitter chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=TEXT_SPLITTER_CHUNK_OVERLAP, help="Text splitter chunk overlap.")
    parser.add_argument("--batch-size", type=int, default=QDRANT_UPLOAD_BATCH_SIZE, help="Qdrant upsert batch size.")
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--create-indexes", action="store_true", help="Create payload indexes for date_created, tag, note_type.")
    parser.add_argument("--create-collection", action="store_true", help="Create collection if it does not exist.")
    parser.add_argument("--vector-size", type=int, default=1536, help="Vector size when creating a collection (default 1536 for ada-002).")
    parser.add_argument("--dedupe", choices=["skip", "overwrite", "off"], default="skip",
                        help="Dedupe strategy per file: skip=überspringt vorhandene, overwrite=löscht und ersetzt, off=immer hinzufügen.")
    parser.add_argument("--only-indexes", action="store_true",
                        help="Nur Payload-Indizes anlegen/aktualisieren (keine Embeddings, kein Upload).")

    args = parser.parse_args()

    run_loader(
        md_folder=args.md_folder,
        collection_name=args.collection,
        document_type=args.doctype,
        embeddings_file=args.embedfile,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        upload_batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        create_indexes=bool(args.create_indexes),
        create_collection_if_missing=bool(args.create_collection),
        vector_size=args.vector_size,
        dedupe_mode=args.dedupe,
        only_indexes=bool(args.only_indexes),
    )


if __name__ == "__main__":
    main()
