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
TEXT_SPLITTER_CHUNK_SIZE = 500
TEXT_SPLITTER_CHUNK_OVERLAP = 50
QDRANT_UPLOAD_BATCH_SIZE = 100  # Retained for potential manual batching logic if re-added


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


# --- Main Script Logic ---
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

    args = parser.parse_args()

    md_folder = args.md_folder
    collection_name = args.collection
    document_type = args.doctype
    embeddings_file = args.embedfile

    if not os.path.isdir(md_folder):
        logging.error(f"Markdown folder not found: {md_folder}")
        return

    # 🔹 Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=60)
        logging.info(f"Successfully connected to Qdrant at {QDRANT_CLOUD_URL}.")
        # Consider adding collection creation logic if it doesn't exist
        # try:
        #     qdrant_client.get_collection(collection_name)
        # except Exception: # Replace with specific Qdrant exception if available
        #     logging.info(f"Collection '{collection_name}' not found. Attempting to create it.")
        #     # You'll need to know the vector size of your embeddings model, e.g., 1536 for text-embedding-ada-002
        #     vector_size = 1536 # Example for text-embedding-ada-002
        #     qdrant_client.recreate_collection(
        #         collection_name=collection_name,
        #         vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        #     )
        #     logging.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {e}")
        return

    # 🔹 Initialize OpenAI embeddings
    try:
        embeddings_model = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        logging.info(f"OpenAI embeddings model '{DEFAULT_EMBEDDING_MODEL}' initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI embeddings: {e}")
        return

    # 🔹 Load all Markdown files from the folder
    try:
        md_files = glob.glob(os.path.join(md_folder, "*.md"))
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

            text_content = md_data.content.strip()
            if not text_content:
                logging.warning(
                    f"No content found in {os.path.basename(file_path)} after removing frontmatter. Skipping.")
                continue

            documents.append({
                "text": text_content,
                "file_name": os.path.basename(file_path),
                "date_created": date_created_formatted,
                "type": document_type
            })
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

    if not documents:
        logging.info("No documents were successfully processed.")
        return

    # 🔹 Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
    )
    split_docs = []  # This will store dicts with text and metadata for each chunk
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk_text in chunks:
            split_docs.append({
                "text": chunk_text,  # This is the actual text chunk
                "file_name": doc["file_name"],
                "date_created": doc["date_created"],
                "type": doc["type"]
            })

    logging.info(f"Created {len(split_docs)} text chunks from {len(documents)} processed Markdown files.")

    # 🔹 Generate or Load Embeddings
    # embedded_docs_with_payload will store tuples of (embedding_vector, payload_dict)
    # The payload_dict is essentially the item from split_docs (which includes the text chunk itself)
    embedded_docs_with_payload = []
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "rb") as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, list) and all(
                        isinstance(item, tuple) and len(item) == 2 and
                        isinstance(item[0], list) and isinstance(item[1], dict)
                        for item in loaded_data
                ):
                    embedded_docs_with_payload = loaded_data
                    logging.info(
                        f"Loaded {len(embedded_docs_with_payload)} embeddings from cache file: {embeddings_file}")
                else:
                    logging.warning(f"Cache file {embeddings_file} has unexpected format. Regenerating embeddings.")
                    if os.path.exists(embeddings_file):  # Ensure removal if invalid
                        os.remove(embeddings_file)
        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as e:  # Added AttributeError
            logging.warning(
                f"Error loading embeddings from cache file {embeddings_file}: {e}. Regenerating embeddings.")
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
        except Exception as e:
            logging.error(f"An unexpected error occurred loading embeddings from {embeddings_file}: {e}. Regenerating.")
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)

    if not embedded_docs_with_payload:
        logging.info(f"Generating embeddings for {len(split_docs)} chunks. This may take a while...")
        try:
            for i, doc_payload in enumerate(split_docs):  # doc_payload is a dict from split_docs
                embedding_vector = embeddings_model.embed_query(doc_payload["text"])
                # The payload stored in the cache now directly matches Qdrant's payload structure
                embedded_docs_with_payload.append((embedding_vector, doc_payload))
                if (i + 1) % 50 == 0:
                    logging.info(f"Generated embeddings for {i + 1}/{len(split_docs)} chunks.")

            with open(embeddings_file, "wb") as f:
                pickle.dump(embedded_docs_with_payload, f)
            logging.info(f"Embeddings generated and saved to {embeddings_file}")
        except Exception as e:
            logging.error(f"Failed to generate or save embeddings: {e}")
            return

    # 🔹 Prepare points for Qdrant using UUIDs
    points_to_upload = []
    for embedding_vector, doc_payload in embedded_docs_with_payload:
        points_to_upload.append(
            models.PointStruct(
                id=str(uuid.uuid4()),  # ✅ Use UUID for unique string ID
                vector=embedding_vector,
                payload=doc_payload  # doc_payload already contains text, file_name, date_created, type
            )
        )

    # 🔹 Upsert Data to Qdrant
    if not points_to_upload:
        logging.info("No points to upload to Qdrant.")
        return

    logging.info(f"Starting upload of {len(points_to_upload)} points to Qdrant collection '{collection_name}'...")
    try:
        # Qdrant client's upsert handles batching for lists of points.
        # For extremely large datasets, manual batching (looping with QDRANT_UPLOAD_BATCH_SIZE)
        # might offer better progress tracking or memory control.
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upload,
            wait=True
        )
        logging.info(
            f"✅ All {len(points_to_upload)} points successfully stored in Qdrant collection '{collection_name}'!")

    except Exception as e:
        logging.error(f"Failed to upload data to Qdrant: {e}")


if __name__ == "__main__":
    main()
