# Markdown to Qdrant Embeddings Processor

This Python script, `LoadTextDataToQdrantCollection.py`, processes a folder of Markdown files. It extracts their content and specified metadata (like `date_created`), generates text embeddings using the OpenAI API (specifically `text-embedding-ada-002`), and stores these embeddings along with their metadata in a Qdrant vector database.

The script is designed to be configurable via environment variables and command-line arguments. It also includes a caching mechanism for embeddings to save on API calls and processing time during subsequent runs.

## Features

* Processes all `.md` files in a specified directory.
* Extracts frontmatter metadata (specifically looking for `date_created`, `created`, or `creation_date`).
* Uses `python-dateutil` for robust date parsing, converting extracted dates into a standardized `YYYY-MM-DDTHH:MM:SSZ` format.
* Splits document content into smaller, manageable chunks for optimal embedding.
* Generates embeddings using OpenAI's `text-embedding-ada-002` model.
* Caches generated embeddings locally (as `(embedding_vector, payload_dictionary)` tuples) to avoid redundant processing and API calls.
* Uploads embeddings and associated metadata (text chunk, original file name, creation date, document type) to a Qdrant collection.
* Uses **UUIDs (Universally Unique Identifiers)** for Qdrant point IDs, ensuring uniqueness and facilitating additive updates without accidental overwrites.
* Configurable via environment variables for sensitive information (API keys, Qdrant URL) and default settings.
* Provides command-line arguments for input folder path and to override certain configurations.
* Includes structured logging for better traceability and debugging.

## Prerequisites

* Python 3.8 or higher.
* `pip` (Python package installer).
* An active OpenAI API key.
* Access to a Qdrant Cloud instance or a self-hosted Qdrant database, along with its URL and API key (if your instance requires one).

## Setup Instructions

1.  **Clone the Repository (if applicable)**
    If this script is part of a Git repository, clone it first:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.
    ```bash
    # Create a virtual environment (e.g., named .venv)
    python -m venv .venv

    # Activate the virtual environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows (Command Prompt):
    # .venv\Scripts\activate.bat
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    ```
    **Important**: If you are troubleshooting dependency issues or upgrading, it's often best to delete your old `.venv` folder and create a fresh one before installing dependencies.

3.  **Install Dependencies**
    Install the required Python packages using the `requirements.txt` file provided with this script:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```txt
    qdrant-client>=1.7.0,<2.0.0
    langchain-core>=0.1.52,<0.2.0
    langchain-openai>=0.1.7,<0.2.0
    langchain>=0.1.20,<0.2.0
    python-frontmatter>=1.0.0,<2.0.0
    python-dateutil>=2.8.0,<3.0.0
    # uuid is part of the Python standard library, no need to add it here.
    ```

4.  **Set Environment Variables**
    This script requires certain environment variables to be set for API keys and Qdrant connection details.

    * **`OPENAI_API_KEY`**: Your API key for OpenAI.
    * **`QDRANT_CLOUD_URL`**: The URL of your Qdrant instance (e.g., `https://<your-qdrant-id>.europe-west3-0.gcp.cloud.qdrant.io`).
    * **`QDRANT_API_KEY`**: Your Qdrant API key (if your instance requires one).

    You can also set these optional environment variables to override default script behavior:
    * **`QDRANT_COLLECTION_NAME`**: Name of the Qdrant collection (defaults to `LinkedInPosts` if not set).
    * **`DOCUMENT_TYPE`**: Type of document for metadata (defaults to `note` if not set).

    **How to set environment variables:**

    * **Linux/macOS (Terminal)**:
        ```bash
        export OPENAI_API_KEY="your_openai_key_here"
        export QDRANT_CLOUD_URL="your_qdrant_url_here"
        export QDRANT_API_KEY="your_qdrant_api_key_here"
        # Optional
        # export QDRANT_COLLECTION_NAME="MyCustomCollection"
        # export DOCUMENT_TYPE="article"
        ```
    * **Windows (Command Prompt)**:
        ```cmd
        set OPENAI_API_KEY=your_openai_key_here
        set QDRANT_CLOUD_URL=your_qdrant_url_here
        set QDRANT_API_KEY=your_qdrant_api_key_here
        ```
    * **Windows (PowerShell)**:
        ```powershell
        $env:OPENAI_API_KEY="your_openai_key_here"
        $env:QDRANT_CLOUD_URL="your_qdrant_url_here"
        $env:QDRANT_API_KEY="your_qdrant_api_key_here"
        ```
    **Note**: For persistent storage of environment variables, consider adding them to your shell's profile file (e.g., `.bashrc`, `.zshrc`) or using a `.env` file with a library like `python-dotenv` (this would require a minor modification to the script to load the `.env` file). **Do not commit your `.env` file with actual keys to Git.**



5.  **Create Qdrant Collection Manually (Important!)**
    Before running the script for the first time (or if the collection does not exist), you need to manually create the collection in your Qdrant instance. The script currently does not automatically create the collection.

    You can do this using the Qdrant Dashboard Console or the Qdrant API.

    **Using the Qdrant Dashboard Console:**
    Navigate to your Qdrant console. The URL will be similar to:
    `https://<your-qdrant-url-placeholder>:6333/dashboard#/console`
    (Replace `<your-qdrant-url-placeholder>` with the actual base URL of your Qdrant instance, e.g., `9dfb92a4-e341-4553-94bd-e95f123456789.europe-west3-0.gcp.cloud.qdrant.io`)

    In the console, execute a PUT request. For a collection named `testCollection` (replace with your desired collection name, which should match `QDRANT_COLLECTION_NAME` if set, or the script's default), the request would look like this:

    ```http
    PUT collections/testCollection
    {
      "vectors": {
        "size": 1536,
        "distance": "Cosine"
      }
    }
    ```
    * **`collections/testCollection`**: Replace `testCollection` with the actual name of your collection.
    * **`size: 1536`**: This is the vector dimension for OpenAI's `text-embedding-ada-002` model. If you use a different embedding model, adjust this size accordingly.
    * **`distance: "Cosine"`**: This is a common distance metric for text embeddings. Other options include "Dot" or "Euclid".

    Ensure the collection is successfully created before proceeding.

## How to Run the Script

Once the setup is complete and environment variables are set, you can run the script from your terminal.

**Basic Usage:**

Provide the path to the folder containing your Markdown files as a command-line argument.

```bash
python LoadTextDataToQdrantCollection.py /path/to/your/markdown_folder
```

Example:

If your Markdown files are in a subfolder named my_blog_posts relative to where the script is:

```bash
python LoadTextDataToQdrantCollection.py ./my_blog_posts
```

Using Optional Arguments:

You can override default settings for the Qdrant collection name, document type, and embeddings cache file path using command-line arguments:
```bash
python LoadTextDataToQdrantCollection.py /path/to/markdown_folder \
    --collection "MyTechnicalArticles" \
    --doctype "article" \
    --embedfile "cache/my_article_embeddings.pickle"
```


### Script Arguments

* `md_folder` (Required): Positional argument specifying the path to the folder containing your Markdown (`.md`) files.

* `--collection TEXT`: (Optional) The name of the Qdrant collection to use.

  * Overrides `QDRANT_COLLECTION_NAME` environment variable if set.

  * Default: `LinkedInPosts` (if neither this argument nor the environment variable is set).

* `--doctype TEXT`: (Optional) The document type to be stored in the metadata for each processed item.

  * Overrides `DOCUMENT_TYPE` environment variable if set.

  * Default: `note` (if neither this argument nor the environment variable is set).

* `--embedfile TEXT`: (Optional) The file path for storing and loading cached embeddings.

  * Default: `embeddings_cache.pickle` (created in the current working directory).


Embeddings Cache

To save costs and time, the script caches generated embeddings.

* By default, it creates a file named `embeddings_cache.pickle` in the directory where the script is run (or the path specified by `--embedfile`).

* This file stores a list of tuples, where each tuple contains `(embedding_vector, payload_dictionary)`.

* On subsequent runs, if this cache file exists and is valid, the script loads embeddings from it instead of re-generating them via the OpenAI API.

* If you modify your Markdown files or want to force re-generation of embeddings for any reason, simply delete this cache file before running the script again.

## Qdrant Point IDs and Data Updates

The script assigns **Universally Unique Identifiers (UUIDs)** as strings (e.g., `"a1b2c3d4-e5f6-7890-1234-567890abcdef"`) to each point (document/chunk) uploaded to Qdrant.

**Benefits of using UUIDs:**

* **Global Uniqueness**: UUIDs are designed to be unique across different systems and different times. This significantly reduces the chance of ID collisions if you are ingesting data from multiple sources or over multiple runs.

* **Additive Updates**: When you re-run the script, new documents or chunks will get new, unique UUIDs. This makes the script suitable for adding new data to an existing Qdrant collection without overwriting previous entries based on simple integer IDs.

* **No Accidental Overwriting**: Unlike sequential integer IDs that might reset or collide, random UUIDs prevent accidental overwriting of existing data when the script is run multiple times.

**Considerations for Updating Existing Content:**

* The current script with random UUIDs treats each processed chunk as a new, independent entry. If a Markdown file's content changes and you re-process it, it will be stored as a *new* point with a *new* UUID in Qdrant, rather than updating the old version.

* If you need to *update* existing documents in Qdrant (e.g., if a Markdown file's content changes and you want to update its embedding rather than add a duplicate), you would need a more sophisticated ID management strategy. This might involve:

  1. Generating *deterministic* UUIDs based on a stable identifier from the source content (e.g., a hash of the file path, or a unique ID field in your frontmatter).

  2. Storing a mapping of your source document identifiers to these deterministic Qdrant UUIDs.

  3. Before uploading, checking if a document with a specific deterministic ID already exists and then using Qdrant's `upsert` operation (which updates if the ID exists, or inserts if it doesn't) with that deterministic ID.

## Logging

The script uses Python's built-in `logging` module to output information about its progress, warnings, and any errors encountered. Log messages include timestamps and severity levels, which helps in monitoring the script's execution and troubleshooting issues.


