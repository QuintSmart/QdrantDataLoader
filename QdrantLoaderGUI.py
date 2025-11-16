#!/usr/bin/env python3
"""
Einfache macOS/Windows GUI für den Qdrant Data Loader (Tkinter-basiert).
Erlaubt die Auswahl eines Markdown-Ordners, Konfiguration von Chunking/Batching,
und optionales Erstellen von Payload-Indizes (date_created, tag, note_type).
"""

import threading
import queue
import logging
import os
import sys
from pathlib import Path
import json

# Tkinter Import
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except ImportError:
    print("Fehler: tkinter ist nicht installiert! Installiere ggf. mit: pip install tk")
    sys.exit(1)

# Importiere die Loader-Funktion aus dem Skript
from LoadTextDataToQdrantCollection import run_loader, DEFAULT_COLLECTION_NAME, DEFAULT_DOCUMENT_TYPE, DEFAULT_EMBEDDINGS_FILE, TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP, QDRANT_UPLOAD_BATCH_SIZE, EMBEDDING_BATCH_SIZE


class TkLoggingHandler(logging.Handler):
    """Leitet Log-Meldungen in eine Tkinter-Queue um."""
    def __init__(self, tk_queue):
        super().__init__()
        self.tk_queue = tk_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tk_queue.put(msg)
        except Exception:
            pass


class QdrantLoaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qdrant Data Loader")
        self.root.geometry("820x600")
        self.root.minsize(800, 560)
        self.settings_path = Path.home() / ".qdrant_loader_gui.json"

        self.log_queue = queue.Queue()
        self._setup_logging()
        self._build_ui()
        self._poll_log_queue()

        # Defaults
        self.var_collection.set(os.environ.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME))
        self.var_doctype.set(os.environ.get("DOCUMENT_TYPE", DEFAULT_DOCUMENT_TYPE))
        self.var_embedfile.set(DEFAULT_EMBEDDINGS_FILE)
        self.var_chunk_size.set(TEXT_SPLITTER_CHUNK_SIZE)
        self.var_chunk_overlap.set(TEXT_SPLITTER_CHUNK_OVERLAP)
        self.var_batch_size.set(QDRANT_UPLOAD_BATCH_SIZE)
        self.var_embedding_batch_size.set(EMBEDDING_BATCH_SIZE)
        self.var_create_indexes.set(True)
        self.var_create_collection.set(True)
        self.var_vector_size.set(1536)
        self.var_dedupe_mode.set("skip")

        # Worker Thread Ref
        self.worker_thread = None

        # Lade gespeicherte Einstellungen (falls vorhanden)
        try:
            self._load_settings()
        except Exception:
            pass

    def _setup_logging(self):
        self.logger = logging.getLogger("QdrantLoaderGUI")
        self.logger.setLevel(logging.INFO)
        # Stream to Tk
        handler = TkLoggingHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # Root logger, damit Logs aus run_loader ebenfalls erscheinen
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _build_ui(self):
        pad_x = 10
        pad_y = 8

        frm_top = ttk.Frame(self.root)
        frm_top.pack(side=tk.TOP, fill=tk.X, padx=pad_x, pady=pad_y)

        # Markdown Folder
        self.var_md_folder = tk.StringVar()
        ttk.Label(frm_top, text="Markdown-Ordner:").grid(row=0, column=0, sticky="w")
        ent_md = ttk.Entry(frm_top, textvariable=self.var_md_folder, width=70)
        ent_md.grid(row=0, column=1, sticky="we", padx=(6, 6))
        btn_browse = ttk.Button(frm_top, text="Durchsuchen…", command=self._browse_folder)
        btn_browse.grid(row=0, column=2, sticky="e")
        frm_top.grid_columnconfigure(1, weight=1)

        # Collection
        self.var_collection = tk.StringVar()
        ttk.Label(frm_top, text="Collection:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_collection, width=30).grid(row=1, column=1, sticky="w", padx=(6, 6))

        # Document Type
        self.var_doctype = tk.StringVar()
        ttk.Label(frm_top, text="Note Type (default):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_doctype, width=30).grid(row=2, column=1, sticky="w", padx=(6, 6))

        # Embeddings Cache
        self.var_embedfile = tk.StringVar()
        ttk.Label(frm_top, text="Embeddings Cache-Datei:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_embedfile, width=50).grid(row=3, column=1, sticky="we", padx=(6, 6))
        ttk.Button(frm_top, text="Wählen…", command=self._choose_cache).grid(row=3, column=2, sticky="e")

        # Chunking
        frm_params = ttk.LabelFrame(self.root, text="Parameter")
        frm_params.pack(side=tk.TOP, fill=tk.X, padx=pad_x, pady=pad_y)

        self.var_chunk_size = tk.IntVar()
        self.var_chunk_overlap = tk.IntVar()
        self.var_batch_size = tk.IntVar()
        self.var_embedding_batch_size = tk.IntVar()
        self.var_create_indexes = tk.BooleanVar()
        self.var_create_collection = tk.BooleanVar()
        self.var_vector_size = tk.IntVar()
        self.var_dedupe_mode = tk.StringVar()

        ttk.Label(frm_params, text="Chunk Size:").grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
        ttk.Entry(frm_params, textvariable=self.var_chunk_size, width=10).grid(row=0, column=1, sticky="w", pady=(6, 2))

        ttk.Label(frm_params, text="Chunk Overlap:").grid(row=0, column=2, sticky="w", padx=(24, 6), pady=(6, 2))
        ttk.Entry(frm_params, textvariable=self.var_chunk_overlap, width=10).grid(row=0, column=3, sticky="w", pady=(6, 2))

        ttk.Label(frm_params, text="Upload Batch Size:").grid(row=1, column=0, sticky="w", padx=(6, 6), pady=(2, 6))
        ttk.Entry(frm_params, textvariable=self.var_batch_size, width=10).grid(row=1, column=1, sticky="w", pady=(2, 6))

        ttk.Label(frm_params, text="Embedding Batch Size:").grid(row=1, column=2, sticky="w", padx=(24, 6), pady=(2, 6))
        ttk.Entry(frm_params, textvariable=self.var_embedding_batch_size, width=10).grid(row=1, column=3, sticky="w", pady=(2, 6))

        ttk.Checkbutton(frm_params, text="Payload-Indizes erstellen (date_created, tag, note_type)", variable=self.var_create_indexes).grid(row=2, column=0, columnspan=4, sticky="w", padx=(6, 6), pady=(2, 6))
        ttk.Checkbutton(frm_params, text="Collection anlegen falls fehlt", variable=self.var_create_collection).grid(row=3, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=(2, 10))
        ttk.Label(frm_params, text="Vector Size (bei Neuanlage):").grid(row=3, column=2, sticky="e", padx=(6, 6))
        ttk.Entry(frm_params, textvariable=self.var_vector_size, width=10).grid(row=3, column=3, sticky="w", padx=(0, 6))

        # Dedupe Mode
        ttk.Label(frm_params, text="Dedupe-Modus:").grid(row=4, column=0, sticky="w", padx=(6, 6), pady=(2, 10))
        cmb_dedupe = ttk.Combobox(frm_params, textvariable=self.var_dedupe_mode, width=12, state="readonly",
                                  values=["skip", "overwrite", "off"])
        cmb_dedupe.grid(row=4, column=1, sticky="w", padx=(0, 6), pady=(2, 10))
        cmb_dedupe.current(0)

        # Aktionen
        frm_actions = ttk.Frame(self.root)
        frm_actions.pack(side=tk.TOP, fill=tk.X, padx=pad_x, pady=(0, pad_y))
        ttk.Button(frm_actions, text="Start", command=self._start).pack(side=tk.LEFT)
        ttk.Button(frm_actions, text="Cache leeren", command=self._clear_cache).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(frm_actions, text="Abbrechen", command=self._cancel).pack(side=tk.LEFT, padx=(8, 0))

        # Log
        frm_log = ttk.LabelFrame(self.root, text="Log")
        frm_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=pad_x, pady=pad_y)
        self.txt_log = scrolledtext.ScrolledText(frm_log, wrap=tk.WORD, height=18)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Markdown-Ordner wählen")
        if path:
            self.var_md_folder.set(path)

    def _choose_cache(self):
        path = filedialog.asksaveasfilename(
            title="Embeddings Cache-Datei wählen",
            defaultextension=".pickle",
            initialfile=DEFAULT_EMBEDDINGS_FILE,
            filetypes=[("Pickle", "*.pickle"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.var_embedfile.set(path)

    def _append_log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def _start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Info", "Ein Lauf ist bereits aktiv.")
            return
        md_folder = self.var_md_folder.get().strip()
        if not md_folder or not os.path.isdir(md_folder):
            messagebox.showerror("Fehler", "Bitte einen gültigen Markdown-Ordner wählen.")
            return

        # Einstellungen speichern
        try:
            self._save_settings()
        except Exception:
            pass

        params = dict(
            md_folder=md_folder,
            collection_name=self.var_collection.get().strip() or DEFAULT_COLLECTION_NAME,
            document_type=self.var_doctype.get().strip() or DEFAULT_DOCUMENT_TYPE,
            embeddings_file=self.var_embedfile.get().strip() or DEFAULT_EMBEDDINGS_FILE,
            chunk_size=int(self.var_chunk_size.get()),
            chunk_overlap=int(self.var_chunk_overlap.get()),
            upload_batch_size=int(self.var_batch_size.get()),
            embedding_batch_size=int(self.var_embedding_batch_size.get()),
            create_indexes=bool(self.var_create_indexes.get()),
            create_collection_if_missing=bool(self.var_create_collection.get()),
            vector_size=int(self.var_vector_size.get() or 1536),
            dedupe_mode=self.var_dedupe_mode.get(),
        )

        self._append_log("Starte Upload…")
        self.worker_thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self.worker_thread.start()

    def _run_worker(self, params):
        try:
            run_loader(**params)
            self.log_queue.put("Fertig.")
        except Exception as e:
            self.log_queue.put(f"Fehler: {e}")

    def _cancel(self):
        messagebox.showinfo("Info", "Abbrechen wird nicht unterstützt. Du kannst das Fenster schließen, um den Prozess zu beenden.")

    def _clear_cache(self):
        path = self.var_embedfile.get().strip()
        if not path:
            messagebox.showinfo("Info", "Kein Cache-Pfad gesetzt.")
            return
        try:
            if os.path.exists(path):
                os.remove(path)
                self._append_log(f"Cache gelöscht: {path}")
                messagebox.showinfo("Erfolg", f"Cache gelöscht:\n{path}")
            else:
                messagebox.showinfo("Info", "Cache-Datei existiert nicht.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Cache nicht löschen:\n{e}")

    def _save_settings(self):
        data = {
            "md_folder": self.var_md_folder.get().strip(),
            "collection": self.var_collection.get().strip(),
            "doctype": self.var_doctype.get().strip(),
            "embedfile": self.var_embedfile.get().strip(),
            "chunk_size": int(self.var_chunk_size.get() or 0),
            "chunk_overlap": int(self.var_chunk_overlap.get() or 0),
            "batch_size": int(self.var_batch_size.get() or 0),
            "embedding_batch_size": int(self.var_embedding_batch_size.get() or 0),
            "create_indexes": bool(self.var_create_indexes.get()),
            "create_collection": bool(self.var_create_collection.get()),
            "vector_size": int(self.var_vector_size.get() or 1536),
            "dedupe_mode": self.var_dedupe_mode.get(),
        }
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_settings(self):
        if not self.settings_path.exists():
            return
        with open(self.settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Felder übernehmen, wenn vorhanden
        if "md_folder" in data:
            self.var_md_folder.set(data["md_folder"])
        if "collection" in data:
            self.var_collection.set(data["collection"])
        if "doctype" in data:
            self.var_doctype.set(data["doctype"])
        if "embedfile" in data:
            self.var_embedfile.set(data["embedfile"])
        if "chunk_size" in data:
            self.var_chunk_size.set(int(data["chunk_size"]))
        if "chunk_overlap" in data:
            self.var_chunk_overlap.set(int(data["chunk_overlap"]))
        if "batch_size" in data:
            self.var_batch_size.set(int(data["batch_size"]))
        if "embedding_batch_size" in data:
            self.var_embedding_batch_size.set(int(data["embedding_batch_size"]))
        if "create_indexes" in data:
            self.var_create_indexes.set(bool(data["create_indexes"]))
        if "create_collection" in data:
            self.var_create_collection.set(bool(data["create_collection"]))
        if "vector_size" in data:
            self.var_vector_size.set(int(data["vector_size"]))
        if "dedupe_mode" in data and data["dedupe_mode"] in ("skip", "overwrite", "off"):
            self.var_dedupe_mode.set(data["dedupe_mode"])


def main():
    root = tk.Tk()
    app = QdrantLoaderGUI(root)
    # Speichere Einstellungen beim Schließen
    def on_close():
        try:
            app._save_settings()
        except Exception:
            pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()


