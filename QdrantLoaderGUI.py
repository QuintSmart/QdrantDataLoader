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
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
import re

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

        # Defaults werden in _build_ui gesetzt

        # Worker Thread Ref
        self.worker_thread = None

        # Lade gespeicherte Einstellungen (falls vorhanden)
        try:
            self._load_settings()
        except Exception:
            pass

        # Lade Collections aus Qdrant (falls möglich)
        try:
            self._load_collections()
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

        # Markdown Pfad (Ordner oder Datei)
        self.var_md_folder = tk.StringVar()

        ttk.Label(frm_top, text="Markdown-Pfad:").grid(row=0, column=0, sticky="w")
        ent_md = ttk.Entry(frm_top, textvariable=self.var_md_folder, width=70)
        ent_md.grid(row=0, column=1, sticky="we", padx=(6, 6))
        frm_top.grid_columnconfigure(1, weight=1)

        btn_browse_folder = ttk.Button(frm_top, text="Ordner…", command=self._browse_folder)
        btn_browse_folder.grid(row=0, column=2, sticky="e", padx=(0, 4))
        btn_browse_file = ttk.Button(frm_top, text="Datei…", command=self._browse_file)
        btn_browse_file.grid(row=1, column=2, sticky="e", padx=(0, 4))

        # Collection (mit Liste aus Qdrant)
        self.var_collection = tk.StringVar()
        ttk.Label(frm_top, text="Collection:").grid(row=2, column=0, sticky="w")
        self.cmb_collection = ttk.Combobox(frm_top, textvariable=self.var_collection, width=40)
        self.cmb_collection.grid(row=2, column=1, sticky="w", padx=(6, 6))
        ttk.Button(frm_top, text="Aktualisieren", command=self._load_collections).grid(
            row=2, column=2, sticky="e"
        )

        # Document Type
        self.var_doctype = tk.StringVar()
        ttk.Label(frm_top, text="Note Type (default):").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_doctype, width=30).grid(row=3, column=1, sticky="w", padx=(6, 6))

        # Embeddings Cache
        self.var_embedfile = tk.StringVar()
        ttk.Label(frm_top, text="Embeddings Cache-Datei:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_embedfile, width=50).grid(row=4, column=1, sticky="we", padx=(6, 6))
        ttk.Button(frm_top, text="Wählen…", command=self._choose_cache).grid(row=4, column=2, sticky="e")

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
        self.var_only_indexes = tk.BooleanVar()

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

        ttk.Checkbutton(frm_params, text="Nur Indizes (kein Upload/Embeddings)", variable=self.var_only_indexes).grid(
            row=4, column=2, columnspan=2, sticky="w", padx=(6, 6), pady=(2, 10)
        )

        # Aktionen
        frm_actions = ttk.Frame(self.root)
        frm_actions.pack(side=tk.TOP, fill=tk.X, padx=pad_x, pady=(0, pad_y))
        ttk.Button(frm_actions, text="Start", command=self._start).pack(side=tk.LEFT)
        ttk.Button(frm_actions, text="Cache leeren", command=self._clear_cache).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(frm_actions, text="Abbrechen", command=self._cancel).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(frm_actions, text="Suche…", command=self._open_search).pack(side=tk.RIGHT)

        # Log
        frm_log = ttk.LabelFrame(self.root, text="Log")
        frm_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=pad_x, pady=pad_y)
        self.txt_log = scrolledtext.ScrolledText(frm_log, wrap=tk.WORD, height=18)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Markdown-Ordner wählen")
        if path:
            self.var_md_folder.set(path)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Markdown-Datei wählen",
            filetypes=[("Markdown", "*.md"), ("Alle Dateien", "*.*")],
        )
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

    def _load_collections(self):
        """Lädt Collections aus Qdrant und füllt die Combobox."""
        url = os.environ.get("QDRANT_CLOUD_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url:
            self._append_log("QDRANT_CLOUD_URL ist nicht gesetzt – Collections können nicht geladen werden.")
            return
        try:
            client = QdrantClient(url=url, api_key=api_key, timeout=30)
            resp = client.get_collections()
            names = []
            # qdrant-client v1.x: resp.collections
            if hasattr(resp, "collections"):
                names = [c.name for c in resp.collections]
            else:
                # Fallback: resp.result.collections
                res = getattr(resp, "result", None) or {}
                colls = res.get("collections", [])
                names = [c.get("name") for c in colls if isinstance(c, dict)]
            self.collections_cache = names
            if hasattr(self, "cmb_collection"):
                self.cmb_collection["values"] = names
            self._append_log(f"Collections geladen: {', '.join(names) if names else '(keine)'}")
        except Exception as e:
            self._append_log(f"Fehler beim Laden der Collections: {e}")

    def _start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Info", "Ein Lauf ist bereits aktiv.")
            return
        md_folder = self.var_md_folder.get().strip()
        if not md_folder:
            messagebox.showerror("Fehler", "Bitte einen gültigen Markdown-Pfad wählen.")
            return
        if not os.path.isdir(md_folder) and not os.path.isfile(md_folder):
            messagebox.showerror("Fehler", "Bitte einen gültigen Markdown-Ordner oder eine gültige Markdown-Datei wählen.")
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
            create_indexes=bool(self.var_create_indexes.get() or self.var_only_indexes.get()),
            create_collection_if_missing=bool(self.var_create_collection.get()),
            vector_size=int(self.var_vector_size.get() or 1536),
            dedupe_mode=self.var_dedupe_mode.get(),
            only_indexes=bool(self.var_only_indexes.get()),
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

    def _open_search(self):
        # Reiche Logger-Funktion in das Suchfenster durch
        SearchWindow(self.root, logger=self._append_log)

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
            "only_indexes": bool(self.var_only_indexes.get()),
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
        if "only_indexes" in data:
            self.var_only_indexes.set(bool(data["only_indexes"]))


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


class SearchWindow:
    def __init__(self, master, logger=None):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title("Suche in Qdrant")
        self.top.geometry("980x680")

        self.client = None
        self.emb = None
        self.logger = logger

        # UI Vars
        self.var_collection = tk.StringVar(value=os.environ.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME))
        self.var_query = tk.StringVar()
        self.var_tags = tk.StringVar()
        self.var_note_type = tk.StringVar()
        self.var_source = tk.StringVar()
        self.var_date_from = tk.StringVar()
        self.var_date_to = tk.StringVar()
        self.var_top_k = tk.IntVar(value=10)

        # Build UI
        self._build_ui()

        # HTML renderer optional
        try:
            from tkhtmlview import HTMLLabel  # noqa
            self._html_supported = True
        except Exception:
            self._html_supported = False

    def _build_ui(self):
        frm_top = ttk.Frame(self.top)
        frm_top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(frm_top, text="Collection:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_collection, width=40).grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(frm_top, text="Top-K:").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_top_k, width=8).grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(frm_top, text="Query (Text, optional):").grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Entry(frm_top, textvariable=self.var_query, width=80).grid(row=1, column=1, columnspan=3, sticky="we", padx=(6, 6))
        frm_top.grid_columnconfigure(1, weight=1)

        # Filter
        frm_filters = ttk.LabelFrame(self.top, text="Filter (optional - mindestens Query oder Filter nötig)")
        frm_filters.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(frm_filters, text="Tags (Kommagetrennt):").grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
        ttk.Entry(frm_filters, textvariable=self.var_tags, width=40).grid(row=0, column=1, sticky="w", pady=(6, 2))

        ttk.Label(frm_filters, text="Note Type:").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=(6, 2))
        ttk.Entry(frm_filters, textvariable=self.var_note_type, width=20).grid(row=0, column=3, sticky="w", pady=(6, 2))

        ttk.Label(frm_filters, text="Source:").grid(row=1, column=0, sticky="w", padx=(6, 6), pady=(2, 2))
        ttk.Entry(frm_filters, textvariable=self.var_source, width=40).grid(row=1, column=1, sticky="w", pady=(2, 2))

        ttk.Label(frm_filters, text="Date From:").grid(row=2, column=0, sticky="w", padx=(6, 6), pady=(2, 2))
        entry_date_from = ttk.Entry(frm_filters, textvariable=self.var_date_from, width=30)
        entry_date_from.grid(row=2, column=1, sticky="w", pady=(2, 2))
        entry_date_from.insert(0, "z.B. 2025-11-01 oder 20251101")
        entry_date_from.config(foreground="gray")
        entry_date_from.bind("<FocusIn>", lambda e: self._on_date_entry_focus_in(entry_date_from, self.var_date_from))
        entry_date_from.bind("<FocusOut>", lambda e: self._on_date_entry_focus_out(entry_date_from, self.var_date_from, "z.B. 2025-11-01 oder 20251101"))

        ttk.Label(frm_filters, text="Date To:").grid(row=3, column=0, sticky="w", padx=(6, 6), pady=(2, 6))
        entry_date_to = ttk.Entry(frm_filters, textvariable=self.var_date_to, width=30)
        entry_date_to.grid(row=3, column=1, sticky="w", pady=(2, 6))
        entry_date_to.insert(0, "z.B. 2025-11-01 oder 20251101")
        entry_date_to.config(foreground="gray")
        entry_date_to.bind("<FocusIn>", lambda e: self._on_date_entry_focus_in(entry_date_to, self.var_date_to))
        entry_date_to.bind("<FocusOut>", lambda e: self._on_date_entry_focus_out(entry_date_to, self.var_date_to, "z.B. 2025-11-01 oder 20251101"))

        ttk.Button(frm_filters, text="Suchen", command=self._search).grid(row=4, column=0, sticky="w", padx=(6, 6), pady=(4, 8))

        # Results area: left list, right preview
        frm_main = ttk.Frame(self.top)
        frm_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.listbox = tk.Listbox(frm_main, width=40)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        sep = ttk.Separator(frm_main, orient="vertical")
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.preview = scrolledtext.ScrolledText(frm_main, wrap=tk.WORD)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Storage for results
        self._hits = []

    def _on_date_entry_focus_in(self, entry, var):
        """Entfernt Placeholder-Text beim Fokussieren"""
        current = entry.get()
        placeholder = "z.B. 2025-11-01 oder 20251101"
        if current == placeholder:
            entry.delete(0, tk.END)
            entry.config(foreground="black")
            var.set("")

    def _on_date_entry_focus_out(self, entry, var, placeholder):
        """Zeigt Placeholder-Text wenn Feld leer ist"""
        current = entry.get().strip()
        if not current:
            entry.delete(0, tk.END)
            entry.insert(0, placeholder)
            entry.config(foreground="gray")
            var.set("")
        else:
            var.set(current)

    def _ensure_clients(self):
        if self.client is None:
            url = os.environ.get("QDRANT_CLOUD_URL")
            api_key = os.environ.get("QDRANT_API_KEY")
            if not url:
                raise RuntimeError("QDRANT_CLOUD_URL ist nicht gesetzt.")
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        if self.emb is None:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt.")
            self.emb = OpenAIEmbeddings(openai_api_key=openai_key)

    def _normalize_date(self, date_str):
        """Konvertiert verschiedene Datumsformate zu ISO-8601 (YYYY-MM-DDTHH:MM:SSZ)"""
        if not date_str or not date_str.strip():
            return None
        date_str = date_str.strip()
        # Versuche verschiedene Formate
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO mit Zeit
            "%Y-%m-%dT%H:%M:%S",   # ISO ohne Z
            "%Y-%m-%d %H:%M:%S",   # ISO-like
            "%Y-%m-%d",            # Nur Datum
            "%Y%m%d",              # YYYYMMDD
        ]
        from datetime import datetime
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Konvertiere zu ISO-8601 mit Z (UTC)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        # Fallback: Versuche dateutil
        try:
            from dateutil import parser as dateutil_parser
            dt = dateutil_parser.parse(date_str)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    def _build_filter(self):
        must = []
        tags_str = self.var_tags.get().strip()
        if tags_str:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            if tags:
                must.append(models.FieldCondition(key="tag", match=models.MatchAny(any=tags)))
        note_t = self.var_note_type.get().strip()
        if note_t:
            must.append(models.FieldCondition(key="note_type", match=models.MatchValue(value=note_t)))
        src = self.var_source.get().strip()
        if src:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=src)))
        date_from = self.var_date_from.get().strip()
        date_to = self.var_date_to.get().strip()
        # Ignoriere Placeholder-Text
        if date_from == "z.B. 2025-11-01 oder 20251101":
            date_from = ""
        if date_to == "z.B. 2025-11-01 oder 20251101":
            date_to = ""
        if date_from or date_to:
            rng = models.Range()
            if date_from:
                normalized = self._normalize_date(date_from)
                if normalized:
                    rng.gte = normalized
                else:
                    self.logger(f"Warnung: Datum 'From' konnte nicht geparst werden: {date_from}")
            if date_to:
                normalized = self._normalize_date(date_to)
                if normalized:
                    rng.lte = normalized
                else:
                    self.logger(f"Warnung: Datum 'To' konnte nicht geparst werden: {date_to}")
            if rng.gte or rng.lte:
                must.append(models.FieldCondition(key="date_created", range=rng))
        if not must:
            return None
        return models.Filter(must=must)

    def _search(self):
        try:
            self._ensure_clients()
            collection = self.var_collection.get().strip()
            if not collection:
                messagebox.showerror("Fehler", "Collection darf nicht leer sein.")
                return
            query_text = self.var_query.get().strip()
            flt = self._build_filter()
            top_k = max(1, int(self.var_top_k.get()))
            
            # Wenn weder Query noch Filter: Fehler
            if not query_text and not flt:
                messagebox.showerror("Fehler", "Bitte mindestens eine Query oder einen Filter angeben.")
                return
            
            # Fall 1: Nur Filter, keine Query -> scroll() verwenden
            if not query_text and flt:
                try:
                    scrolled = self.client.scroll(
                        collection_name=collection,
                        scroll_filter=flt,
                        with_payload=True,
                        with_vectors=False,
                        limit=top_k
                    )
                    # scroll returns (points, next_page)
                    if scrolled and scrolled[0]:
                        self._hits = scrolled[0]
                    else:
                        self._hits = []
                    if self.logger:
                        self.logger(f"Filter-Suche abgeschlossen: {len(self._hits)} Treffer (Top-K={top_k}, Collection={collection}).")
                    self._refresh_list()
                    return
                except Exception as e:
                    messagebox.showerror("Fehler", f"Filter-Suche fehlgeschlagen: {e}")
                    return
            
            # Fall 2: Query vorhanden -> Vektor-Suche (mit optionalem Filter)
            vector = self.emb.embed_query(query_text)
            # Kompatibel: Versuche neue Query API, falle auf ältere Search API zurück
            res = None
            try:
                res = self.client.query_points(
                    collection_name=collection,
                    query=vector,
                    filter=flt,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                # Ältere Clients erwarten query_filter und search()-Methode
                res = self.client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                    with_vectors=False,
                )
            # Ergebnisse je nach Client-Version extrahieren
            if hasattr(res, "points"):
                self._hits = res.points
            elif isinstance(res, list):
                self._hits = res
            else:
                self._hits = getattr(res, "result", [])
            # Fallback: Wenn 0 Treffer, versuche Full-Text auf 'text' (erfordert text-Index)
            if not self._hits:
                try:
                    ft_filter = self._build_filter()
                    # Ergänze MatchText auf Feld 'text'
                    conds = (ft_filter.must[:] if ft_filter else [])
                    conds.append(models.FieldCondition(key="text", match=models.MatchText(text=query_text)))
                    ft = models.Filter(must=conds)
                    # Scroll als Volltext-Suche
                    scrolled = self.client.scroll(
                        collection_name=collection,
                        scroll_filter=ft,
                        with_payload=True,
                        with_vectors=False,
                        limit=top_k
                    )
                    # scroll returns (points, next_page)
                    if scrolled and scrolled[0]:
                        self._hits = scrolled[0]
                        if self.logger:
                            self.logger(f"Vector-Suche 0 Treffer; Full-Text-Fallback ergab {len(self._hits)} Treffer.")
                except Exception as e:
                    if self.logger:
                        self.logger(f"Full-Text-Fallback fehlgeschlagen: {e}")
            # Logge Trefferanzahl ins Haupt-Log
            if self.logger:
                try:
                    self.logger(f"Suche abgeschlossen: {len(self._hits)} Treffer (Top-K={top_k}, Collection={collection}).")
                except Exception:
                    pass
            self._refresh_list()
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            if self.logger:
                try:
                    self.logger(f"Suche fehlgeschlagen: {e}")
                except Exception:
                    pass

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, h in enumerate(self._hits):
            fn = ""
            try:
                fn = h.payload.get("file_name", "")
            except Exception:
                pass
            score = getattr(h, "score", None)
            label = f"{i+1}. {fn}  (score={score:.4f})" if score is not None else f"{i+1}. {fn}"
            self.listbox.insert(tk.END, label)
        self.preview.delete("1.0", tk.END)

    def _on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        hit = self._hits[idx]
        payload = getattr(hit, "payload", {}) or {}
        text = payload.get("text", "")
        # Render markdown (fallback plaintext)
        rendered = None
        try:
            import markdown  # noqa
            html = markdown.markdown(text)
            if self._html_supported:
                # Replace preview with HTMLLabel
                for child in self.preview.master.pack_slaves():
                    pass
                self.preview.delete("1.0", tk.END)
                self.preview.insert(tk.END, self._strip_md(text))
                # For simplicity, keep text widget, but we rendered above if needed.
            else:
                rendered = self._strip_md(text)
        except Exception:
            rendered = self._strip_md(text)
        if rendered is None:
            # if HTML support was not fully integrated, show plain text fallback
            rendered = self._strip_md(text)
        self.preview.delete("1.0", tk.END)
        self.preview.insert(tk.END, rendered)

    def _strip_md(self, s: str) -> str:
        # Minimal Markdown->Text Fallback (korrekte Regex-Escapes)
        s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)              # Überschriften
        s = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", s)                      # Inline-Code
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)                          # Bold
        s = re.sub(r"\*([^*]+)\*", r"\1", s)                              # Italic
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)                    # Links [text](url)
        return s


if __name__ == "__main__":
    main()


