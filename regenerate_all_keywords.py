#!/usr/bin/env python3
"""
Batch regenerate keywords for all documents using the improved generate_keywords function.
"""
import sqlite3
import os
import sys
from scraper import generate_keywords, load_spacy_model

DB_PATH = "yac_docs.db"
if len(sys.argv) > 1:
    DB_PATH = sys.argv[1]

if not os.path.exists(DB_PATH):
    print(f"Error: Database {DB_PATH} not found")
    sys.exit(1)

# Load spaCy model
load_spacy_model()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT rowid, full_text FROM documents WHERE full_text IS NOT NULL")
rows = cursor.fetchall()
print(f"Found {len(rows)} documents with text to process.")
updates = 0
for rowid, full_text in rows:
    if not full_text or len(full_text.strip()) < 20:
        continue
    keywords = generate_keywords(full_text)
    if not keywords:
        continue
    import json
    keywords_json = json.dumps(keywords)
    cursor.execute("UPDATE documents SET keywords = ? WHERE rowid = ?", (keywords_json, rowid))
    updates += 1
    print(f"  Updated row {rowid}: {keywords}")
conn.commit()
print(f"Successfully updated {updates} documents with new keywords.")
conn.close() 