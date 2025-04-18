#!/usr/bin/env python3
"""
Improved script to clean and fix keywords in the database:
- Splits long/concatenated keywords
- Removes generic/boilerplate keywords unless highly relevant
- Deduplicates and limits to 5 meaningful keywords
- Handles both JSON and comma-separated formats
- Updates the database in-place
"""
import sqlite3
import json
import re
import sys
import os

# List of generic keywords to deprioritize
GENERIC_KEYWORDS = set([
    'youth', 'support', 'advocacy', 'legal', 'services', 'document', 'information', 'resource',
    'report', 'centre', 'center', 'annual', 'family', 'queensland', 'brisbane', 'system', 'person',
    'community', 'partnership', 'justice', 'child', 'young', 'people', 'sheet', 'form', 'pdf', 'webpage'
])

# Helper to split long/concatenated keywords
def split_keyword(keyword):
    # Try to split at camelCase
    camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', keyword)
    # Try to split at common word boundaries
    common_words = [
        'youth', 'young', 'people', 'support', 'services', 'legal', 'advocacy',
        'centre', 'center', 'annual', 'report', 'family', 'aboriginal', 'torres',
        'justice', 'system', 'person', 'community', 'queensland', 'brisbane',
        'partnership', 'jessica', 'sarah', 'tim', 'child', 'sheet', 'form', 'pdf', 'webpage'
    ]
    temp_keyword = camel_split
    for word in common_words:
        temp_keyword = re.sub(f'({word})', r' \1', temp_keyword, flags=re.IGNORECASE)
    temp_keyword = re.sub(r'\s+', ' ', temp_keyword).strip()
    # Extract words of 3+ chars
    words = [w for w in temp_keyword.split() if len(w) >= 3]
    return words if words else [keyword]

def clean_keywords(raw_keywords):
    # Parse JSON or comma-separated
    if not raw_keywords:
        return []
    try:
        if raw_keywords.strip().startswith('[') and raw_keywords.strip().endswith(']'):
            keywords = json.loads(raw_keywords)
            if not isinstance(keywords, list):
                keywords = []
        else:
            keywords = [k.strip() for k in raw_keywords.split(',') if k.strip()]
    except Exception:
        keywords = [k.strip() for k in raw_keywords.split(',') if k.strip()]
    # Flatten, split long/concatenated
    flat_keywords = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw = kw.strip()
        if len(kw) > 20 and ' ' not in kw:
            flat_keywords.extend(split_keyword(kw))
        else:
            flat_keywords.append(kw)
    # Remove duplicates, preserve order
    seen = set()
    unique_keywords = [k for k in flat_keywords if k.lower() not in seen and not seen.add(k.lower())]
    # Remove generic keywords unless they're the only ones
    filtered = [k for k in unique_keywords if k.lower() not in GENERIC_KEYWORDS]
    if not filtered:
        filtered = unique_keywords[:]
    # Limit to 5
    return filtered[:5]

def fix_database_keywords(db_path):
    print(f"Fixing keywords in {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT rowid, keywords FROM documents WHERE keywords IS NOT NULL")
    records = cursor.fetchall()
    print(f"Found {len(records)} records with keywords")
    updates = 0
    for rowid, raw_keywords in records:
        cleaned = clean_keywords(raw_keywords)
        new_keywords_json = json.dumps(cleaned)
        if new_keywords_json != raw_keywords:
            cursor.execute("UPDATE documents SET keywords = ? WHERE rowid = ?", (new_keywords_json, rowid))
            updates += 1
            print(f"  Updated row {rowid}: {cleaned}")
    conn.commit()
    print(f"Successfully updated {updates} records with improved keywords")
    conn.close()
    return updates > 0

if __name__ == "__main__":
    db_path = "yac_docs.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found")
        sys.exit(1)
    if fix_database_keywords(db_path):
        print("Keyword cleanup completed successfully")
        sys.exit(0)
    else:
        print("Keyword cleanup failed or no updates were needed")
        sys.exit(1) 