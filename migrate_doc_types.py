#!/usr/bin/env python3
"""
Script to update the database with document types.
This adds a document_type field and sets appropriate values for existing records.
"""

import sqlite3
import os
import sys

def migrate_document_types(db_path):
    """Add document_type column and populate it based on file extensions"""
    print(f'Updating document types in {db_path}...')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First check if document_type column exists, add it if not
        cursor.execute('PRAGMA table_info(documents)')
        columns = [col[1] for col in cursor.fetchall()]
        if 'document_type' not in columns:
            print('Adding document_type column...')
            cursor.execute('ALTER TABLE documents ADD COLUMN document_type TEXT DEFAULT "pdf"')
            conn.commit()
            print('Column added successfully')
        else:
            print('document_type column already exists')

        # Update all records based on filename extension
        cursor.execute('SELECT rowid, filename FROM documents')
        records = cursor.fetchall()
        updates = 0

        for rowid, filename in records:
            if filename:
                ext = os.path.splitext(filename)[1].lower()
                doc_type = 'pdf' if ext in ('.pdf', '.doc', '.docx') else 'webpage'
                cursor.execute('UPDATE documents SET document_type = ? WHERE rowid = ?', (doc_type, rowid))
                updates += 1

        conn.commit()
        print(f'Updated document types for {updates} records')
        conn.close()
        return True
    except Exception as e:
        print(f'Error during migration: {e}')
        return False

if __name__ == '__main__':
    db_path = 'yac_docs.db'
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    if migrate_document_types(db_path):
        print('Migration completed successfully')
        sys.exit(0)
    else:
        print('Migration failed')
        sys.exit(1) 