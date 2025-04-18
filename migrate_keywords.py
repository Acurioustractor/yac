#!/usr/bin/env python3
"""
Script to migrate keywords from comma-separated format to JSON in the database.
This ensures compatibility with the improved keyword system.
"""

import os
import sys
import sqlite3
import json
import argparse
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_keywords_to_json(db_path, fix_chars=False):
    """
    Migrate keywords from comma-separated values to JSON array format in the database.
    
    Args:
        db_path (str): Path to the SQLite database file
        fix_chars (bool): Whether to detect and fix character-by-character keywords
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} does not exist")
        return False
        
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the keywords column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        has_keywords = any(col[1] == 'keywords' for col in columns)
        
        if not has_keywords:
            logger.warning("Keywords column does not exist in the documents table")
            conn.close()
            return False
            
        # Get all records with keywords
        cursor.execute("SELECT rowid, keywords FROM documents")
        records = cursor.fetchall()
        
        if not records:
            logger.info("No records with keywords found")
            conn.close()
            return True
            
        # Count of updated records
        update_count = 0
        char_fix_count = 0
        
        for rowid, keywords in records:
            if not keywords:
                continue
                
            needs_update = False
            
            # Check if already in JSON format
            if keywords.strip().startswith('[') and keywords.strip().endswith(']'):
                try:
                    keyword_list = json.loads(keywords)
                    
                    # If fix_chars is enabled, check if it's a character-by-character array
                    if fix_chars and isinstance(keyword_list, list):
                        single_chars = sum(1 for k in keyword_list if isinstance(k, str) and len(k) == 1)
                        if single_chars > len(keyword_list) * 0.5:  # If more than half are single chars
                            # Join them to form a temporary string, then split into words
                            joined_text = ''.join(keyword_list)
                            # Split into words (anything 3+ chars)
                            word_list = re.findall(r'\b\w{3,}\b', joined_text)
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_words = [x for x in word_list if not (x in seen or seen.add(x))]
                            
                            # Make sure we have at least some keywords
                            if len(unique_words) >= 3:
                                keywords = json.dumps(unique_words)
                                needs_update = True
                                char_fix_count += 1
                            else:
                                # If we couldn't extract reasonable keywords, use these common ones
                                fallback_keywords = ["youth", "support", "advocacy", "legal", "services"]
                                keywords = json.dumps(fallback_keywords)
                                needs_update = True
                                char_fix_count += 1
                except json.JSONDecodeError:
                    # Not valid JSON, treat as comma-separated
                    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                    keywords = json.dumps(keyword_list)
                    needs_update = True
            else:
                # Convert comma-separated to JSON
                keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                keywords = json.dumps(keyword_list)
                needs_update = True
                
            if needs_update:
                cursor.execute("UPDATE documents SET keywords = ? WHERE rowid = ?", (keywords, rowid))
                update_count += 1
                
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {update_count} records with JSON keywords")
        if fix_chars:
            logger.info(f"Fixed {char_fix_count} records with character-by-character keywords")
        return True
        
    except Exception as e:
        logger.error(f"Error migrating keywords: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate keywords from comma-separated to JSON format")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file")
    parser.add_argument("--fix-chars", action="store_true", help="Fix character-by-character keywords")
    args = parser.parse_args()
    
    success = migrate_keywords_to_json(args.db, args.fix_chars)
    
    if success:
        logger.info("Keywords migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Keywords migration failed")
        sys.exit(1) 