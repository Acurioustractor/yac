from flask import Flask, jsonify, send_from_directory, render_template, request, Response, url_for
import sqlite3
import os
import argparse
import json
import logging
import datetime
import hashlib
import time
import re

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_NAME = "yac_docs.db"
DOWNLOAD_DIR = "downloaded_docs"

# Track app version/timestamp for cache busting
APP_VERSION = int(time.time())

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

@app.context_processor
def inject_version():
    """Inject version info into all templates for cache busting"""
    return {'app_version': APP_VERSION}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/api/test')
def test_api():
    """Simple endpoint to test if the API is working"""
    logger.info("API test endpoint called")
    return jsonify({
        "status": "ok",
        "message": "API is working correctly",
        "timestamp": str(datetime.datetime.now())
    })

def fix_character_keywords(keywords_str):
    """Fix issues with keywords that might be character-by-character or overly concatenated."""
    if not keywords_str:
        return []
        
    try:
        # Check if it's a valid JSON array
        if keywords_str.strip().startswith('[') and keywords_str.strip().endswith(']'):
            keywords_list = json.loads(keywords_str)
            
            # If not a list or empty, use default keywords
            if not isinstance(keywords_list, list) or not keywords_list:
                return ["document", "information", "resource"]
                
            # Process each keyword for potential fixes
            processed_keywords = []
            
            for keyword in keywords_list:
                if not isinstance(keyword, str) or not keyword.strip():
                    continue
                    
                keyword = keyword.strip().lower()
                
                # Check if it's mostly single characters (like ["y", "o", "u", "t", "h"])
                if len([c for c in keyword if len(c.strip()) == 1]) > len(keyword) * 0.6:
                    # Likely character-by-character, join and process as a single keyword
                    joined = ''.join(keyword).lower()
                    processed_keywords.append(joined)
                    continue
                
                # Handle very long concatenated keywords (no spaces but multiple words stuck together)
                if len(keyword) > 15 and ' ' not in keyword:
                    # Try to split at common word boundaries for known terms in our domain
                    common_words = [
                        'youth', 'young', 'people', 'support', 'services', 'legal', 'advocacy',
                        'centre', 'center', 'annual', 'report', 'family', 'aboriginal', 'torres', 
                        'justice', 'system', 'person', 'community', 'queensland', 'brisbane', 
                        'partnership', 'jessica', 'sarah', 'tim'
                    ]
                    
                    # Temporary string for splitting
                    temp_keyword = keyword
                    for word in common_words:
                        # Insert space before each occurrence of the word (case insensitive)
                        temp_keyword = re.sub(f'({word})', r' \1', temp_keyword, flags=re.IGNORECASE)
                    
                    # Remove leading/trailing spaces and clean up multiple spaces
                    temp_keyword = re.sub(r'\s+', ' ', temp_keyword).strip()
                    
                    if ' ' in temp_keyword:
                        # Successfully split, extract meaningful words (3+ chars)
                        words = [w for w in temp_keyword.split() if len(w) >= 3]
                        processed_keywords.extend(words[:3])  # Limit to 3 words from a single concatenated keyword
                    else:
                        # If still no spaces, try splitting camelCase
                        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', keyword)
                        words = [w.lower() for w in camel_split.split() if len(w) >= 3]
                        if len(words) > 1:
                            processed_keywords.extend(words[:3])
                        else:
                            # Last resort: split into chunks of reasonable size
                            if len(keyword) > 30:  # Very long single "word"
                                chunks = [keyword[i:i+8] for i in range(0, len(keyword), 8)]
                                processed_keywords.extend(chunks[:3])
                            else:
                                processed_keywords.append(keyword)
                else:
                    processed_keywords.append(keyword)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = [k for k in processed_keywords if not (k in seen or seen.add(k))]
            
            # Ensure we have at least 3 keywords
            if len(unique_keywords) < 3:
                default_words = ["document", "information", "resource"]
                for word in default_words:
                    if word not in unique_keywords:
                        unique_keywords.append(word)
                        if len(unique_keywords) >= 3:
                            break
            
            # Return up to 5 keywords
            return unique_keywords[:5]
            
        else:
            # Not a JSON array, try to extract meaningful keywords
            if len(keywords_str) > 30 and ' ' not in keywords_str:
                # Potentially a single long concatenated string
                return fix_character_keywords(json.dumps([keywords_str]))
            else:
                # Simple string, just split by commas
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                return keywords[:5] if keywords else ["document", "information", "resource"]
    
    except Exception as e:
        logger.error(f"Error fixing keywords: {e}")
        return ["document", "information", "resource"]

@app.route('/api/documents')
def get_documents():
    """Get all documents from the database"""
    try:
        logger.info("Getting documents from the database")
        raw_format = request.args.get('raw', 'false').lower() == 'true'
        doc_type_filter = request.args.get('type', 'all').lower()
        
        # Connect to the database
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if the database exists and has the documents table
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            if not cursor.fetchone():
                logger.error("Documents table does not exist in the database")
                error_message = {"error": "Documents table does not exist in the database"}
                return jsonify(error_message) if not raw_format else json.dumps(error_message), 500
        except Exception as e:
            logger.error(f"Error checking database structure: {str(e)}")
            error_message = {"error": f"Database error: {str(e)}"}
            return jsonify(error_message) if not raw_format else json.dumps(error_message), 500
        
        # Check if document_type column exists, add it if not
        cursor.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'document_type' not in columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN document_type TEXT DEFAULT 'pdf'")
                conn.commit()
                logger.info("Added document_type column to documents table")
            except Exception as e:
                logger.error(f"Error adding document_type column: {str(e)}")
        
        # Get documents with optional type filtering
        if doc_type_filter != 'all':
            cursor.execute("SELECT * FROM documents WHERE document_type = ?", (doc_type_filter,))
        else:
            cursor.execute("SELECT * FROM documents")
        
        documents = cursor.fetchall()
        
        # Convert to list of dictionaries
        result = []
        for doc in documents:
            doc_dict = {key: doc[key] for key in doc.keys()}
            
            # Handle keywords using our improved function
            if 'keywords' in doc_dict and doc_dict['keywords'] and isinstance(doc_dict['keywords'], str):
                # Process the keywords with our enhanced function
                fixed_keywords = fix_character_keywords(doc_dict['keywords'])
                # The frontend expects a JSON string
                doc_dict['keywords'] = json.dumps(fixed_keywords)
            
            # Determine document_type if not set
            if 'document_type' not in doc_dict or not doc_dict['document_type']:
                if doc_dict.get('filename', '').lower().endswith(('.pdf', '.doc', '.docx')):
                    doc_dict['document_type'] = 'pdf'
                else:
                    doc_dict['document_type'] = 'webpage'
                
            result.append(doc_dict)
        
        # Log response summary
        logger.info(f"Returning {len(result)} documents")
        
        # Return the result
        if raw_format:
            # Return raw JSON for more verbose debugging
            return Response(json.dumps(result), mimetype='application/json')
        else:
            return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        error_message = {"error": f"Error getting documents: {str(e)}"}
        return jsonify(error_message) if not raw_format else json.dumps(error_message), 500
    
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a file"""
    logger.info(f"Download requested for file: {filename}")
    try:
        # Ensure the file exists
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<path:filename>')
def get_document_api(filename):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT source_url, local_path, filename, download_date, 
               keywords, summary, full_text, document_date, document_type
            FROM documents 
            WHERE filename = ?
        """, (filename,))
        
        document = cursor.fetchone()
        conn.close()
        
        if document:
            doc_dict = dict(document)
            
            # Handle keywords using our improved function
            if 'keywords' in doc_dict and doc_dict['keywords'] and isinstance(doc_dict['keywords'], str):
                fixed_keywords = fix_character_keywords(doc_dict['keywords'])
                # The frontend expects a JSON string
                doc_dict['keywords'] = json.dumps(fixed_keywords)
            
            return jsonify(doc_dict)
        else:
            return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        app.logger.error(f"Error fetching document details for {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

@app.after_request
def add_cache_control_headers(response):
    """Add headers to prevent browser caching"""
    # Prevent caching for HTML, JavaScript, and CSS responses
    if response.mimetype in ['text/html', 'text/javascript', 'text/css', 'application/json']:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app for YAC document explorer')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Print helpful startup message
    print(f"Starting YAC Document Explorer:")
    print(f"  - Database: {os.path.abspath(DB_NAME)}")
    print(f"  - Download Directory: {os.path.abspath(DOWNLOAD_DIR)}")
    print(f"  - Web Interface: http://{args.host}:{args.port}/")
    print(f"  - API Endpoint: http://{args.host}:{args.port}/api/documents")
    print(f"  - API Test Endpoint: http://{args.host}:{args.port}/api/test")
    
    app.run(host=args.host, port=args.port, debug=args.debug) 