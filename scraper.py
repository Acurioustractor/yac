import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import re
import sqlite3
from datetime import datetime
import fitz  # PyMuPDF
import docx # python-docx
import openai
from openai import OpenAI
import PyPDF2
from dateutil.parser import parse as date_parse # For better date parsing
from dateutil.parser import ParserError
import json
import sys

# NLP Imports for Stage 3
import spacy
import nltk # Added for checking/downloading punkt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
# Note: python-docx reads .docx, not older .doc
TARGET_DOC_EXTENSIONS = {'.pdf', '.doc', '.docx'} 
DOWNLOAD_DIR = "downloaded_docs"
MAX_PAGES = 500 
REQUEST_DELAY = 0.1
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
NUM_SUMMARY_SENTENCES = 3 # Number of sentences for the summary
NUM_KEYWORDS = 15 # Max number of keywords to store
# --- End Configuration ---

# --- Global Variables (Load NLP model once) ---
NLP = None
def load_spacy_model():
    global NLP
    if NLP is None:
        try:
            print("Loading spaCy model (en_core_web_sm)...")
            NLP = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully.")
        except OSError:
            print("Error loading spaCy model 'en_core_web_sm'.")
            print("Please ensure it's downloaded: python -m spacy download en_core_web_sm")
            NLP = False # Indicate failure
    return NLP
# --- End Global Variables ---

# --- NLTK Data Check ---
def ensure_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer already available.")
        try:
            # Additional check for punkt_tab
            nltk.data.find('tokenizers/punkt_tab')
            print("NLTK 'punkt_tab' also available.")
        except LookupError:
            # Try to download punkt_tab if not available
            print("NLTK 'punkt_tab' not found. Attempting download...")
            try:
                nltk.download('punkt') # This should include punkt_tab
                print("NLTK 'punkt' downloaded with dependencies.")
                return True
            except Exception as e:
                print(f"\033[91mError downloading NLTK resources: {e}\033[0m")
                print("Will use fallback summarization.")
        return True
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Attempting download...")
        try:
            nltk.download('punkt')
            print("NLTK 'punkt' downloaded successfully.")
            return True
        except Exception as e:
            print(f"\033[91mError downloading NLTK 'punkt': {e}\033[0m")
            print("Will use fallback summarization.")
            return False
# --- End NLTK Data Check ---

def setup_database(db_name_arg):
    """Connects to the SQLite database and creates/updates the table."""
    conn = sqlite3.connect(db_name_arg)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        source_url TEXT PRIMARY KEY,
        local_path TEXT,
        filename TEXT,
        download_date TEXT,
        last_checked_date TEXT,
        keywords TEXT,         -- Added Stage 3
        summary TEXT,          -- Added Stage 3
        full_text TEXT,
        document_date TEXT
    )
    ''')
    # Add columns if they don't exist (for upgrades)
    columns_to_add = {'full_text': 'TEXT', 'keywords': 'TEXT', 'summary': 'TEXT', 'document_date': 'TEXT'}
    cursor.execute("PRAGMA table_info(documents)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    for col, col_type in columns_to_add.items():
        if col not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE documents ADD COLUMN {col} {col_type}")
                print(f"Added '{col}' column to database.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    print(f"Warning: Could not add column '{col}'. Error: {e}")
                # else: column already exists, ignore
            
    conn.commit()
    return conn, cursor

def add_or_update_document_record(cursor, conn, url, local_path, filename, timestamp, full_text, keywords, summary, document_date):
    """Adds/updates document record including keywords and summary."""
    keywords_str = None
    if keywords and isinstance(keywords, list):
        try:
            # Store as JSON for better handling of special characters and consistent parsing
            keywords_str = json.dumps(keywords)
        except Exception as e:
            print(f"      Warning: Error serializing keywords to JSON: {e}")
            # Fallback to comma-separated format if JSON fails
            keywords_str = ", ".join(keywords)
    elif keywords and isinstance(keywords, str): 
        # If already a string, try to parse and re-serialize to ensure JSON format
        try:
            # Check if it looks like a JSON array
            if keywords.strip().startswith('[') and keywords.strip().endswith(']'):
                parsed = json.loads(keywords)
                if isinstance(parsed, list):
                    keywords_str = keywords  # Already valid JSON
                else:
                    keywords_str = json.dumps([keywords])  # Wrap in array
            else:
                # Assume comma-separated and convert to JSON
                kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
                keywords_str = json.dumps(kw_list)
        except json.JSONDecodeError:
            # If parsing fails, treat as a single keyword or comma-separated list
            if ',' in keywords:
                kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
                keywords_str = json.dumps(kw_list)
            else:
                keywords_str = json.dumps([keywords])
    
    cursor.execute("""
    INSERT INTO documents (source_url, local_path, filename, download_date, last_checked_date, full_text, keywords, summary, document_date) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) 
    ON CONFLICT(source_url) DO UPDATE SET 
        last_checked_date = excluded.last_checked_date,
        full_text = excluded.full_text,
        keywords = excluded.keywords,     -- Always update keywords
        summary = excluded.summary,       -- Always update summary
        local_path = CASE WHEN excluded.local_path IS NOT NULL THEN excluded.local_path ELSE local_path END,
        filename = CASE WHEN excluded.filename IS NOT NULL THEN excluded.filename ELSE filename END,
        document_date = CASE WHEN excluded.document_date IS NOT NULL THEN excluded.document_date ELSE document_date END
        -- Ensure we explicitly update keywords and summary even if other fields are kept
        -- Note: excluded.full_text is used, assuming we always want the latest text if re-processed.
    """, (url, local_path, filename, timestamp, timestamp, full_text, keywords_str, summary, document_date))
    conn.commit()

def extract_text(file_path):
    """Extracts text content and metadata from PDF or DOCX files, with improved date extraction."""
    if not file_path or not os.path.exists(file_path):
        return None, None

    text = ""
    metadata_date = None
    extracted_date_str = None # Store the final extracted date string
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    print(f"    Extracting text and date from: {os.path.basename(file_path)}")

    try:
        if extension == '.pdf':
            try:
                # Try PyMuPDF first for better metadata access
                doc = fitz.open(file_path)
                metadata = doc.metadata
                metadata_date = metadata.get('creationDate') or metadata.get('modDate')
                if metadata_date and metadata_date.startswith('D:'):
                    # Convert PDF Date (D:YYYYMMDDHHMMSS...) to YYYY-MM-DD
                    pdf_date_str = metadata_date[2:10] # YYYYMMDD
                    if len(pdf_date_str) == 8 and pdf_date_str.isdigit():
                         extracted_date_str = f"{pdf_date_str[:4]}-{pdf_date_str[4:6]}-{pdf_date_str[6:8]}"
                         print(f"      Found date in PDF metadata: {extracted_date_str}")

                for page in doc:
                    text += page.get_text() or ""
                doc.close()
            except Exception as fitz_error:
                print(f"      PyMuPDF error: {fitz_error}. Falling back to PyPDF2.")
                # Fallback to PyPDF2 if PyMuPDF fails
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata = reader.metadata
                metadata_date = metadata.get('/CreationDate') or metadata.get('/ModDate')
                if metadata_date and metadata_date.startswith('D:'):
                    pdf_date_str = metadata_date[2:10]
                    if len(pdf_date_str) == 8 and pdf_date_str.isdigit():
                        # Check if we already found a date, PyPDF2 is fallback
                        if not extracted_date_str:
                            extracted_date_str = f"{pdf_date_str[:4]}-{pdf_date_str[4:6]}-{pdf_date_str[6:8]}"
                            print(f"      Found date in PDF metadata (PyPDF2): {extracted_date_str}")
                                
                for page in reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                        except Exception as page_extract_error:
                             print(f"      Warning: Error extracting text from a PDF page: {page_extract_error}")

        elif extension == '.docx':
            doc = docx.Document(file_path)
            # Check document properties for dates
            try:
                core_props = doc.core_properties
                if core_props.modified and isinstance(core_props.modified, datetime):
                     extracted_date_str = core_props.modified.strftime('%Y-%m-%d')
                     print(f"      Found date in DOCX properties (modified): {extracted_date_str}")
                elif core_props.created and isinstance(core_props.created, datetime):
                    if not extracted_date_str: # Prioritize modified date
                         extracted_date_str = core_props.created.strftime('%Y-%m-%d')
                         print(f"      Found date in DOCX properties (created): {extracted_date_str}")
            except Exception as prop_error:
                print(f"      Warning: Could not read DOCX core properties: {prop_error}")
                
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif extension == '.doc':
            print(f"    \033[93mWarning: Skipping text extraction for unsupported .doc file: {os.path.basename(file_path)}\033[0m")
            return None, None
        else:
            return None, None

        cleaned_text = ' '.join(text.split())
        if cleaned_text:
            print(f"    Text extracted (~{len(cleaned_text)} chars)")
        else:
             print(f"    Warning: No text could be extracted from {os.path.basename(file_path)}")
             # Return None, None if no text, as date extraction from text is impossible
             return None, extracted_date_str # Return metadata date if found, even with no text

        # --- Improved Date Extraction from Text (if not found in metadata) ---
        if not extracted_date_str and cleaned_text:
            print("      Searching for date within extracted text...")
            # Look for dates near keywords first (e.g., Updated, Effective, Published, Revised)
            keywords = ['updated', 'effective', 'published', 'revised', 'current as at', 'date']
            window = 150 # Characters around keyword to check
            best_date_from_text = None

            for keyword in keywords:
                # Case-insensitive search for keyword
                for match in re.finditer(keyword, cleaned_text, re.IGNORECASE):
                    start = max(0, match.start() - window)
                    end = min(len(cleaned_text), match.end() + window)
                    nearby_text = cleaned_text[start:end]
                    
                    # Regex for various date formats within this window
                    # (Month YYYY, Mon YYYY, DD Month YYYY, YYYY-MM-DD, DD/MM/YYYY, YYYY/YY, etc.)
                    # Prioritize fuller dates
                    patterns = [
                        r'(\d{1,2})[\s\-/]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-/]+(\d{4})\b', # DD Mon YYYY
                        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-/]+(\d{1,2})[\s,]+(\d{4})\b', # Mon DD, YYYY
                        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-/]+(\d{4})\b', # Mon YYYY
                        r'(\d{4})[\s\-/]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*', # YYYY Mon
                        r'\b(\d{4})[\s\-/]*(\d{1,2})[\s\-/]*(\d{1,2})\b', # YYYY-MM-DD (basic)
                        r'\b(\d{1,2})[\s\-/]+(\d{1,2})[\s\-/]+(\d{4})\b', # DD/MM/YYYY
                        r'\b(\d{4})/(\d{2})\b' # YYYY/YY (e.g., financial year)
                    ]
                    
                    date_found_near_keyword = False
                    for pattern in patterns:
                        for date_match in re.finditer(pattern, nearby_text, re.IGNORECASE):
                            date_str = date_match.group(0)
                            try:
                                # Use dateutil.parser for robust parsing
                                parsed_dt = date_parse(date_str, dayfirst=True, yearfirst=False) # Assume day first common outside US
                                formatted_date = parsed_dt.strftime('%Y-%m-%d')
                                best_date_from_text = formatted_date # Prioritize dates near keywords
                                print(f"        Found date '{formatted_date}' near keyword '{keyword}'")
                                date_found_near_keyword = True
                                break # Stop checking patterns for this keyword match
                            except (ParserError, ValueError) as parse_error:
                                # Handle specific case like '2023/24'
                                if len(date_match.groups()) == 2 and pattern.endswith('YY\\b'):
                                     year1 = int(date_match.group(1))
                                     year2 = int(date_match.group(2))
                                     # Heuristic: take the later year of the financial year
                                     if year2 == (year1 + 1) % 100:
                                         formatted_date = str(year1 + 1) + "-06-30" # Assume end of financial year
                                         best_date_from_text = formatted_date
                                         print(f"        Interpreted financial year '{date_str}' as date '{formatted_date}' near keyword '{keyword}'")
                                         date_found_near_keyword = True
                                         break
                                else:
                                     print(f"        Could not parse potential date '{date_str}' near keyword '{keyword}': {parse_error}")
                                continue # Try next match
                        if date_found_near_keyword: break # Stop checking patterns if date found for this keyword instance
                if best_date_from_text: break # Stop checking other keywords if a date was found near one

            # If no date found near keywords, search the whole document (less reliable)
            if not best_date_from_text:
                 print("      No date found near keywords, searching entire text...")
                 for pattern in patterns: # Use same patterns as above
                     for date_match in re.finditer(pattern, cleaned_text, re.IGNORECASE):
                        date_str = date_match.group(0)
                        try:
                            parsed_dt = date_parse(date_str, dayfirst=True, yearfirst=False)
                            formatted_date = parsed_dt.strftime('%Y-%m-%d')
                            best_date_from_text = formatted_date # Take the first plausible one found
                            print(f"        Found potential date '{formatted_date}' in text (no keyword context)")
                            break # Stop checking patterns
                        except (ParserError, ValueError) as parse_error:
                             if len(date_match.groups()) == 2 and pattern.endswith('YY\\b'):
                                year1 = int(date_match.group(1))
                                year2 = int(date_match.group(2))
                                if year2 == (year1 + 1) % 100:
                                     formatted_date = str(year1 + 1) + "-06-30"
                                     best_date_from_text = formatted_date
                                     print(f"        Interpreted financial year '{date_str}' as date '{formatted_date}' (no keyword context)")
                                     break
                             else:
                                print(f"        Could not parse potential date '{date_str}': {parse_error}")
                             continue
                     if best_date_from_text: break # Stop checking other patterns
            
            extracted_date_str = best_date_from_text # Use date found in text

        # Final check and return
        if extracted_date_str:
             print(f"    ==> Final extracted date: {extracted_date_str}")
        else:
            print(f"    ==> No reliable date could be extracted.")
            
        return cleaned_text, extracted_date_str

    except Exception as e:
        print(f"    \033[91mError during text/date extraction for {os.path.basename(file_path)}: {e}\033[0m")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None # Return None for text and date on major error

def generate_summary(text):
    """Generates a summary using Sumy LSA or fallback to a simpler method if NLTK fails."""
    if not text:
        return None
    
    # Ensure NLTK data is available before proceeding
    if not ensure_nltk_punkt():
        print("      Skipping NLTK summary generation due to missing NLTK data.")
        return fallback_summary(text)
         
    try:
        print("      Generating summary...")
        try:
            # Try using Sumy LSA summarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary_sentences = summarizer(parser.document, NUM_SUMMARY_SENTENCES)
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            print(f"        Summary generated ({len(summary)} chars).")
            return summary
        except LookupError as e:
            # If punkt_tab or other NLTK resource is missing, use fallback
            print(f"      NLTK resource error: {e}")
            print("      Using fallback summarization method...")
            return fallback_summary(text)
    except Exception as e:
        print(f"      \033[91mError generating summary: {e}\033[0m")
        return fallback_summary(text)

def fallback_summary(text):
    """Simple fallback summary method that extracts first few sentences."""
    try:
        # Split text by sentence-ending punctuation and whitespace
        sentences = []
        for s in re.split(r'(?<=[.!?])\s+', text):
            if s and len(s.strip()) > 10:  # Only consider non-empty, reasonably sized sentences
                sentences.append(s.strip())
        
        # Take first few sentences as summary
        if sentences:
            summary_sentences = sentences[:min(NUM_SUMMARY_SENTENCES, len(sentences))]
            summary = " ".join(summary_sentences)
            print(f"        Fallback summary generated ({len(summary)} chars).")
            return summary
        else:
            # If no sentences found, take first N characters
            print("        No sentences found. Using text preview.")
            preview = text[:300] + "..." if len(text) > 300 else text
            return preview
    except Exception as e:
        print(f"      \033[91mError in fallback summary: {e}\033[0m")
        # Try using OpenAI if available
        try:
            if openai.api_key:
                return openai_summary(text)
        except Exception as openai_error:
            print(f"      \033[91mError using OpenAI summary: {openai_error}\033[0m")
            
        # Last resort fallback
        return text[:300] + "..." if len(text) > 300 else text

def openai_summary(text):
    """Generate an analytical summary using OpenAI API (v1.0.0+ syntax)."""
    # Check for API key first (best practice)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("      OpenAI API key not configured. Cannot generate AI summary.")
        return None
        
    try:
        print("      Generating analytical summary with OpenAI...")
        # Limit text size for API more aggressively for summary tasks
        max_text_len = 6000 # Adjust based on typical document length and token limits
        if len(text) > max_text_len:
            truncated_text = text[:max_text_len]
            print(f"        Warning: Text truncated to {max_text_len} chars for OpenAI summary.")
        else:
            truncated_text = text
            
        # Construct a more specific prompt
        # (Prompt definition remains the same)
        prompt_content = f"""Analyze the following document text, likely an info sheet or report from a youth advocacy center. 
Provide a concise summary (around {NUM_SUMMARY_SENTENCES}-{NUM_SUMMARY_SENTENCES+2} sentences) highlighting:

1. The main purpose or topic.
2. Key information or advice provided.
3. The intended audience (e.g., young people, parents, professionals).

Focus on the core message, relevance, and any calls to action. Avoid generic phrases.

Document Text:

{truncated_text}"""

        # Instantiate the client (automatically uses OPENAI_API_KEY env var)
        client = OpenAI()

        # Updated API Call using ChatCompletion
        response = client.chat.completions.create(
            # Using a recommended chat model - check OpenAI docs for latest/best options
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at summarizing documents."}, # Optional system message
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=200, 
            temperature=0.5 
        )
        
        # Updated response access
        summary = response.choices[0].message.content.strip()
        print(f"        OpenAI analytical summary generated ({len(summary)} chars).")
        return summary
    except Exception as e:
        print(f"      \033[91mError with OpenAI summary generation (v1.0.0+): {e}\033[0m")
        # Print specific OpenAI errors if available
        if hasattr(e, 'response') and e.response:
             print(f"        OpenAI Response Error: {e.response.status_code} - {e.response.text}")
        return None # Return None on failure, so fallback can be tried

def generate_keywords(text):
    """
    Improved: Generates relevant, document-specific keywords using spaCy and TF-IDF.
    - Extracts named entities and noun phrases with spaCy
    - Uses TF-IDF to find unique, high-value terms
    - Filters out generic/boilerplate keywords
    - Deduplicates and limits to 5 best keywords
    """
    if not text or not NLP:
        return None

    GENERIC_KEYWORDS = set([
        'youth', 'support', 'advocacy', 'legal', 'services', 'document', 'information', 'resource',
        'report', 'centre', 'center', 'annual', 'family', 'queensland', 'brisbane', 'system', 'person',
        'community', 'partnership', 'justice', 'child', 'young', 'people', 'sheet', 'form', 'pdf', 'webpage'
    ])

    try:
        print("      Generating improved keywords...")
        max_text_len = 100000
        doc = NLP(text[:max_text_len])
        candidates = set()

        # 1. Named entities (ORG, GPE, PERSON, etc.)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC", "PERSON", "PRODUCT", "LAW", "WORK_OF_ART"]:
                ent_text = ent.text.strip().lower()
                if len(ent_text) > 2 and ent_text not in GENERIC_KEYWORDS:
                    candidates.add(ent_text)

        # 2. Noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            if len(chunk_text) > 2 and chunk_text not in GENERIC_KEYWORDS:
                candidates.add(chunk_text)

        # 3. TF-IDF top terms
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=20,
                ngram_range=(1, 2),
                token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b'
            )
            tfidf = vectorizer.fit_transform([text])
            feature_array = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf.toarray()[0]
            top_indices = tfidf_scores.argsort()[::-1]
            for idx in top_indices:
                term = feature_array[idx].lower()
                if term not in GENERIC_KEYWORDS and len(term) > 2:
                    candidates.add(term)
                if len(candidates) >= 20:
                    break
        except Exception as e:
            print(f"      Warning: TF-IDF failed: {e}")

        # 4. Remove duplicates, filter, and prioritize
        def is_valid_keyword(k):
            if "http" in k or "www" in k:
                return False
            words = k.split()
            if len(words) == 0 or len(words) > 2:
                return False
            # Only allow alphabetic (no numbers or special chars)
            if not all(word.isalpha() for word in words):
                return False
            return True
        filtered = [k for k in candidates if k not in GENERIC_KEYWORDS and is_valid_keyword(k)]
        if not filtered:
            filtered = [k for k in candidates if is_valid_keyword(k)]
        # Sort by length and alphabetically for stability
        filtered = sorted(filtered, key=lambda x: (-len(x), x))
        # Limit to 5
        keywords = filtered[:5]
        print(f"        Improved keywords generated: {keywords}")
        return keywords
    except Exception as e:
        print(f"      Error generating improved keywords: {e}")
        return ["document", "report", "information", "resource"]

def sanitize_filename(filename):
    """Removes or replaces characters that are invalid in filenames."""
    filename = filename.strip()
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    max_len = 200
    if len(filename) > max_len:
        name, ext = os.path.splitext(filename)
        filename = name[:max_len - len(ext)] + ext
    # Ensure filename is not empty after sanitization
    if not filename:
        filename = "sanitized_empty_filename"
    return filename

def download_document(url, dest_folder):
    """Downloads a document from a URL and saves it. Returns (save_path, final_filename) on success, (None, best_guess_filename) otherwise."""
    save_path = None
    filename = None
    sanitized_filename = None
    try:
        # Derive initial filename
        path = urlparse(url).path
        filename = os.path.basename(path)
        if not filename:
            # Attempt to use the part before the last / if path ends with /
            parts = [p for p in path.split('/') if p]
            if parts:
                 filename = parts[-1] + os.path.splitext(path)[-1] # Get extension from original path
            else:
                 filename = urlparse(url).netloc # Fallback to domain if path is truly empty/root
        if not filename: # Final fallback
            filename = "downloaded_file"

        # Start download process
        print(f"  Attempting to download: {url}")
        response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        response.raise_for_status()

        # Sanitize and determine final save path
        sanitized_filename = sanitize_filename(filename)
        potential_save_path = os.path.join(dest_folder, sanitized_filename)
        
        counter = 1
        temp_save_path = potential_save_path
        while os.path.exists(temp_save_path):
            name, ext = os.path.splitext(potential_save_path)
            temp_save_path = f"{name}_{counter}{ext}"
            counter += 1
        save_path = temp_save_path # Final path determined
        final_filename = os.path.basename(save_path) # Get the actual final filename

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    \033[92mSuccessfully downloaded and saved to: {save_path}\033[0m")
        return save_path, final_filename # Return path and final filename

    except requests.exceptions.RequestException as e:
        print(f"    \033[91mError downloading {url}: {e}\033[0m")
    except Exception as e:
        print(f"    \033[91mAn unexpected error occurred downloading {url}: {e}\033[0m")
    
    # Return None for path, and the best guess sanitized filename if download failed
    best_guess_filename = sanitize_filename(filename if filename else "unknown_file")
    return None, best_guess_filename

def scrape_page(url, base_domain):
    """
    Fetches a URL, parses its HTML, and returns two sets: 
    one with absolute document URLs and one with absolute internal HTML page URLs.
    """
    doc_links = set()
    page_links = set()

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'html' not in content_type.lower():
            return doc_links, page_links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching or processing URL {url}: {e}")
        return doc_links, page_links
    except Exception as e:
        print(f"An unexpected error occurred processing {url}: {e}")
        return doc_links, page_links

    soup = BeautifulSoup(response.text, 'html.parser')
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if not href or href.startswith( ('#', 'mailto:', 'tel:', 'javascript:') ):
            continue
            
        absolute_url = urljoin(url, href)
        parsed_url = urlparse(absolute_url)
        path = parsed_url.path
        domain = parsed_url.netloc

        if not path: # Skip if path is empty
            continue

        # Check domain first for efficiency maybe?
        # if domain != base_domain and not any(path.lower().endswith(ext) for ext in TARGET_DOC_EXTENSIONS):
        #     continue

        if any(path.lower().endswith(ext) for ext in TARGET_DOC_EXTENSIONS):
            # Allow downloading docs from external domains if desired, otherwise check domain == base_domain
            if absolute_url.startswith(('http://', 'https://')):
                 doc_links.add(absolute_url)
        elif domain == base_domain:
            non_html_exts = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js', '.zip', '.xml'} 
            # Avoid queueing URLs that are documents we aren't targeting or other non-crawlable files
            if not any(path.lower().endswith(ext) for ext in non_html_exts.union(TARGET_DOC_EXTENSIONS)):
                clean_url = absolute_url.split('#')[0]
                if clean_url.startswith(('http://', 'https://')):
                    page_links.add(clean_url)
            
    return doc_links, page_links

def crawl_site(start_url, db_name_arg, incremental=True, include_webpages=True):
    base_domain = urlparse(start_url).netloc
    if not base_domain:
        print(f"Invalid start URL: {start_url}")
        return 0, 0

    # Load NLP model once at the start of the crawl
    if not load_spacy_model():
        print("Exiting due to spaCy model loading failure.")
        return 0, 0

    # Ensure NLTK data needed by sumy is available before starting crawl
    ensure_nltk_punkt()

    pages_crawled = 0
    downloaded_count = 0
    failed_count = 0
    webpages_added = 0
    db_added_count = 0
    db_doc_urls = set()
    conn, cursor = None, None

    try:
        conn, cursor = setup_database(db_name_arg)
        print(f"Database '{db_name_arg}' connected.")

        # Get existing document URLs from DB to avoid reprocessing
        if incremental:
            cursor.execute("SELECT source_url FROM documents")
            db_doc_urls = {row[0] for row in cursor.fetchall()}
            print(f"Incremental mode: Found {len(db_doc_urls)} existing documents in database")

        if not os.path.exists(DOWNLOAD_DIR):
            print(f"Creating download directory: {DOWNLOAD_DIR}")
            os.makedirs(DOWNLOAD_DIR)
        else:
            print(f"Using existing download directory: {DOWNLOAD_DIR}")

        queue = {start_url}
        visited = set()

        print(f"\nStarting crawl from {start_url} (domain: {base_domain})")
        print(f"Will crawl a maximum of {MAX_PAGES} pages.")
        print(f"Target document extensions: {TARGET_DOC_EXTENSIONS}")
        print(f"Also capturing webpages: {'Yes' if include_webpages else 'No'}")
        print(f"Mode: {'Incremental' if incremental else 'Full'} crawl")

        while queue and pages_crawled < MAX_PAGES:
            current_url = queue.pop()
            
            if current_url in visited:
                continue
                
            visited.add(current_url)
            pages_crawled += 1
            print(f"\nCrawling ({pages_crawled}/{MAX_PAGES}): {current_url}")

            docs_on_page, new_pages_to_visit = scrape_page(current_url, base_domain)
            
            # --- Process webpage content if required --- 
            if include_webpages and current_url.startswith(('http://', 'https://')) and urlparse(current_url).netloc == base_domain:
                # Check if we already have this webpage in DB
                if incremental and current_url in db_doc_urls:
                    print(f"  Webpage {current_url} already in database, skipping.")
                else:
                    # Scrape webpage content
                    title, content, publish_date = scrape_page_content(current_url)
                    if content:
                        # Add to database
                        if add_webpage_to_db(cursor, conn, current_url, title, content, publish_date):
                            webpages_added += 1
                            db_added_count += 1
                            db_doc_urls.add(current_url)
            
            # --- Process only new documents --- 
            new_docs_found = docs_on_page - db_doc_urls
            # --- If not incremental, process all docs found --- 
            if not incremental:
                 new_docs_found = docs_on_page # Process all if not incremental
                 print(f"  Full crawl mode: Found {len(new_docs_found)} document link(s) on this page (will process all).")
            elif new_docs_found:
                 print(f"  Incremental mode: Found {len(new_docs_found)} new document link(s) on this page.")

            # --- Process documents (new or all depending on mode) --- 
            if new_docs_found:
                for doc_url in new_docs_found:
                    # --- Check if already in DB (needed for full crawl mode mainly) --- 
                    cursor.execute("SELECT source_url, local_path, full_text, keywords, summary FROM documents WHERE source_url = ?", (doc_url,))
                    existing_record = cursor.fetchone()
                    needs_processing = True
                    if existing_record and incremental:
                         # Already processed in incremental mode, skip (unless re-processing is desired later)
                         # Add to db_doc_urls to ensure it's tracked as seen even if skipped
                         db_doc_urls.add(doc_url)
                         continue # Skip to next document
                    elif existing_record and not incremental:
                         # Full crawl mode - we have a record, maybe re-process?
                         # For now, let's just update the last_checked date and potentially re-download if missing
                         # but skip text extraction/summary if already present?
                         print(f"    Document {doc_url} exists in DB (Full Crawl). Checking local file and potentially updating...")
                         # Access tuple elements by index
                         local_save_path = existing_record[1] # local_path is at index 1
                         final_filename = os.path.basename(local_save_path) if local_save_path else None
                         extracted_text = existing_record[2] # full_text is at index 2
                         keywords = existing_record[3]       # keywords is at index 3
                         summary = existing_record[4]        # summary is at index 4
                         # Attempt to get document_date if it exists (might not in older DB versions)
                         document_date = None
                         try:
                              # Check number of columns returned
                              if len(existing_record) > 5:
                                   document_date = existing_record[5] # Assume document_date is next if present
                         except IndexError:
                              pass # document_date column doesn't exist or wasn't selected
                         needs_processing = False # Assume we don't need full processing unless checks fail
                         # Check if file still exists, re-download if not
                         if local_save_path and not os.path.exists(local_save_path):
                              print(f"      Local file missing for {final_filename}. Re-downloading...")
                              local_save_path, final_filename = download_document(doc_url, DOWNLOAD_DIR)
                              if local_save_path:
                                   downloaded_count += 1
                                   needs_processing = True # Need to process text if re-downloaded
                              else:
                                   failed_count += 1
                         elif not local_save_path:
                              print(f"      No local path in DB for {doc_url}. Attempting download...")
                              local_save_path, final_filename = download_document(doc_url, DOWNLOAD_DIR)
                              if local_save_path:
                                   downloaded_count += 1
                                   needs_processing = True # Need to process text if downloaded
                              else:
                                   failed_count += 1
                         # Optional: Force re-processing even in full crawl if text/summary is missing
                         if not extracted_text or not summary:
                              print(f"      Missing text or summary for existing record {final_filename}. Marking for processing.")
                              needs_processing = True
                    # --- Download if it's a truly new document --- 
                    if needs_processing and not existing_record:
                        local_save_path, final_filename = download_document(doc_url, DOWNLOAD_DIR)
                        if local_save_path:
                            downloaded_count += 1
                        else:
                            failed_count += 1
                            # Cannot process if download failed
                            add_or_update_document_record(cursor, conn, doc_url, None, final_filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), None, None, None, None)
                            db_added_count +=1 
                            db_doc_urls.add(doc_url)
                            continue # Skip to next document

                    # --- Process Text, Date, Keywords, Summary --- 
                    if needs_processing and local_save_path: 
                        try:
                            print(f"    Processing text/summary/keywords for: {final_filename}")
                            extracted_text, document_date = extract_text(local_save_path)
                                
                            if extracted_text:
                                keywords = generate_keywords(extracted_text)
                                
                                # --- Updated Summary Logic --- 
                                summary = None
                                # 1. Try OpenAI first if key exists
                                if openai.api_key:
                                    summary = openai_summary(extracted_text)
                                
                                # 2. Fallback to Sumy LSA if OpenAI failed or no key
                                if not summary:
                                    summary = generate_summary(extracted_text)
                                # 3. Fallback to basic sentence extraction if Sumy failed
                                # (generate_summary already calls fallback_summary internally)
                                if not summary: # Check again in case generate_summary returned None directly
                                    summary = fallback_summary(extracted_text)
                                # Ensure summary is never completely None if text exists
                                if not summary and extracted_text:
                                    summary = extracted_text[:300] + "..." # Final safety net
                                    print("        Used basic text preview as final summary fallback.")
                                    
                        except Exception as e:
                            print(f"    \033[91mError during post-download processing for {final_filename}: {e}\033[0m")
                            # Keep extracted_text/date as None if error occurs here
                            extracted_text = None
                            keywords = None
                            summary = None
                            # document_date might be from metadata, preserve if possible?
                            # For safety, reset if processing failed badly
                            # document_date = None 
                    
                    # --- Add/Update Database Record --- 
                    # Always update last_checked_date, potentially update other fields
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    add_or_update_document_record(cursor, conn, doc_url, local_save_path, final_filename, timestamp, extracted_text, keywords, summary, document_date)
                    
                    # Update document_type to PDF
                    cursor.execute("UPDATE documents SET document_type = ? WHERE source_url = ?", ("pdf", doc_url))
                    conn.commit()
                    
                    db_added_count +=1 # Increment if added OR updated
                    db_doc_urls.add(doc_url) # Mark as processed/seen
            
            # --- Update queue --- 
            new_pages_to_queue = new_pages_to_visit - visited - queue
            queue.update(new_pages_to_queue)

            time.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"\n--- An error occurred during crawl: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print(f"\nDatabase '{db_name_arg}' connection closed.")

    print("\n--- Crawl Summary ---")
    if pages_crawled >= MAX_PAGES:
        print(f"Reached maximum page limit ({MAX_PAGES}). Crawl stopped.")
    else:
        print("Crawl finished.")
    print(f"Pages crawled: {pages_crawled}")
    print(f"Total unique document links processed for DB: {len(db_doc_urls)}")
    print(f"Documents successfully downloaded: {downloaded_count}")
    print(f"Document download failures: {failed_count}")
    print(f"Webpages scraped and added: {webpages_added}")
    print(f"Database records added/updated: {db_added_count}")
    if downloaded_count > 0 or failed_count > 0:
         print(f"Downloads attempted in directory: '{os.path.abspath(DOWNLOAD_DIR)}'")
    print(f"Database file: '{os.path.abspath(db_name_arg)}'")

    return downloaded_count, db_added_count

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # It's best to store your API key in an environment variable

def summarize_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use a free-tier model if available
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=150  # Adjust based on your needs
    )
    return response.choices[0].text.strip()

def extract_keywords(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Extract keywords from the following text:\n\n{text}",
        max_tokens=50  # Adjust based on your needs
    )
    return response.choices[0].text.strip()

def migrate_keywords_to_json(db_name_arg):
    """Migrates existing comma-separated keywords in database to JSON format."""
    try:
        print(f"Migrating keywords to JSON format in {db_name_arg}...")
        conn = sqlite3.connect(db_name_arg)
        cursor = conn.cursor()
        
        # First check if the table and column exist
        cursor.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if 'keywords' not in columns:
            print("  Keywords column not found in database. No migration needed.")
            conn.close()
            return
        
        # Get all records with keywords
        cursor.execute("SELECT rowid, keywords FROM documents WHERE keywords IS NOT NULL")
        rows = cursor.fetchall()
        
        print(f"  Found {len(rows)} records with keywords to check")
        updates = 0
        
        for rowid, keywords in rows:
            # Skip empty keywords
            if not keywords or not keywords.strip():
                continue
                
            # Check if already in JSON format
            if keywords.strip().startswith('[') and keywords.strip().endswith(']'):
                try:
                    # Validate it's proper JSON
                    json.loads(keywords)
                    # Skip if it's already valid JSON
                    continue
                except json.JSONDecodeError:
                    # Not valid JSON despite brackets, will convert
                    pass
            
            # Convert comma-separated to JSON
            try:
                # Split by comma, clean up, and convert to JSON
                keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
                if keywords_list:
                    json_keywords = json.dumps(keywords_list)
                    cursor.execute("UPDATE documents SET keywords = ? WHERE rowid = ?", 
                                 (json_keywords, rowid))
                    updates += 1
            except Exception as e:
                print(f"    Error converting keywords for row {rowid}: {e}")
        
        conn.commit()
        print(f"  Successfully migrated {updates} records to JSON format")
        conn.close()
        return updates
    except Exception as e:
        print(f"Error during keywords migration: {e}")
        return 0

def scrape_page_content(url):
    """
    Extracts content from a webpage for storage in the database.
    Returns a tuple of (title, content, publish_date)
    """
    try:
        print(f"  Extracting webpage content from: {url}")
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "Unknown Title"
        title = title.strip()
        print(f"    Found title: {title}")
        
        # Extract main content
        # Try common content containers
        content_tags = [
            soup.find('article'),
            soup.find('main'),
            soup.find(id='content'),
            soup.find(class_='content'),
            soup.find(id='main'),
            soup.find(class_='main')
        ]
        
        content_tag = next((tag for tag in content_tags if tag), None)
        
        if not content_tag:
            # Fallback to body
            content_tag = soup.body
        
        # Remove scripts, styles, and other non-content elements
        for tag in content_tag.find_all(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
            
        content = content_tag.get_text(separator=' ', strip=True)
        
        # Truncate content if too long
        if len(content) > 100000:
            content = content[:100000] + "... [content truncated]"
            
        print(f"    Extracted {len(content)} characters of content")
        
        # Try to find publish date
        publish_date = None
        
        # Method 1: Look for meta tags
        meta_dates = []
        
        for meta in soup.find_all('meta'):
            property = meta.get('property', '').lower()
            name = meta.get('name', '').lower()
            
            if property in ['article:published_time', 'og:published_time'] or name in ['date', 'publish-date', 'publication-date']:
                date_str = meta.get('content')
                if date_str:
                    try:
                        parsed_date = date_parse(date_str)
                        meta_dates.append(parsed_date.strftime('%Y-%m-%d'))
                    except (ValueError, ParserError):
                        pass
        
        # Method 2: Look for time tags
        time_tags = soup.find_all('time')
        for time_tag in time_tags:
            datetime_attr = time_tag.get('datetime')
            if datetime_attr:
                try:
                    parsed_date = date_parse(datetime_attr)
                    meta_dates.append(parsed_date.strftime('%Y-%m-%d'))
                except (ValueError, ParserError):
                    pass
        
        # Use the most common date found
        if meta_dates:
            publish_date = Counter(meta_dates).most_common(1)[0][0]
            print(f"    Found publish date: {publish_date}")
            
        return title, content, publish_date
    except Exception as e:
        print(f"  \033[91mError extracting webpage content: {e}\033[0m")
        return None, None, None
        
def add_webpage_to_db(cursor, conn, url, title, content, publish_date):
    """
    Adds a webpage to the database.
    """
    try:
        # Generate a filename-like ID for the webpage
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        if not path:
            path = parsed_url.netloc
        
        # Replace special characters and make it URL-safe
        page_id = re.sub(r'[^a-zA-Z0-9_-]', '_', path)
        
        # Add a date stamp if available
        if publish_date:
            page_id = f"{page_id}_{publish_date.replace('-', '')}"
        
        # Truncate if too long
        if len(page_id) > 200:
            page_id = page_id[:200]
            
        # Generate keywords and summary
        keywords = generate_keywords(content) if content else None
        
        # Generate summary
        summary = None
        if content:
            # Try OpenAI if available
            if openai.api_key:
                summary = openai_summary(content)
            
            # Fallback to Sumy
            if not summary:
                summary = generate_summary(content)
                
            # Final fallback
            if not summary:
                summary = fallback_summary(content)
                
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to database
        add_or_update_document_record(
            cursor, conn, url, None, page_id, timestamp, 
            content, keywords, summary, publish_date
        )
        
        # Update document_type column
        cursor.execute(
            "UPDATE documents SET document_type = ? WHERE source_url = ?", 
            ("webpage", url)
        )
        conn.commit()
        
        print(f"  \033[92mSuccessfully added webpage to database: {page_id}\033[0m")
        return True
    except Exception as e:
        print(f"  \033[91mError adding webpage to database: {e}\033[0m")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawl a website and download documents.')
    parser.add_argument('--url', default="https://yac.net.au/", help='Target URL to crawl')
    parser.add_argument('--db', default="yac_docs.db", help='Database filename')
    parser.add_argument('--full', action='store_true', help='Perform a full crawl (not incremental)')
    parser.add_argument('--migrate-only', action='store_true', help='Only migrate keywords to JSON format without crawling')
    parser.add_argument('--no-webpages', action='store_true', help='Do not scrape webpage content, only documents')
    
    args = parser.parse_args()
    
    # Always migrate keywords to ensure consistent format
    migrate_keywords_to_json(args.db)
    
    # If migration-only mode, exit after migration
    if args.migrate_only:
        print("Migration completed. Exiting without crawling.")
        sys.exit(0)
    
    print(f"Starting site crawl, document download, and DB logging for {args.url}...")
    print(f"Database file will be: {args.db}")
    
    successful_downloads, db_records = crawl_site(
        args.url, 
        args.db, 
        not args.full,
        not args.no_webpages
    )
    
    print(f"\nProcess completed. {successful_downloads} documents downloaded. {db_records} DB records managed.") 