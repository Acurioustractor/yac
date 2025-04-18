# YAC AI Document Manager

A document management system for scraping, storing, and viewing PDF and DOCX documents.

## Keywords System Improvements

The document keyword system has been improved with the following features:

### 1. Enhanced Keyword Generation

- The spaCy-based keyword extraction now utilizes lemmatization to recognize similar words
- Improved filtering to exclude low-quality keywords
- Better handling of multi-word phrases
- Smarter frequency-based ranking with duplicate elimination

### 2. Robust Storage Format

- Keywords are now stored in JSON format instead of comma-separated strings
- This allows for special characters in keywords without breaking parsing
- More consistent handling across different parts of the application

### 3. Frontend Improvements

- Improved keyword display with clickable keyword buttons
- More robust parsing that handles both JSON and legacy comma-separated formats
- Clear visual indication of active keyword filters
- Ability to clear filters with a dedicated button

### 4. Migration Tools

A migration system is included to convert existing keywords to the new format:

- Automatic migration runs at application startup
- A dedicated migration script (`migrate_keywords.py`) can be run manually:

```bash
python3 migrate_keywords.py --db yac_docs.db
```

## Running the Application

To start the application with all improvements:

```bash
./start_app.sh
```

This will run the keyword migration and start the Flask server on port 5001.

### Preventing Browser Caching

If you want to ensure you're always seeing the latest frontend without caching issues:

```bash
# Start with a fresh browser automatically opened
./start_app.sh -b
```

This will:
1. Start the server in the background
2. Open a browser with cache disabled
3. Keep running until you press Ctrl+C

You can also:
1. Use the "CLEAR CACHE & REFRESH" button in the UI
2. Manually run `./fresh_browser.sh` to launch a cache-disabled browser
3. Use Ctrl+Shift+R (or Cmd+Shift+R on Mac) to force refresh in your browser

## System Requirements

- Python 3.x
- spaCy with 'en_core_web_sm' model
- NLTK with 'punkt' data
- Flask
- Other required packages from imports 