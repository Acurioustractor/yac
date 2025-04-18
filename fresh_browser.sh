#!/bin/bash

# Configuration
APP_URL="http://127.0.0.1:5001/?fresh=$(date +%s)"
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SAFARI_PATH="/Applications/Safari.app/Contents/MacOS/Safari"

# Check if Chrome is installed
if [ -f "$CHROME_PATH" ]; then
    echo "Launching Chrome with cache disabled..."
    # Launch Chrome with cache disabled
    "$CHROME_PATH" --incognito --disable-application-cache --disable-cache "$APP_URL"
    exit 0
fi

# If Chrome isn't available, try Safari
if [ -f "$SAFARI_PATH" ]; then
    echo "Launching Safari in private mode..."
    # Safari doesn't have direct command line options for disabling cache
    # but private browsing helps
    open -a Safari "$APP_URL"
    echo "Please select 'New Private Window' from the File menu"
    exit 0
fi

# As a last resort, try the default browser
echo "Chrome or Safari not found. Launching with default browser..."
open "$APP_URL"
echo "Tip: You may need to use Ctrl+Shift+R (or Cmd+Shift+R on Mac) to force refresh" 