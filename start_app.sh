#!/bin/bash

# YAC AI Flask App Starter Script
# Starts the Flask app with proper configuration

# Configuration
APP_PATH="app.py"
DEFAULT_PORT=5001
HOST="127.0.0.1"
MAX_PORT_ATTEMPTS=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATE_SCRIPT="${SCRIPT_DIR}/migrate_keywords.py"
DB_PATH="${SCRIPT_DIR}/yac_docs.db"
FRESH_BROWSER="${SCRIPT_DIR}/fresh_browser.sh"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line options
OPEN_BROWSER=false
while getopts "ob" opt; do
  case $opt in
    o|b) OPEN_BROWSER=true ;;
    *) echo "Unknown option: $opt" ;;
  esac
done

# Run the keyword migration script first
echo "Running keyword migration script..."
cd "${SCRIPT_DIR}" && python3 "${MIGRATE_SCRIPT}" --db "${DB_PATH}"

# Function to check if a port is in use
is_port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
    return $?
}

# Function to kill process using a specific port
kill_process_on_port() {
    local port=$1
    local pid=$(lsof -ti :$port)
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Found process (PID: $pid) using port $port${NC}"
        echo -e "${YELLOW}Terminating process...${NC}"
        kill -15 $pid 2>/dev/null
        sleep 1
        
        # Check if process is still running
        if is_port_in_use $port; then
            echo -e "${YELLOW}Process still active, using forceful termination...${NC}"
            kill -9 $pid 2>/dev/null
            sleep 1
        fi
        
        # Final check
        if ! is_port_in_use $port; then
            echo -e "${GREEN}Successfully freed port $port${NC}"
            return 0
        else
            echo -e "${RED}Failed to free port $port${NC}"
            return 1
        fi
    fi
    return 1
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local current_port=$start_port
    local attempts=0
    
    while [ $attempts -lt $MAX_PORT_ATTEMPTS ]; do
        if ! is_port_in_use $current_port; then
            echo $current_port
            return 0
        fi
        ((current_port++))
        ((attempts++))
    done
    
    echo -1
    return 1
}

# Start the application
start_application() {
    local port=$1
    
    echo -e "${BLUE}Starting YAC AI Document Explorer on http://$HOST:$port/${NC}"
    python3 "${APP_PATH}" --port "${port}" --host "${HOST}"
}

# Main execution starts here
PORT=$DEFAULT_PORT

# Check if the default port is in use
if is_port_in_use $PORT; then
    echo -e "${YELLOW}Port $PORT is already in use.${NC}"
    
    # Ask user what to do
    echo -e "Choose an option:"
    echo "1) Terminate the process using port $PORT and continue"
    echo "2) Try a different port automatically"
    read -p "Enter choice (1 or 2, default is 2): " choice
    
    if [ "$choice" == "1" ]; then
        # Try to kill the process
        if kill_process_on_port $PORT; then
            echo -e "${GREEN}Port $PORT is now available.${NC}"
            # Small delay to ensure port is fully released
            sleep 1
        else
            echo -e "${RED}Could not free port $PORT. Trying alternative port...${NC}"
            PORT=$(find_available_port $((PORT + 1)))
        fi
    else
        # Find an available port
        PORT=$(find_available_port $((PORT + 1)))
    fi
    
    # If port is still -1, all attempts failed
    if [ "$PORT" == "-1" ]; then
        echo -e "${RED}Could not find an available port after $MAX_PORT_ATTEMPTS attempts.${NC}"
        echo "Please check running processes and free up some ports manually."
        exit 1
    fi
fi

# Start the application with the selected port
start_application $PORT

# Start the app in background if we need to open browser
if [ "$OPEN_BROWSER" = true ]; then
    # Start app in background
    cd "${SCRIPT_DIR}" && python3 "${APP_PATH}" --port "${PORT}" --host "${HOST}" &
    APP_PID=$!
    
    # Give the server a moment to start
    echo "Waiting for server to start..."
    sleep 2
    
    # Open with fresh browser
    echo "Opening with cache-disabled browser..."
    bash "${FRESH_BROWSER}"
    
    # Keep the script running until Ctrl+C
    echo "Server running with PID: $APP_PID"
    echo "Press Ctrl+C to stop the server"
    
    # Wait for user to interrupt
    trap "kill $APP_PID; echo 'Server stopped'; exit 0" INT
    wait $APP_PID
else
    # Start normally in foreground
    cd "${SCRIPT_DIR}" && python3 "${APP_PATH}" --port "${PORT}" --host "${HOST}"
fi 