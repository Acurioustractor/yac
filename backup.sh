#!/bin/bash

# YAC AI Document Database Backup Script
# Create a timestamped backup of the database

# Configuration
DB_FILE="yac_docs.db"
BACKUP_DIR="backups"
DATE_FORMAT=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/yac_docs_${DATE_FORMAT}.db"
MAX_BACKUPS=10  # Keep only the last 10 backups

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Create backup
echo "Creating backup of ${DB_FILE} to ${BACKUP_FILE}..."
if [ -f "${DB_FILE}" ]; then
    cp "${DB_FILE}" "${BACKUP_FILE}"
    echo "Backup completed successfully."
    
    # Remove old backups if we have more than MAX_BACKUPS
    NUM_BACKUPS=$(ls -1 "${BACKUP_DIR}"/yac_docs_*.db 2>/dev/null | wc -l)
    if [ "${NUM_BACKUPS}" -gt "${MAX_BACKUPS}" ]; then
        echo "Removing old backups (keeping last ${MAX_BACKUPS})..."
        ls -t "${BACKUP_DIR}"/yac_docs_*.db | tail -n +$((MAX_BACKUPS+1)) | xargs rm -f
    fi
else
    echo "Error: Database file ${DB_FILE} not found!"
    exit 1
fi

echo "Backup process completed at $(date)." 