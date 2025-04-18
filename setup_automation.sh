#!/bin/bash

# YAC AI Automation Setup Script
# Sets up cron jobs for regular backups and crawling

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="${SCRIPT_DIR}/backup.sh"
SCRAPER_SCRIPT="${SCRIPT_DIR}/scraper.py"
DB_FILE="${SCRIPT_DIR}/yac_docs.db"

# Ensure scripts are executable
chmod +x "${BACKUP_SCRIPT}"

# Create cron entries
# Run backup daily at 1 AM
BACKUP_CRON="0 1 * * * ${BACKUP_SCRIPT}"

# Run crawler weekly on Sunday at 2 AM (incremental mode)
CRAWLER_CRON="0 2 * * 0 cd ${SCRIPT_DIR} && python3 ${SCRAPER_SCRIPT} --db ${DB_FILE}"

# Full crawl monthly on the 1st at 3 AM
FULL_CRAWLER_CRON="0 3 1 * * cd ${SCRIPT_DIR} && python3 ${SCRAPER_SCRIPT} --db ${DB_FILE} --full"

# Display cron entries
echo "The following cron jobs will be added:"
echo "----------------------------------------"
echo "Daily backup at 1 AM:"
echo "${BACKUP_CRON}"
echo
echo "Weekly incremental crawl on Sunday at 2 AM:"
echo "${CRAWLER_CRON}"
echo
echo "Monthly full crawl on the 1st at 3 AM:"
echo "${FULL_CRAWLER_CRON}"
echo "----------------------------------------"

echo
echo "To install these cron jobs, run:"
echo "crontab -e"
echo
echo "Then add these lines to your crontab:"
echo "${BACKUP_CRON}"
echo "${CRAWLER_CRON}"
echo "${FULL_CRAWLER_CRON}"
echo
echo "Or you can run this command to install them automatically:"
echo "(crontab -l 2>/dev/null; echo \"${BACKUP_CRON}\"; echo \"${CRAWLER_CRON}\"; echo \"${FULL_CRAWLER_CRON}\") | crontab -" 