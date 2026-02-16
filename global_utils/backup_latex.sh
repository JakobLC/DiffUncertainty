#!/bin/bash

# Get current date and time for filename
DATETIME=$(date +"%Y%m%d_%H%M%S")

# Define paths
SOURCE_DIR="/home/jloch/Desktop/diff/writing/ECCV2026/ECCV_2026_AU_EU"
BACKUP_DIR="/data/eccv_backups"
BACKUP_FILE="backup_${DATETIME}.zip"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create zip archive of the source directory
echo "Creating backup: ${BACKUP_FILE}"
cd "$(dirname "$SOURCE_DIR")" && zip -r "$BACKUP_PATH" "$(basename "$SOURCE_DIR")"

# Check if zip was successful
if [ $? -eq 0 ]; then
    echo "Backup created successfully at: ${BACKUP_PATH}"
else
    echo "Error: Backup failed"
    exit 1
fi
