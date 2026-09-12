#!/usr/bin/env bash
# Helper script to download noodulz/pokemon-dataset-1000 into project folder
set -e
TARGET_DIR="pokemon-dataset-1000"
mkdir -p "$TARGET_DIR"

if [ -f ~/.kaggle/kaggle.json ]; then
  echo "Found ~/.kaggle/kaggle.json — downloading dataset..."
  kaggle datasets download -d noodulz/pokemon-dataset-1000 -p "$TARGET_DIR" --unzip
  echo "Download complete."
else
  echo "No Kaggle token at ~/.kaggle/kaggle.json"
  echo "You can either:"
  echo "  1) Place kaggle.json at ~/.kaggle/kaggle.json and rerun this script"
  echo "  2) Export KAGGLE_USERNAME and KAGGLE_KEY environment variables and run the kaggle command manually"
  exit 2
fi
