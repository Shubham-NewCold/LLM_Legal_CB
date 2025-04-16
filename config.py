# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask configuration
DEBUG = False # Keep False for production/deployment
PORT = int(os.environ.get("PORT", 5000))

# Directory settings
PDF_DIR = "pdfs"
# PERSIST_DIRECTORY = "faiss_db" # No longer needed

# --- ColBERT Settings ---
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERT_INDEX_ROOT = "colbert_indices"
COLBERT_INDEX_NAME = "legal_docs_colbert_v2"
COLBERT_METADATA_MAP_FILENAME = "colbert_metadata_map.pkl"
COLBERT_PID_DOCID_MAP_FILENAME = "colbert_pid_docid_map.pkl"

# --- LLM Settings ---
TEMPERATURE = 0.2
MAX_TOKENS = 2048
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://nec-us2-ai.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# LangChain tracing
PROJECT_NAME = "pr-new-molecule-89"

# Parsing thresholds
MAX_TOKENS_THRESHOLD = 350

# --- Retrieval Settings ---
# Initial number of documents to retrieve from ColBERT
COLBERT_TOP_K = 150 # Set high for initial pool

# --- ColBERT Score Thresholds & Limits --- <<< MODIFIED SECTION
# Global flag to enable/disable score thresholding
COLBERT_SCORE_THRESHOLD_ENABLED = True

# Threshold for single-customer queries (applied before min/max limits)
# Tune this based on observed scores for relevant single-customer results
COLBERT_SINGLE_CUSTOMER_SCORE_THRESHOLD = 10.0 # <<< EXAMPLE VALUE - TUNE!

# Threshold for general (no customer) queries (applied before min/max limits)
# Might be lower as relevance could be broader
COLBERT_GENERAL_SCORE_THRESHOLD = 9.0 # <<< EXAMPLE VALUE - TUNE!

# Threshold for comparative queries (applied *during* Pass 2 of smart selection)
# This filters candidates added *after* the initial minimum coverage pass
COLBERT_PASS_2_SCORE_THRESHOLD = 10.0 # <<< EXAMPLE VALUE - TUNE!

# Overall Minimum and Maximum number of chunks to send to the LLM
# Applied *after* thresholding and smart selection logic
DYNAMIC_K_MIN_CHUNKS = 1
DYNAMIC_K_MAX_CHUNKS = 15

# --- REMOVE OLD/UNUSED SETTINGS ---
# ...

# --- Azure Checks ---
# [ ... Keep Azure credential checks ... ]
if not AZURE_OPENAI_ENDPOINT: print("CRITICAL WARNING: AZURE_OPENAI_ENDPOINT not set.")
if not AZURE_OPENAI_API_KEY: print("CRITICAL WARNING: AZURE_OPENAI_API_KEY not set.")
if not AZURE_OPENAI_DEPLOYMENT_NAME: print("CRITICAL WARNING: AZURE_OPENAI_DEPLOYMENT_NAME not set.")

print(f"--- Config Loaded ---")
print(f"Retrieval Method: ColBERT")
print(f"ColBERT Model: {COLBERT_MODEL_NAME}")
print(f"ColBERT Index Name: {COLBERT_INDEX_NAME}")
print(f"ColBERT Initial Retrieval K: {COLBERT_TOP_K}")
print(f"ColBERT Score Thresholding Enabled: {COLBERT_SCORE_THRESHOLD_ENABLED}")
if COLBERT_SCORE_THRESHOLD_ENABLED:
    print(f"  Single Customer Threshold: {COLBERT_SINGLE_CUSTOMER_SCORE_THRESHOLD}")
    print(f"  General Query Threshold: {COLBERT_GENERAL_SCORE_THRESHOLD}")
    print(f"  Comparative Pass 2 Threshold: {COLBERT_PASS_2_SCORE_THRESHOLD}")
print(f"Final Chunk Range (Min/Max): {DYNAMIC_K_MIN_CHUNKS}/{DYNAMIC_K_MAX_CHUNKS}")
print(f"LLM Provider: Azure OpenAI")
print(f"  Endpoint Set: {'Yes' if AZURE_OPENAI_ENDPOINT else 'NO'}")
print(f"  API Key Set: {'Yes' if AZURE_OPENAI_API_KEY else 'NO'}")
print(f"  Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME if AZURE_OPENAI_DEPLOYMENT_NAME else 'NOT SET'}")
print(f"  API Version: {AZURE_OPENAI_API_VERSION}")
print(f"--------------------")