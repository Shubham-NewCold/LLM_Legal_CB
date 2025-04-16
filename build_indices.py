# build_indices.py
import os
import sys
import pickle
import shutil
import traceback
import re # For cleaning IDs
from typing import List # Make sure this is imported

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from config import (
    PDF_DIR,
    COLBERT_MODEL_NAME,
    COLBERT_INDEX_ROOT,
    COLBERT_INDEX_NAME,
    COLBERT_METADATA_MAP_FILENAME,
    COLBERT_PID_DOCID_MAP_FILENAME,
    # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY # Needed for pdf_extractor
)
from document_processing.pdf_extractor import extract_documents_from_pdf
from document_processing.parser import pyparse_hierarchical_chunk_text
from langchain_core.documents import Document

# --- ColBERT Imports ---
from colbert import Indexer
from colbert.infra import ColBERTConfig

# --- Define paths ---
PDF_DIR_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), PDF_DIR))
# Define root path for ColBERT related files (maps, parent of index dir)
COLBERT_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), COLBERT_INDEX_ROOT))
# Paths for our mapping files (within the base path)
METADATA_MAP_PATH = os.path.join(COLBERT_BASE_PATH, COLBERT_METADATA_MAP_FILENAME)
PID_DOCID_MAP_PATH = os.path.join(COLBERT_BASE_PATH, COLBERT_PID_DOCID_MAP_FILENAME)

CUSTOMER_LIST_FILE = "detected_customers.txt"
CUSTOMER_LIST_PATH = os.path.join(project_root, CUSTOMER_LIST_FILE)


# --- Document Loading and Parsing Logic (Unchanged) ---
def load_all_documents_for_indexing(pdf_directory):
    """Loads PDFs, extracts pages, parses hierarchically for indexing."""
    all_final_documents = []
    overall_chunk_index = 0 # Keep track if needed, though not strictly used later

    if not os.path.isdir(pdf_directory):
        print(f"ERROR: PDF directory not found: {pdf_directory}")
        return []

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {pdf_directory}")

    for file in pdf_files:
        file_path = os.path.join(pdf_directory, file)
        print(f"\n--- Processing {file} ---") # More concise logging
        try:
            # This function now extracts customer name using LLM and adds to metadata
            page_documents = extract_documents_from_pdf(file_path)
            if not page_documents:
                 print(f"WARN: No documents extracted from {file}. Skipping.")
                 continue
        except Exception as e:
            print(f"ERROR: Failed to extract pages from {file}: {e}")
            traceback.print_exc()
            continue

        current_hierarchy_stack = []
        # Process each page document extracted from the PDF
        for doc_index, doc_obj in enumerate(page_documents):
            page_content = doc_obj.page_content
            page_metadata = doc_obj.metadata
            source_file = page_metadata.get('source', file) # Should always be set by extractor
            page_number = page_metadata.get('page_number', doc_index + 1) # Use index as fallback
            # Customer name is already in metadata from pdf_extractor
            customer_name = page_metadata.get('customer', 'Unknown Customer')
            region_name = page_metadata.get('region', 'Unknown Region')
            word_count = len(page_content.split())

            # Prepare metadata for the parser, ensuring customer name is included
            parser_metadata = {
                'source': source_file,
                'customer': customer_name, # Pass the extracted/normalized name
                'region': region_name,
                'clause': 'N/A', # Default clause, parser will update if header found
                'hierarchy': [] # Default hierarchy, parser will update
            }
            # Merge any other metadata from the original page document
            parser_metadata.update({k: v for k, v in page_metadata.items() if k not in parser_metadata})

            try:
                from config import MAX_TOKENS_THRESHOLD as MTT
            except ImportError:
                MTT = 350 # Use default from config if import fails
                print(f"WARN: MAX_TOKENS_THRESHOLD not found in config, using default {MTT}")

            # Decide whether to parse hierarchically based on word count
            if word_count > MTT:
                # print(f"DEBUG: Page {page_number} word count ({word_count}) > {MTT}. Parsing hierarchically.") # Optional Debug
                try:
                    # Parse the page content into smaller chunks
                    parsed_page_docs, current_hierarchy_stack = pyparse_hierarchical_chunk_text(
                        full_text=page_content,
                        source_name=source_file,
                        page_number=page_number,
                        extra_metadata=parser_metadata, # Pass combined metadata
                        initial_stack=current_hierarchy_stack # Maintain hierarchy across pages
                    )
                    # Add the resulting chunks (which inherit/update metadata)
                    all_final_documents.extend(parsed_page_docs)
                except Exception as e:
                    print(f"ERROR: Failed to parse page {page_number} of {file}: {e}")
                    traceback.print_exc()
                    print(f"  WARNING: Adding page {page_number} as whole chunk due to parsing error.")
                    # Add the original page document as a single chunk, ensuring metadata is consistent
                    doc_obj.metadata = parser_metadata # Use the prepared metadata
                    doc_obj.metadata['page_number'] = page_number
                    # Try to add current hierarchy state even on error
                    doc_obj.metadata['hierarchy'] = [item[0] for item in current_hierarchy_stack] if current_hierarchy_stack else []
                    doc_obj.metadata['clause'] = current_hierarchy_stack[-1][0] if current_hierarchy_stack else 'N/A'
                    all_final_documents.append(doc_obj)
            else:
                # Page is short enough, add as a single chunk
                # print(f"DEBUG: Page {page_number} word count ({word_count}) <= {MTT}. Adding as whole chunk.") # Optional Debug
                doc_obj.metadata = parser_metadata # Use the prepared metadata
                doc_obj.metadata['page_number'] = page_number
                # Add current hierarchy state
                doc_obj.metadata['hierarchy'] = [item[0] for item in current_hierarchy_stack] if current_hierarchy_stack else []
                doc_obj.metadata['clause'] = current_hierarchy_stack[-1][0] if current_hierarchy_stack else 'N/A'
                all_final_documents.append(doc_obj)

    print(f"\nTotal documents processed into chunks: {len(all_final_documents)}")
    return all_final_documents


# --- Function to Build ColBERT Index ---
def build_colbert_index(documents: List[Document]):
    """Builds and saves a ColBERT index from LangChain documents."""
    if not documents:
        print("ERROR: No documents provided to build ColBERT index.")
        return False

    print(f"\nPreparing data for ColBERT indexing ({len(documents)} documents)...")

    # --- 1. Prepare data in ColBERT format (List of strings) & Metadata Map ---
    collection = []
    # Modify metadata_map to store content AND metadata
    metadata_map = {} # Dictionary: doc_id -> {'page_content': str, 'metadata': dict}
    doc_id_counter = 0

    for doc in documents:
        doc_id = f"doc_{doc_id_counter}"
        doc_id_counter += 1

        content = doc.page_content or ""
        cleaned_content = content.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        collection.append(cleaned_content) # ColBERT still just needs the text

        # --- Store BOTH content and metadata in the map --- <<< MODIFIED
        metadata_map[doc_id] = {
            'page_content': content, # Store original content
            'metadata': doc.metadata # Store original metadata dict
        }
        # --- End Modification ---

    print(f"Data prepared. {len(collection)} entries in collection.")

    # --- 2. Configure and Run ColBERT Indexer ---
    # Ensure the BASE directory exists
    if not os.path.exists(COLBERT_BASE_PATH):
        os.makedirs(COLBERT_BASE_PATH)
        print(f"Created ColBERT base directory: {COLBERT_BASE_PATH}")

    # --- Check and remove the *expected default* index path if it exists ---
    # This path includes the default experiment structure
    default_exp_index_path = os.path.join(COLBERT_BASE_PATH, "experiments", "default", "indexes", COLBERT_INDEX_NAME)
    if os.path.exists(default_exp_index_path):
        print(f"Removing existing ColBERT index structure at: {default_exp_index_path}")
        # Remove the specific index dir within the experiment structure
        shutil.rmtree(default_exp_index_path)
        # Optionally remove parent dirs if empty, but safer to leave them
    # --- End check ---

    try:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPUs.")
    except:
        num_gpus = 0
        print("WARN: Could not detect GPUs, defaulting to CPU indexing.")

    # --- Use 'root' in ColBERTConfig ---
    # Let ColBERT manage its experiment structure within COLBERT_BASE_PATH
    config = ColBERTConfig(
        root=COLBERT_BASE_PATH,          # <<< Use root for experiment structure base
        index_name=COLBERT_INDEX_NAME, # Still needed to name the final index dir within experiments/...
        checkpoint=COLBERT_MODEL_NAME,
        nbits=2,
        gpus=num_gpus,
        doc_maxlen=220,
        query_maxlen=32
        # Remove index_root if present
    )
    # --- End Modification ---

    print(f"Initializing ColBERT Indexer with config: {config}")
    indexer = Indexer(checkpoint=COLBERT_MODEL_NAME, config=config) # Pass config

    try:
        print("Starting ColBERT indexing process (this may take a while)...")
        # Pass the index name (used relative to root/experiment/indexes)
        indexer.index(name=COLBERT_INDEX_NAME, collection=collection)
        print("ColBERT indexing complete.")

        # --- 3. Save the Metadata Map (Now contains content+metadata) --- <<< MODIFIED
        print(f"Saving enhanced metadata map to {METADATA_MAP_PATH}...")
        with open(METADATA_MAP_PATH, "wb") as f:
            pickle.dump(metadata_map, f)
        print("Metadata map saved.")
        # --- End Modification ---

        # --- 4. Save PID to DocID Mapping (Unchanged) ---
        pid_docid_map = {pid: f"doc_{pid}" for pid in range(len(collection))}
        print(f"Saving PID-to-DocID map to {PID_DOCID_MAP_PATH}...")
        with open(PID_DOCID_MAP_PATH, "wb") as f:
            pickle.dump(pid_docid_map, f)
        print("PID-to-DocID map saved.")

        return True

    except Exception as e:
        print(f"ERROR: ColBERT indexing failed: {e}")
        traceback.print_exc()
        return False


# --- Main Indexing Logic ---
if __name__ == "__main__":
    print("Starting ColBERT indexing process...")

    # --- 1. Load and process documents ---
    print(f"\nLoading documents from: {PDF_DIR_ABS}")
    final_chunks = load_all_documents_for_indexing(PDF_DIR_ABS)

    if not final_chunks:
        print("ERROR: No documents were processed. Exiting.")
        sys.exit(1)

    # --- 2. Build ColBERT Index ---
    print("\nBuilding ColBERT index...")
    success = build_colbert_index(final_chunks)

    if not success:
        print("ERROR: ColBERT index creation failed.")
        sys.exit(1)

    # --- 3. Update detected_customers.txt (Unchanged) ---
    print(f"\nUpdating customer list file: {CUSTOMER_LIST_PATH}")
    detected_customers = set()
    for doc in final_chunks:
        # Extract customer name from metadata, default to "Unknown Customer"
        customer_name = doc.metadata.get('customer', 'Unknown Customer')
        # Add to set if it's a valid name (not empty and not the default unknown)
        if customer_name and customer_name != 'Unknown Customer':
            detected_customers.add(customer_name)

    if detected_customers:
        # Sort the unique names alphabetically for consistency
        sorted_customers = sorted(list(detected_customers))
        try:
            # Open the file in write mode ('w'), overwriting existing content
            with open(CUSTOMER_LIST_PATH, 'w') as f:
                for name in sorted_customers:
                    f.write(name + '\n') # Write each name on a new line
            print(f"Successfully updated {CUSTOMER_LIST_FILE} with {len(sorted_customers)} unique customer names:")
            for name in sorted_customers: print(f"  - {name}") # Log the names written
        except Exception as e:
            print(f"ERROR writing to {CUSTOMER_LIST_FILE}: {e}")
            traceback.print_exc()
            # Decide if this error should stop the whole process
            # sys.exit(1)
    else:
        print(f"WARN: No valid customer names found in processed documents. {CUSTOMER_LIST_FILE} not updated.")
        # Optionally clear the file if no customers are found
        try:
            with open(CUSTOMER_LIST_PATH, 'w') as f:
                pass # Creates an empty file or clears existing one
            print(f"Cleared content of {CUSTOMER_LIST_FILE} as no customers were detected.")
        except Exception as e:
            print(f"ERROR clearing {CUSTOMER_LIST_FILE}: {e}")


    print("\nColBERT Indexing process complete.")