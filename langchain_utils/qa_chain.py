# langchain_utils/qa_chain.py
import os
import sys
import pickle
import traceback
import copy
from typing import Any, Dict, List, Optional, Sequence, Union

# --- LangChain Imports (Keep) ---
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug

# --- ColBERT Imports ---
try:
    from colbert import Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
except ImportError:
    print("ERROR: colbert-ai library not found. ColBERT search will be disabled.")
    print("Please install it: pip install colbert-ai[faiss-cpu,torch]")
    Searcher = None
    ColBERTConfig = None

# --- Local Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Updated Config Imports ---
from config import (
    TEMPERATURE, MAX_TOKENS, PROJECT_NAME,
    # ColBERT Config
    COLBERT_INDEX_ROOT, COLBERT_INDEX_NAME,
    COLBERT_METADATA_MAP_FILENAME, COLBERT_PID_DOCID_MAP_FILENAME,
    # Azure Config (Keep)
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME
)

# --- System Prompt Import (Keep) ---
try:
    from system_prompt import system_prompt
    print("--- Successfully imported system_prompt from system_prompt.py ---")
except ImportError:
    print("--- WARNING: Could not import system_prompt. Using a basic default for Reduce step. ---")
    system_prompt = """
You are a helpful AI assistant. Synthesize the provided summaries to answer the question based *only* on the summaries. Attribute information using the metadata (Source, Customer, Clause). If the question asks for specific information (e.g., 'chilled temperature'), only include data explicitly matching that criteria from the summaries. If no matching information is found for a requested entity, state that clearly.
""" # Keep fallback


# --- Global Variables ---
map_reduce_chain: Optional[MapReduceDocumentsChain] = None
llm_instance: Optional[AzureChatOpenAI] = None
detected_customer_names: List[str] = []
CUSTOMER_LIST_FILE = "detected_customers.txt"

# --- ColBERT Globals ---
colbert_searcher: Optional[Searcher] = None
colbert_metadata_map: Dict[str, Dict] = {}
colbert_pid_docid_map: Dict[int, str] = {}

# --- Check Azure Credentials (Keep) ---
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    print("CRITICAL ERROR in qa_chain.py: Azure OpenAI credentials missing in config. LLM initialization will fail.")


# --- MapReduce Chain Setup (Keep as is) ---
def setup_map_reduce_chain() -> MapReduceDocumentsChain:
    global llm_instance
    if llm_instance is None:
        if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
             raise ValueError("Azure OpenAI credentials not found in config. Cannot initialize LLM.")
        try:
            print(f"Initializing AzureChatOpenAI (Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME})...")
            llm_instance = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            print("AzureChatOpenAI initialized successfully.")
        except Exception as e:
             print(f"ERROR initializing AzureChatOpenAI: {e}")
             traceback.print_exc()
             sys.exit(1)

    llm = llm_instance
    map_template = """
You will be provided with a document excerpt preceded by its source metadata (Source, Page, Customer, Clause).
Your task is to analyze ONLY the text of the document excerpt BELOW the '---' line.
Based *only* on this excerpt text, identify and extract the exact sentences or key points that are relevant to answering the user's question.

User Question: {question}

Document Excerpt with Metadata:
{page_content}

**Critical Instructions:**
1.  Focus *only* on the text provided in the excerpt *below* the '---' line.
2.  Extract verbatim sentences or concise key points from the excerpt text that *directly provide information relevant to the User Question's topic(s) and entities*.
3.  **Pay special attention to any specific details, entities, figures, conditions, requirements, or keywords mentioned *in the User Question*. Extract the exact text from the excerpt that contains or directly addresses these specifics.**
4.  **Handling Comparative Questions:** If the User Question asks for a comparison between entities (e.g., 'compare A and B about topic X'), your task in *this step* is to extract any information about **topic X** that relates to **either** entity A **or** entity B, *if that information is present in this specific excerpt*. Do not discard relevant information about entity A just because entity B is not mentioned here, or vice-versa. The final comparison synthesis will happen in a later step based on all extractions.
5.  Use the context of the surrounding text in the excerpt to determine relevance, even if the specific keyword from the question isn't in the exact sentence being extracted.
6.  Extract information regardless of formatting (e.g., inside lists, tables, or ```code blocks```).
7.  **Your entire output MUST start with the *exact* metadata line provided above (everything before the '---'), followed by ' --- ', and then either your extracted text OR the specific phrase \"No relevant information found in this excerpt.\"**
8.  If the excerpt text contains NO information relevant *at all* to the **topic(s) or target entities** mentioned in the User Question (considering instruction #4 for comparisons), your output MUST be the metadata line followed by ' --- No relevant information found in this excerpt.'.
9.  Do NOT add explanations, introductions, summaries, or any text other than the required metadata prefix and the extracted relevant text (or the \"No relevant information\" message).
10. Do NOT attempt to answer the overall User Question.

**Output:**
"""
    map_prompt = PromptTemplate(input_variables=["page_content", "question"], template=map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

    reduce_template = f"""
{system_prompt}

Original User Question: {{question}}

Extracted Information Summaries (Metadata --- Content):
{{doc_summaries}}

Based *only* on the summaries above and following all instructions in the initial system prompt (especially regarding strict grounding, requested entities, and specific query terms like 'chilled'), provide the final answer to the Original User Question. If multiple summaries provide relevant details for the same point, synthesize them concisely. If summaries indicate no specific information was found for a requested entity or criteria (e.g., 'chilled temperature for Patties'), explicitly state that in the final answer.

Final Answer:"""
    reduce_prompt = PromptTemplate(input_variables=["doc_summaries", "question"], template=reduce_template)
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="doc_summaries",
        document_separator="\n\n---\n\n",
        verbose=True
    )

    chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=combine_documents_chain,
        document_variable_name="page_content",
        input_key="input_documents",
        output_key="output_text",
        verbose=True
    )
    return chain


# --- Application Initialization ---
def initialize_app():
    """Initializes ColBERT searcher, LLM chain, and detected customer names."""
    global map_reduce_chain, detected_customer_names, llm_instance
    global colbert_searcher, colbert_metadata_map, colbert_pid_docid_map
    set_debug(True)

    print("Initializing application components...")

    # --- Load ColBERT Index and Mappings ---
    colbert_base_path = os.path.abspath(os.path.join(project_root, COLBERT_INDEX_ROOT)) # .../LLM_Legal_V55/colbert_indices
    # Maps are still directly in the base path
    metadata_map_path = os.path.join(colbert_base_path, COLBERT_METADATA_MAP_FILENAME)
    pid_docid_map_path = os.path.join(colbert_base_path, COLBERT_PID_DOCID_MAP_FILENAME)

    # --- Path where the index is expected to be built by default ---
    # Includes the default experiment structure relative to the root used during indexing
    expected_nested_index_path = os.path.join(colbert_base_path, "experiments", "default", "indexes", COLBERT_INDEX_NAME) # <<< PATH FOR CHECKING/DEBUGGING

    if not Searcher or not ColBERTConfig:
        print("CRITICAL ERROR: ColBERT library not loaded correctly. Exiting.")
        sys.exit(1)

    # --- MODIFIED CHECKS ---
    # 1. Check if the BASE directory exists (where maps should be)
    print(f"DEBUG: Checking for ColBERT base path: {colbert_base_path}")
    if not os.path.exists(colbert_base_path):
         print(f"CRITICAL ERROR: ColBERT base directory '{colbert_base_path}' not found.")
         print("Please run 'python build_indices.py' first.")
         sys.exit(1)

    # 2. Check if map files exist
    if not os.path.exists(metadata_map_path):
         print(f"CRITICAL ERROR: ColBERT metadata map '{metadata_map_path}' not found.")
         print("Please run 'python build_indices.py' first.")
         sys.exit(1)
    if not os.path.exists(pid_docid_map_path):
         print(f"CRITICAL ERROR: ColBERT PID->DocID map '{pid_docid_map_path}' not found.")
         print("Please run 'python build_indices.py' first.")
         sys.exit(1)

    # 3. Add a check for the nested path just for logging confirmation
    print(f"DEBUG: Also checking expected nested index path: {expected_nested_index_path}")
    if not os.path.exists(expected_nested_index_path):
        # This is now just a warning, as the Searcher might still find it
        print(f"WARN: The expected nested index path '{expected_nested_index_path}' was NOT found by os.path.exists(). Will proceed assuming Searcher can locate it.")

    try:
        print(f"Loading ColBERT metadata map from {metadata_map_path}...")
        with open(metadata_map_path, "rb") as f:
            colbert_metadata_map = pickle.load(f)
        print(f"Loading ColBERT PID->DocID map from {pid_docid_map_path}...")
        with open(pid_docid_map_path, "rb") as f:
            colbert_pid_docid_map = pickle.load(f)
        print("ColBERT maps loaded successfully.")

        print(f"Initializing ColBERT Searcher for index '{COLBERT_INDEX_NAME}' relative to root '{colbert_base_path}'...")
        # --- MODIFIED Searcher Initialization ---
        # Initialize Searcher simply with the index name and the root path.
        # Let the Searcher handle finding the index within the root/experiment structure.
        searcher_root_path = colbert_base_path # The path containing 'experiments' dir
        # Create a minimal config just specifying the root for the Searcher
        config_for_search = ColBERTConfig(
            root=searcher_root_path
            # experiment='default' # Usually not needed if root is correct
        )
        colbert_searcher = Searcher(
            index=COLBERT_INDEX_NAME,
            config=config_for_search # Pass the config specifying the root
        )
        # --- End Modification ---
        print("ColBERT Searcher initialized successfully.")

    except Exception as e:
        # If Searcher init fails, it might provide a more specific error
        print(f"CRITICAL ERROR initializing ColBERT Searcher or loading maps: {e}")
        # Check if the error message indicates the index wasn't found by the Searcher itself
        if "Missing:" in str(e) or "directory not found" in str(e).lower() or "AssertionError: index_path" in str(e):
             print(f"-> Searcher failed to find index components for '{COLBERT_INDEX_NAME}' within '{colbert_base_path}'.")
             print(f"-> Expected structure might be: '{expected_nested_index_path}'")
             print(f"-> Ensure 'build_indices.py' completed successfully and created the index there.")
        traceback.print_exc()
        sys.exit(1)

    # --- Load Detected Customer Names (Keep as is) ---
    try:
        customer_list_path = os.path.join(project_root, CUSTOMER_LIST_FILE)
        with open(customer_list_path, "r") as f:
            names = sorted(list(set(line.strip() for line in f if line.strip() and line.strip() != "Unknown Customer")))
            detected_customer_names = names
        print(f"Loaded {len(detected_customer_names)} unique customer names from {customer_list_path}.")
    except FileNotFoundError:
        print(f"WARN: {CUSTOMER_LIST_FILE} not found at {project_root}. Customer name list is empty.")
        detected_customer_names = []
    except Exception as e:
        print(f"Error loading {CUSTOMER_LIST_FILE}: {e}")
        detected_customer_names = []

    # --- Chain Setup (Keep as is) ---
    try:
        map_reduce_chain = setup_map_reduce_chain()
        print("MapReduce chain initialized successfully (using Azure OpenAI).")
    except Exception as e:
        print(f"CRITICAL ERROR setting up MapReduce chain: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Application initialization complete.")

# --- Function to get detected names (Keep as is) ---
def get_detected_customer_names() -> List[str]:
    """Returns the list of customer names detected during initialization."""
    global detected_customer_names
    return detected_customer_names

# --- Direct Execution Test Block (Optional - Unchanged) ---
# if __name__ == '__main__':
#     print("Running qa_chain.py directly...")
#     # Add any test logic here if needed, e.g., initialize and test chain components
#     try:
#         initialize_app()
#         print("Initialization successful in direct run.")
#         # Example test:
#         if map_reduce_chain and retriever:
#             print("Chain and retriever seem loaded.")
#         else:
#             print("Chain or retriever failed to load.")
#     except SystemExit:
#         print("Initialization failed, exiting.")
#     except Exception as e:
#         print(f"An error occurred during direct run test: {e}")