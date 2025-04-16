# routes.py
import time
from flask import Blueprint, render_template, request, jsonify
import langchain_utils.qa_chain as qa_module
from langchain_utils.qa_chain import get_detected_customer_names
import markdown
from langchain_core.callbacks.manager import CallbackManager
try:
    from email_tracer import EmailLangChainTracer
except ImportError:
    print("WARN: email_tracer.py not found. Using basic CallbackManager.")
    class EmailLangChainTracer:
        def __init__(self, project_name): pass # Dummy init
import re
import sys
import traceback
import itertools # <<< Need itertools for select_smart_chunks
# import torch # No longer needed here
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any, Set, Optional
import os

# --- Config Imports ---
# Import necessary config values including the new threshold variables
from config import (
    PROJECT_NAME,
    COLBERT_TOP_K, # Use for ColBERT initial retrieval
    COLBERT_SCORE_THRESHOLD_ENABLED,        # <<< ADDED
    COLBERT_SINGLE_CUSTOMER_SCORE_THRESHOLD,# <<< ADDED
    COLBERT_GENERAL_SCORE_THRESHOLD,        # <<< ADDED
    COLBERT_PASS_2_SCORE_THRESHOLD,         # <<< ADDED (for smart chunks)
    DYNAMIC_K_MIN_CHUNKS,                   # <<< ADDED
    DYNAMIC_K_MAX_CHUNKS                    # <<< ADDED
)
print("--- Imported settings from config ---")

main_blueprint = Blueprint("main", __name__)

# --- Helpers (generate_keyword, get_customer_filter_keyword) (Keep as is) ---
def generate_keyword(customer_name):
    if not customer_name: return None
    name_cleaned = customer_name
    suffixes = [' Pty Ltd', ' Pty Limited', ' Ltd', ' Limited', ' Inc']
    for suffix in suffixes:
        if name_cleaned.lower().endswith(suffix.lower()):
            name_cleaned = name_cleaned[:-len(suffix)].strip()
            break
    parts = name_cleaned.split()
    return parts[0].lower() if parts else None

def get_customer_filter_keyword(query) -> Tuple[Optional[str], List[str]]:
    detected_names = get_detected_customer_names()
    found_original_names: List[str] = []
    if not detected_names:
        print("DEBUG [Filter]: No detected customer names loaded. Cannot filter.")
        return None, found_original_names
    customer_keywords_map = {}
    for name in detected_names:
        keyword = generate_keyword(name)
        if keyword:
            customer_keywords_map[keyword] = name
            full_name_keyword = name.lower().replace(" ", "")
            if full_name_keyword != keyword:
                 customer_keywords_map[full_name_keyword] = name
    query_lower = query.lower()
    sorted_keywords = sorted(customer_keywords_map.keys(), key=len, reverse=True)
    temp_query = query_lower
    for query_keyword in sorted_keywords:
        regex_pattern = rf'\b{re.escape(query_keyword)}\b'
        match = re.search(regex_pattern, temp_query)
        if match:
            original_name = customer_keywords_map[query_keyword]
            if original_name not in found_original_names:
                 found_original_names.append(original_name)
                 temp_query = temp_query[:match.start()] + "_"*len(query_keyword) + temp_query[match.end():]
    filter_name = None
    if len(found_original_names) == 1:
        filter_name = found_original_names[0]
        print(f"DEBUG [Filter]: SUCCESS - Identified single customer: '{filter_name}'")
    elif len(found_original_names) > 1:
        print(f"DEBUG [Filter]: Multiple customers found ({found_original_names}). Will use comparative logic.")
    else:
        print("DEBUG [Filter]: No specific customer detected.")
    return filter_name, list(set(found_original_names))
# --- End Helper ---


# --- Smart Selection Function Definition --- <<< ADDED
def select_smart_chunks(
    ranked_docs_with_scores: List[Tuple[Document, float]], # Input contains ColBERT scores
    target_customers: List[str],
    max_chunks: int, # Overall max limit (DYNAMIC_K_MAX_CHUNKS)
    pass_2_threshold_enabled: bool, # <<< ADDED parameter
    pass_2_score_threshold: float,  # <<< ADDED parameter
    min_docs_per_target: int = 2
) -> List[Document]:
    """
    Selects up to max_chunks documents using Round-Robin for comparative queries.
    Pass 1 ensures minimum coverage.
    Pass 2 fills remaining slots using Round-Robin, applying a score threshold if enabled.
    """
    if not ranked_docs_with_scores:
        print("DEBUG [SelectChunks]: No ranked documents provided.")
        return []

    target_customer_list = sorted(list(set(target_customers)))
    is_comparative = len(target_customer_list) > 1

    # This function is designed for comparative queries
    if not is_comparative:
         print("ERROR [SelectChunks]: This function should only be called for comparative queries.")
         selected_docs_list = [doc for doc, score in ranked_docs_with_scores[:max_chunks]]
         return selected_docs_list

    # --- Comparative Case ---
    pass_2_threshold_value_str = f"{pass_2_score_threshold:.4f}" if pass_2_threshold_enabled else 'Disabled'
    print(f"DEBUG [SelectChunks]: Comparative query. Selecting smart chunks. Targets: {target_customer_list}, Max: {max_chunks}, Min per target: {min_docs_per_target}, Pass2 Threshold: {pass_2_threshold_value_str}")

    selected_docs_list: List[Document] = []
    customer_doc_counts: Dict[str, int] = {cust: 0 for cust in target_customer_list}
    docs_added_keys: Set[str] = set() # Track added docs using unique key

    # --- Pass 1: Ensure minimum coverage using the globally sorted list (by ColBERT score) ---
    print(f"  DEBUG [SelectChunks]: Pass 1 - Ensuring up to {min_docs_per_target} docs per target...")
    for doc, score in ranked_docs_with_scores:
        if len(selected_docs_list) >= max_chunks: break

        if not hasattr(doc, 'metadata') or 'customer' not in doc.metadata:
            print(f"WARN [SelectChunks P1]: Skipping doc without customer metadata: {doc.page_content[:50]}...")
            continue

        customer = doc.metadata.get('customer', 'Unknown Customer')
        if customer in target_customer_list and customer_doc_counts[customer] < min_docs_per_target:
            source = doc.metadata.get('source', 'UnknownSource')
            page = doc.metadata.get('page_number', 'UnknownPage')
            try: content_hash = hash(doc.page_content)
            except TypeError: content_hash = hash(str(doc.page_content))
            doc_key = f"{source}-{page}-{content_hash}"

            if doc_key not in docs_added_keys:
                selected_docs_list.append(doc)
                docs_added_keys.add(doc_key)
                customer_doc_counts[customer] += 1
                print(f"    DEBUG [SelectChunks P1]: Added doc for '{customer}' (Overall Rank: {len(selected_docs_list)}, ColBERT Score: {score:.4f})")

        all_targets_met_min = all(count >= min_docs_per_target for count in customer_doc_counts.values())
        if all_targets_met_min:
             print(f"  DEBUG [SelectChunks]: Minimum doc count ({min_docs_per_target}) met for all targets in Pass 1.")
             break

    print(f"  DEBUG [SelectChunks]: Docs selected after Pass 1: {len(selected_docs_list)}")
    print(f"  DEBUG [SelectChunks]: Customer counts after Pass 1: {customer_doc_counts}")

    # --- Pass 2: Fill remaining slots using Round-Robin with Threshold ---
    if len(selected_docs_list) < max_chunks:
        print(f"  DEBUG [SelectChunks]: Pass 2 - Filling remaining {max_chunks - len(selected_docs_list)} slots using Round-Robin (Threshold: {pass_2_threshold_value_str})...")

        # Create customer-specific lists (as before)
        customer_specific_ranked_docs: Dict[str, List[Tuple[Document, float]]] = {cust: [] for cust in target_customer_list}
        for doc, score in ranked_docs_with_scores:
            if not hasattr(doc, 'metadata') or 'customer' not in doc.metadata:
                continue
            customer = doc.metadata.get('customer', 'Unknown Customer')
            if customer in target_customer_list:
                customer_specific_ranked_docs[customer].append((doc, score))

        customer_next_candidate_index: Dict[str, int] = {cust: 0 for cust in target_customer_list}
        customer_cycle = itertools.cycle(target_customer_list)
        num_targets = len(target_customer_list)
        attempts_since_last_add = 0

        while len(selected_docs_list) < max_chunks and attempts_since_last_add < num_targets:
            current_customer = next(customer_cycle)
            customer_list = customer_specific_ranked_docs.get(current_customer, [])
            start_index = customer_next_candidate_index[current_customer]
            found_doc_for_customer = False

            for i in range(start_index, len(customer_list)):
                doc, score = customer_list[i]
                if not hasattr(doc, 'metadata'): continue

                # --- Apply threshold check if enabled --- <<< MODIFIED
                if pass_2_threshold_enabled and score < pass_2_score_threshold:
                    # print(f"    DEBUG [SelectChunks RR]: Skipping Pass 2 doc (Cust: {current_customer}, Score: {score:.4f} < {pass_2_score_threshold})") # Verbose log
                    # Since the list is sorted, all subsequent docs for this customer might also be below threshold
                    # Mark as exhausted for this round and move to next customer
                    customer_next_candidate_index[current_customer] = len(customer_list)
                    break # Stop checking this customer's list for this cycle turn

                # --- Check if already added ---
                source = doc.metadata.get('source', 'UnknownSource')
                page = doc.metadata.get('page_number', 'UnknownPage')
                try: content_hash = hash(doc.page_content)
                except TypeError: content_hash = hash(str(doc.page_content))
                doc_key = f"{source}-{page}-{content_hash}"

                if doc_key not in docs_added_keys:
                    selected_docs_list.append(doc)
                    docs_added_keys.add(doc_key)
                    customer_next_candidate_index[current_customer] = i + 1
                    found_doc_for_customer = True
                    attempts_since_last_add = 0
                    print(f"    DEBUG [SelectChunks RR]: Added doc via Pass 2 for '{current_customer}' (Overall Rank: {len(selected_docs_list)}, ColBERT Score: {score:.4f})")
                    break # Found doc for this turn

            if not found_doc_for_customer:
                # This means we either exhausted the list or hit the threshold break above
                attempts_since_last_add += 1
            else:
                 if len(selected_docs_list) >= max_chunks:
                     break

        if attempts_since_last_add >= num_targets:
            print(f"  DEBUG [SelectChunks RR]: Pass 2 stopped early. No more eligible documents found for any target customer (or threshold not met).")

    # Note: Min/Max limits are applied *after* this function returns in the main route
    print(f"DEBUG [SelectChunks]: Selected chunk count before final limits: {len(selected_docs_list)}")
    return selected_docs_list
# --- End Smart Selection ---


@main_blueprint.route("/", methods=["GET", "POST"])
def home():
    query_for_template = ""
    answer_for_template = ""
    sources_for_template = None
    email_for_template = ""

    if request.method == "POST":
        request_start_time = time.time()

        debug_retrieval = False
        user_query = ""
        user_email = ""

        # Handle JSON or Form data (keep as is)
        if request.is_json:
            try:
                data = request.get_json()
                user_query = data.get("query", "").strip()
                user_email = data.get("email", "").strip()
                debug_retrieval = data.get("debug_retrieval", False) is True
            except Exception as e:
                 print(f"ERROR: Failed to parse JSON request body: {e}")
                 return jsonify({"error": "Invalid JSON payload"}), 400
        else:
            user_query = request.form.get("query", "").strip()
            user_email = request.form.get("email", "").strip()
            debug_retrieval = request.form.get("debug_retrieval") == "true"


        query_for_template = user_query
        email_for_template = user_email

        answer = "An error occurred processing your request."
        sources = []

        print(f"\n--- NEW REQUEST ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        print(f"User Query: {user_query}")
        print(f"User Email: {user_email}")

        # --- Simplified Readiness Check ---
        colbert_ready = qa_module.colbert_searcher is not None and qa_module.colbert_metadata_map is not None and qa_module.colbert_pid_docid_map is not None
        map_reduce_ready = qa_module.map_reduce_chain is not None

        print(f"ColBERT Searcher Ready: {colbert_ready}")
        print(f"MapReduce Chain Ready: {map_reduce_ready}")

        if not user_query:
            # [ ... handle empty query ... ]
            answer = "Please enter a valid query."
            sources = None
            if request.is_json: return jsonify({"error": answer}), 400
            else:
                answer_for_template = answer; sources_for_template = sources
                return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template), 400


        if not colbert_ready or not map_reduce_ready:
             error_details = []
             if not colbert_ready: error_details.append("ColBERT components missing.")
             if not map_reduce_ready: error_details.append("MapReduce chain missing.")
             error_msg = f"System not ready. Details: {' '.join(error_details)} Please check initialization and config."
             print(f"ERROR: {error_msg}")
             if request.is_json: return jsonify({"error": error_msg}), 500
             else:
                 answer_for_template = f"Error: {error_msg}"; sources_for_template = None
                 return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template), 500

        # Setup callbacks (keep as is)
        try:
            tracer = EmailLangChainTracer(project_name=PROJECT_NAME)
            callback_manager = CallbackManager([tracer])
        except NameError:
             callback_manager = CallbackManager([])
        except Exception as e:
            print(f"Error initializing tracer: {e}")
            callback_manager = CallbackManager([])


        # --- Get filter name (can be used post-retrieval if needed) ---
        filter_customer_name, target_customers_in_query = get_customer_filter_keyword(user_query)

        t_start_pipeline = time.time()
        t_last = t_start_pipeline

        try:
            # --- ColBERT RETRIEVAL ---
            # Use COLBERT_TOP_K for initial retrieval
            print(f"Querying ColBERT index with k={COLBERT_TOP_K}...")
            docs_to_process: List[Document] = [] # Initialize docs_to_process here
            retrieved_docs_with_scores: List[Tuple[Document, float]] = [] # Store results with scores

            try:
                searcher = qa_module.colbert_searcher
                metadata_map = qa_module.colbert_metadata_map
                pid_docid_map = qa_module.colbert_pid_docid_map

                print(f"DEBUG: Performing ColBERT search for query: '{user_query}' with k={COLBERT_TOP_K}")
                results_pids, _, results_scores = searcher.search(user_query, k=COLBERT_TOP_K) # Use COLBERT_TOP_K
                print(f"DEBUG: ColBERT search returned {len(results_pids)} PIDs.")

                # Retrieve content and metadata (as before)
                if results_pids:
                    for i, pid in enumerate(results_pids):
                        doc_id = pid_docid_map.get(pid)
                        if not doc_id: continue
                        combined_data = metadata_map.get(doc_id)
                        if not combined_data: continue
                        page_content = combined_data.get('page_content', '')
                        original_metadata = combined_data.get('metadata', {})
                        if not page_content: continue
                        metadata_with_score = original_metadata.copy()
                        score = float(results_scores[i]) if results_scores and i < len(results_scores) else 0.0
                        metadata_with_score["colbert_score"] = score
                        metadata_with_score["colbert_rank"] = i + 1
                        doc = Document(
                            page_content=page_content,
                            metadata={k: v for k, v in metadata_with_score.items() if v is not None}
                        )
                        retrieved_docs_with_scores.append((doc, score)) # Store tuple

            except Exception as colbert_err:
                print(f"ERROR during ColBERT search or processing: {colbert_err}")
                traceback.print_exc()
                retrieved_docs_with_scores = [] # Ensure list is empty on error

            t_now = time.time(); print(f"DEBUG Timing: ColBERT Search & Map took {t_now-t_last:.4f}s"); t_last = t_now

            # --- Document Selection Logic --- <<< MODIFIED
            # Determine the final list of documents to send to the LLM
            selected_docs_list_before_limits: List[Document] = []

            if filter_customer_name:
                # Case 1: Single customer filter
                print(f"DEBUG: Applying post-retrieval filter for single customer: '{filter_customer_name}'")
                candidates_for_processing = [
                    (doc, score) for doc, score in retrieved_docs_with_scores
                    if hasattr(doc, 'metadata') and doc.metadata.get('customer') == filter_customer_name
                ]
                # Apply threshold if enabled
                if COLBERT_SCORE_THRESHOLD_ENABLED:
                    threshold = COLBERT_SINGLE_CUSTOMER_SCORE_THRESHOLD # Use single customer threshold
                    print(f"DEBUG: Applying score threshold ({threshold}) to single customer results...")
                    candidates_for_processing = [
                        (doc, score) for doc, score in candidates_for_processing
                        if score >= threshold
                    ]
                selected_docs_list_before_limits = [doc for doc, score in candidates_for_processing]
                print(f"DEBUG: Found {len(selected_docs_list_before_limits)} documents for single customer post-threshold.")

            elif len(target_customers_in_query) > 1:
                # Case 2: Comparative query - Use Smart Chunk Selection
                print(f"DEBUG: Applying smart chunk selection for multiple customers: {target_customers_in_query}")
                # Filter first to get candidates for the target customers
                comparative_candidates = [
                    (doc, score) for doc, score in retrieved_docs_with_scores
                    if hasattr(doc, 'metadata') and doc.metadata.get('customer') in target_customers_in_query
                ]
                print(f"DEBUG: Found {len(comparative_candidates)} candidates matching target customers.")

                # Call select_smart_chunks, passing the overall max limit and threshold info
                selected_docs_list_before_limits = select_smart_chunks(
                     ranked_docs_with_scores=comparative_candidates,
                     target_customers=target_customers_in_query,
                     max_chunks=DYNAMIC_K_MAX_CHUNKS, # Pass overall max limit
                     pass_2_threshold_enabled=COLBERT_SCORE_THRESHOLD_ENABLED, # <<< Pass enable flag
                     pass_2_score_threshold=COLBERT_PASS_2_SCORE_THRESHOLD,   # <<< Pass threshold value
                     # min_docs_per_target=2 # Use default or configure
                 )
                # The function itself prints the final count

            else:
                # Case 3: No specific customer - Apply threshold if enabled
                print("DEBUG: No specific customer filter applied.")
                candidates_for_processing = retrieved_docs_with_scores
                if COLBERT_SCORE_THRESHOLD_ENABLED:
                     threshold = COLBERT_GENERAL_SCORE_THRESHOLD # Use general threshold
                     print(f"DEBUG: Applying score threshold ({threshold}) to general results...")
                     candidates_for_processing = [
                         (doc, score) for doc, score in candidates_for_processing
                         if score >= threshold
                     ]
                selected_docs_list_before_limits = [doc for doc, score in candidates_for_processing]
                print(f"DEBUG: Found {len(selected_docs_list_before_limits)} documents post-threshold (no customer filter).")

            # --- Apply Final Min/Max Limits ---
            final_selected_count = len(selected_docs_list_before_limits)
            if final_selected_count < DYNAMIC_K_MIN_CHUNKS:
                 print(f"WARN: Selected docs ({final_selected_count}) less than MIN_CHUNKS ({DYNAMIC_K_MIN_CHUNKS}). Selecting top {DYNAMIC_K_MIN_CHUNKS} overall instead.")
                 # Fallback: take the absolute top MIN_CHUNKS from the initial retrieved list
                 docs_to_process = [doc for doc, score in retrieved_docs_with_scores[:DYNAMIC_K_MIN_CHUNKS]]
            elif final_selected_count > DYNAMIC_K_MAX_CHUNKS:
                 print(f"WARN: Selected docs ({final_selected_count}) more than MAX_CHUNKS ({DYNAMIC_K_MAX_CHUNKS}). Truncating.")
                 docs_to_process = selected_docs_list_before_limits[:DYNAMIC_K_MAX_CHUNKS] # Truncate
            else:
                 # Count is within min/max bounds
                 docs_to_process = selected_docs_list_before_limits

            print(f"DEBUG: Final document count sent to LLM: {len(docs_to_process)}")
            t_now = time.time(); print(f"DEBUG Timing: Document Selection took {t_now-t_last:.4f}s"); t_last = t_now
            # --- END Document Selection Logic ---


            # --- Chain Execution ---
            # [ ... rest of the code remains the same ... ]
            if not docs_to_process:
                 # Update error message slightly
                 if filter_customer_name:
                     answer = f"Could not find relevant documents via ColBERT specifically for '{filter_customer_name}' matching your query."
                 elif len(target_customers_in_query) > 1:
                      answer = f"Could not find relevant documents via ColBERT for customers '{', '.join(target_customers_in_query)}' matching your query after filtering/selection." # Updated msg
                 elif not answer or answer == "An error occurred processing your request.":
                    answer = f"Could not find relevant documents via ColBERT for your query."
                 sources = []
                 t_now = time.time(); print(f"DEBUG Timing: MapReduce Chain skipped (No docs)"); t_last = time.time() # Fix t_last update
            else:
                # Prepare documents for MapReduce (same logic as before)
                processed_docs_for_map = []
                for doc in docs_to_process: # Use the final selected results
                    # !!! Ensure doc is a Document object with metadata here !!!
                    if not isinstance(doc, Document) or not hasattr(doc, 'metadata'):
                        print(f"WARN: Skipping item in docs_to_process that is not a valid Document with metadata: {type(doc)}")
                        continue
                    meta = doc.metadata
                    header_parts = [
                        f"Source: {meta.get('source', 'Unknown')}",
                        f"Page: {meta.get('page_number', 'N/A')}",
                        f"Customer: {meta.get('customer', 'Unknown')}",
                        f"Clause: {meta.get('clause', 'N/A')}",
                    ]
                    header = " | ".join(header_parts) + "\n---\n"
                    content = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
                    cleaned_content = re.sub(r'^```[a-zA-Z]*\n', '', content, flags=re.MULTILINE)
                    cleaned_content = re.sub(r'\n```$', '', cleaned_content, flags=re.MULTILINE)
                    processed_docs_for_map.append( Document(page_content=header + cleaned_content.strip(), metadata=meta) )

                if not processed_docs_for_map:
                     answer = "Error: No valid documents remained after metadata check."
                     sources = []
                     t_now = time.time(); print(f"DEBUG Timing: MapReduce Chain skipped (No valid docs after check)"); t_last = time.time() # Fix t_last update
                else:
                    chain_input = { "input_documents": processed_docs_for_map, "question": user_query }
                    print(f"DEBUG: Invoking MapReduce chain with {len(processed_docs_for_map)} documents...")
                    try:
                        chain_config = {"callbacks": callback_manager}
                        if user_email: chain_config["metadata"] = {"user_email": user_email}
                        result = qa_module.map_reduce_chain.invoke(chain_input, config=chain_config)
                        answer_raw = result.get("output_text", "Error: Could not generate answer from MapReduce chain.")
                        answer = answer_raw
                    except Exception as e:
                         print(f"Error invoking MapReduce chain: {e}"); traceback.print_exc()
                         answer = "Error processing query via MapReduce chain."
                    t_now = time.time(); print(f"DEBUG Timing: MapReduce Chain ({len(processed_docs_for_map)} docs) took {t_now-t_last:.4f}s"); t_last = time.time() # Fix t_last update

            # --- Source Generation (Input is now docs_to_process) ---
            sources = []
            seen_sources = set()
            for doc in docs_to_process: # Use the final selected results
                # !!! Ensure doc is a Document object with metadata here !!!
                if not isinstance(doc, Document) or not hasattr(doc, 'metadata'):
                    continue # Skip if not valid
                meta = doc.metadata
                source_file = meta.get('source', 'Unknown Source')
                page_num = meta.get('page_number', 'N/A')
                customer_display = meta.get('customer', 'Unknown Customer')
                source_key = f"{source_file}|Page {page_num}"

                # Add ColBERT rank/score to source string for info
                source_str = f"{source_file} (Cust: {customer_display}, Pg: {page_num}"
                colbert_rank = meta.get('colbert_rank')
                colbert_score = meta.get('colbert_score')
                if colbert_rank is not None:
                    source_str += f", Rank: {colbert_rank}"
                if colbert_score is not None:
                     source_str += f", Score: {colbert_score:.3f}" # Format score
                source_str += ")"

                # Add clause/hierarchy info if available
                clause_display = meta.get('clause', None)
                hierarchy_list = meta.get('hierarchy', [])
                if clause_display and clause_display != 'N/A':
                    source_str += f" (Clause: {clause_display})"
                elif hierarchy_list and isinstance(hierarchy_list, list) and hierarchy_list:
                     try: source_str += f" (Section: {hierarchy_list[-1]})"
                     except IndexError: pass

                sources.append(source_str)
                # seen_sources.add(source_key) # Maybe remove this if you want all sources listed

            # --- Final Formatting (Unchanged) ---
            is_error_answer = "Error:" in answer or "Could not find" in answer or "System not ready" in answer
            if not is_error_answer:
                if not isinstance(answer, str): answer = str(answer)
                try: answer = markdown.markdown(answer, extensions=['fenced_code', 'tables', 'nl2br'])
                except Exception as md_err: print(f"WARN: Markdown conversion failed: {md_err}. Returning raw answer.")
            elif not isinstance(answer, str): answer = str(answer)
            t_now = time.time(); print(f"DEBUG Timing: Source Gen/Formatting took {t_now-t_last:.4f}s"); t_last = time.time() # Fix t_last update


        except Exception as e:
             print(f"Error during main processing block: {e}")
             traceback.print_exc()
             answer = "An unexpected error occurred while processing your query."
             sources = []

        # --- Return Response ---
        answer_for_template = answer
        sources_for_template = sources

        total_request_time = time.time() - request_start_time
        print(f"--- Request Complete --- Total Time: {total_request_time:.4f}s ---")

        if request.is_json: return jsonify({"answer": answer, "sources": sources})
        else: return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template)

    # --- GET request (Unchanged) ---
    return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template)