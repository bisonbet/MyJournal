#!/usr/bin/env python3
import requests
import argparse
import os
import json
import tiktoken # For token counting for chunking
import re

# --- Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "cogito:32b"

MODELS_TO_RUN_LIST = [
    "cogito:32b",
    "qwen3:30b-a3b",               
    "phi4-mini:latest",
    "granite3.3:8b"                 
]

# Context window sizes for different summarization passes
DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX = 4096
DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX = 8192
DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX = 16000 # As per user request for final summary

# Max new tokens to generate for each pass's output
DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS = 750   # Output of initial pass, input to intermediate
DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS = 2500 # Output of intermediate pass, input to final
DEFAULT_FINAL_PASS_MAX_NEW_TOKENS = 5000    # Final summary output

# Other chunking parameters
DEFAULT_TARGET_CHUNK_TOKENS_RATIO = 0.6 # Ratio of num_ctx to use for text content in a chunk
DEFAULT_TOKEN_OVERLAP = 200
TIKTOKEN_ENCODING_FOR_CHUNKNG = "o200k_base"
BUFFER_FOR_PROMPT_AND_GENERATION_MARGIN = 300 # General buffer for prompt template and generation margin


def get_tokenizer_for_counting():
    """Returns a tiktoken tokenizer for estimating token counts for chunking."""
    try:
        return tiktoken.get_encoding(TIKTOKEN_ENCODING_FOR_CHUNKNG)
    except Exception as e:
        print(f"Could not load tiktoken encoding '{TIKTOKEN_ENCODING_FOR_CHUNKNG}'. Error: {e}")
        print("Please ensure tiktoken is installed correctly.")
        print("Falling back to 'cl100k_base' if 'o200k_base' is unavailable for some reason.")
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as e_fallback:
            print(f"Could not load fallback tiktoken encoding 'cl100k_base'. Error: {e_fallback}")
            raise

def check_ollama_status(ollama_url, debug_mode=False):
    """Checks if the Ollama server is responsive."""
    if debug_mode:
        print(f"Checking Ollama status at {ollama_url}...")
    try:
        response = requests.get(ollama_url, timeout=10)
        response.raise_for_status()
        if debug_mode:
            print("Ollama server is responsive.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        print(f"Please ensure Ollama is running and accessible at '{ollama_url}'.")
        return False

def split_text_into_chunks(text, tokenizer, max_content_tokens_for_chunk, overlap_tokens, debug_mode=False):
    """
    Splits text into chunks.
    'max_content_tokens_for_chunk' is the target token count for the actual text content of the chunk.
    """
    if debug_mode:
        print(f"Splitting text into chunks of target content ~{max_content_tokens_for_chunk} tokens with {overlap_tokens} token overlap...")
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_content_tokens_for_chunk, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end_idx == len(tokens):
            break
        advance_by = max_content_tokens_for_chunk - overlap_tokens
        if advance_by <= 0 : 
            if debug_mode:
                print(f"Warning: Chunk advance step is {advance_by}. Check overlap_tokens and chunk size. Advancing by 1 to prevent stall.")
            advance_by = 1
        start_idx += advance_by
        if start_idx >= len(tokens): 
            break
    if debug_mode:
        print(f"Split into {len(chunks)} chunks.")
    return chunks

def generate_summary_ollama(text_chunk, ollama_url, ollama_model,
                            num_ctx_for_call, max_new_tokens_for_generation, prompt_template,
                            temperature=0.7, top_p=0.9, debug_mode=False):
    """Generates a summary for a text chunk using a specified Ollama model."""
    prompt = prompt_template.format(text_chunk=text_chunk)
    api_url = f"{ollama_url.rstrip('/')}/api/generate"

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": True,  # Enable streaming
        "options": {
            "num_ctx": int(num_ctx_for_call),
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": int(max_new_tokens_for_generation)
        }
    }
    
    try:
        counting_tokenizer = get_tokenizer_for_counting() 
        estimated_prompt_tokens = len(counting_tokenizer.encode(prompt))
        if debug_mode:
            print(f"Ollama Call (Model: {ollama_model}): Num Ctx: {num_ctx_for_call}, Est. Prompt Tokens: ~{estimated_prompt_tokens}, Target Gen. Tokens: {max_new_tokens_for_generation}.")
    except Exception as e_tok:
        if debug_mode:
            print(f"Note: Could not estimate prompt tokens due to tokenizer issue: {e_tok}")
            print(f"Ollama Call (Model: {ollama_model}): Num Ctx: {num_ctx_for_call}, Target Gen. Tokens: {max_new_tokens_for_generation}.")

    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=600)
        response.raise_for_status()
        
        # Initialize empty summary
        summary = ""
        
        # Process the stream
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        chunk = json_response['response']
                        print(chunk, end='', flush=True)  # Print in real-time
                        summary += chunk
                    
                    # Check for errors in the stream
                    if 'error' in json_response:
                        error_msg = json_response['error']
                        print(f"\nError from Ollama: {error_msg}")
                        return f"[Ollama Error: {error_msg}]"
                        
                except json.JSONDecodeError as e:
                    print(f"\nError decoding JSON from stream: {e}")
                    continue

        summary = summary.strip()
        if not summary:
            if debug_mode:
                print("\nWarning: Ollama returned an empty summary.")
            return f"[Ollama returned empty summary]"

        print()  # Add a newline after the summary
        return summary

    except requests.exceptions.HTTPError as http_err:
        error_content = http_err.response.text
        if debug_mode:
            print(f"\nHTTP error occurred with Ollama: {http_err} - {error_content}")
        try:
            error_details = http_err.response.json()
            return f"[Ollama HTTP Error: {http_err}. Details: {error_details.get('error', 'N/A')}]"
        except json.JSONDecodeError:
            return f"[Ollama HTTP Error: {http_err}. Response: {error_content}]"
    except requests.exceptions.RequestException as e:
        if debug_mode:
            print(f"\nError making request to Ollama: {e}")
        return f"[Error connecting to Ollama: {e}]"
    except json.JSONDecodeError as e:
        if debug_mode:
            print(f"\nError decoding Ollama's JSON response: {e}")
        return f"[Error decoding Ollama response: {e}]"

def save_debug_file(original_input_filepath, model_name, step_label, content_to_save):
    """Saves intermediate summaries for debugging with .md extension."""
    input_dir = os.path.dirname(original_input_filepath)
    input_filename_base = os.path.splitext(os.path.basename(original_input_filepath))[0]
    sanitized_model_name = model_name.replace(":", "_").replace("/", "_")
    
    # Construct filename with .md extension
    debug_filename = f"{input_filename_base}-DEBUG-{step_label}-{sanitized_model_name}.md" 
    debug_filepath = os.path.join(input_dir, debug_filename)

    try:
        # Ensure the directory exists (it should, as it's the input file's dir)
        os.makedirs(input_dir, exist_ok=True) 
        with open(debug_filepath, "w", encoding="utf-8") as f:
            f.write(f"# DEBUG FILE (Markdown)\n") 
            f.write(f"# Original Input: {original_input_filepath}\n")
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Step Label: {step_label}\n")
            f.write(f"# -----------------------------------\n\n")
            f.write(content_to_save)
        print(f"Saved debug file: {debug_filepath}")
    except Exception as e:
        print(f"Error saving debug file '{debug_filepath}': {e}")

def remove_thinking_tokens(text):
    """Remove any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def summarize_transcript(full_text, tokenizer_for_counting, ollama_url, ollama_model,
                         # Pass-specific configurations
                         initial_pass_ctx, initial_pass_max_gen,
                         intermediate_pass_ctx, intermediate_pass_max_gen,
                         final_pass_ctx, final_pass_max_gen,
                         # General chunking config
                         target_chunk_tokens_ratio, overlap_tokens,
                         # For debug saving
                         original_input_filepath,
                         debug_mode # New flag
                         ):
    """
    Summarizes a long text using a multi-stage map-reduce approach.
    """

    prompt_template_initial_chunk = """
This is an excerpt from a day's audio transcript. It includes various conversations, spoken content (which may include personal interactions, media like TV shows, or other background audio), and speaker labels where available. Your primary goal is to extract and summarize the most meaningful information relevant to the user's activities, discussions, and commitments.

IMPORTANT FILTERING GUIDELINES:
* Exclude content that is clearly from TV shows, movies, or entertainment media, especially:
  - Crime shows, murder mysteries, or detective stories
  - Dramatic or sensational content that is clearly fictional
  - News broadcasts about distant events not directly relevant to the user
  - Entertainment programming that doesn't relate to the user's actual activities
* Focus on real-world interactions, personal discussions, and actual activities
* If uncertain whether content is from media or real life, prioritize real-world content

Please produce a detailed yet concise summary of this excerpt, specifically focusing on:

* **Key Topics Discussed:** What were the main subjects of conversation or focus? (Excluding entertainment/media content)
* **Important Decisions:** What significant decisions were made or actively considered?
* **Salient Questions Asked:** What key questions were posed that reveal important uncertainties, information needs, or areas for future exploration?
* **Clear Action Items & Tasks:** What specific tasks, assignments, or follow-ups were mentioned or agreed upon? Identify who is responsible if stated.
* **Notable Insights & Reflections:** Were there any significant realizations, "aha" moments, or noteworthy reflections shared by any speaker?
* **Key Information & Facts:** Were any critical pieces of information, data, or facts shared that are important to remember?
* **Specific Reminders & Notes-to-Self:** CRITICALLY IMPORTANT: Identify any phrases where a speaker explicitly states an intention to remember, note, or look something up. This includes phrases like "making a note of X", "make a note to Y", "remind me to Z", "I need to look up A", "let's look up B". Preserve these exact phrases or their direct intent. Clearly label these as "REMINDER:" or "NOTE-TO-SELF:" or "LOOKUP:" in your summary of this excerpt. These must be passed on.

Retain crucial context. Attribute key statements, decisions, questions, action items, and insights to specific speakers whenever possible and relevant for clarity and accountability. If parts of the transcript appear to be background noise, TV, or clearly irrelevant to the user's direct activities or discussions, please minimize their inclusion in the summary unless they directly influence a subsequent interaction. Prioritize the most significant information to maintain conciseness while ensuring all critical details are captured.

Transcript Excerpt:
---
{text_chunk}
---
Summary of Excerpt (including any REMINDER/NOTE-TO-SELF/LOOKUP items):
"""

    prompt_template_combine_summaries = """
You are tasked with synthesizing a series of individual summaries, which represent consecutive segments of a single day's audio transcript. Your goal is to create a single, coherent, and comprehensive overview of the entire day.

The output MUST be in well-formatted Markdown, designed for maximum readability and a visually appealing presentation. Employ elements such as:
* **Main Headings (e.g., ## Daily Overview, ## Key Themes)**
* **Sub-headings (e.g., ### Morning Session, ### Project Alpha Discussion)** where logical.
* **Bullet points (`*` or `-`)** for lists of items (e.g., action items, key decisions).
* **Bold text (`**text**`)** for emphasis on key terms or outcomes.
* **Numbered lists** if the order or sequence is important.

In this synthesized daily summary, please focus on:

* **Overall Narrative & Day's Flow:** Provide a sense of how the day progressed.
* **Overarching Themes:** Identify recurring topics or central themes that spanned multiple conversations or time segments.
* **Most Valuable Ideas & Insights:** Highlight the most significant concepts, conclusions, or realizations from the day.
* **Key Learnings & Takeaways:** What were the primary lessons learned or critical pieces of information that emerged?
* **Progression & Connections:** Show how discussions or projects evolved throughout the day and connect related items from different summaries.
* **Consolidated Key Decisions & Action Items:** Create unified lists of all major decisions made and critical action items, noting any deadlines or responsible parties mentioned.
* **Unresolved Important Matters:** List any significant questions or issues that were raised but not resolved, or topics needing further attention.
* **CRITICAL: Reminders, Notes-to-Self, and Lookups:** If the individual summaries contain items explicitly labeled as "REMINDER:", "NOTE-TO-SELF:", or "LOOKUP:", you MUST consolidate these into a dedicated section in your final output, perhaps under a heading like "### Reminders & Items for Follow-up". Preserve the original intent and as much of the original phrasing as possible for these items. Do not lose these.
* **Redundancy Management:** Synthesize information intelligently. If multiple summaries mention the same event or detail, consolidate it effectively rather than repeating it, ensuring the most complete version is retained.

The aim is to produce a high-level yet sufficiently detailed digest that allows someone to quickly grasp the essence and important outcomes of the day, including all explicitly stated reminders or notes.

IMPORTANT: Your response MUST be only the Markdown summary itself. Do NOT include any commentary, self-critique, analysis of these instructions, praise for the instructions, or suggestions for improvement on your own output. Simply provide the requested Markdown summary of the combined texts.

Combined Summaries:
---
{text_chunk} ---
Comprehensive Daily Summary (Markdown, including a dedicated section for Reminders/Notes-to-Self/Lookups if present):
"""

    def get_max_text_tokens_for_prompt_input(context_window, max_generation_tokens):
        return context_window - max_generation_tokens - BUFFER_FOR_PROMPT_AND_GENERATION_MARGIN

    max_text_for_direct_final_summary = get_max_text_tokens_for_prompt_input(final_pass_ctx, final_pass_max_gen)
    full_text_tokens = len(tokenizer_for_counting.encode(full_text))

    if full_text_tokens <= max_text_for_direct_final_summary:
        if debug_mode:
            print(f"Original text ({full_text_tokens} tokens) is short enough for a single final pass. Max input: {max_text_for_direct_final_summary}.")
        return generate_summary_ollama(full_text, ollama_url, ollama_model,
                                       final_pass_ctx, final_pass_max_gen,
                                       prompt_template_initial_chunk,
                                       debug_mode=debug_mode)

    if debug_mode:
        print(f"\n--- Stage 1: Initial Chunk Summarization (Ctx: {initial_pass_ctx}, MaxGen: {initial_pass_max_gen}) ---")
    max_content_tokens_for_initial_chunk = int(initial_pass_ctx * target_chunk_tokens_ratio)
    safe_max_content_for_initial_chunk = get_max_text_tokens_for_prompt_input(initial_pass_ctx, initial_pass_max_gen)
    
    if max_content_tokens_for_initial_chunk > safe_max_content_for_initial_chunk:
        if debug_mode:
            print(f"Warning (Stage 1): Calculated chunk content tokens ({max_content_tokens_for_initial_chunk}) > safe max ({safe_max_content_for_initial_chunk}). Adjusting.")
        max_content_tokens_for_initial_chunk = safe_max_content_for_initial_chunk
    
    if max_content_tokens_for_initial_chunk <= 0:
        return f"[Config Error (Stage 1): initial_pass_ctx ({initial_pass_ctx}) too small for initial_pass_max_gen ({initial_pass_max_gen}) and buffer. Cannot proceed.]"

    initial_chunks = split_text_into_chunks(full_text, tokenizer_for_counting,
                                            max_content_tokens_for_initial_chunk, overlap_tokens,
                                            debug_mode=debug_mode)
    if not initial_chunks: return "[Error (Stage 1): No initial chunks created]"

    initial_summaries = []
    for i, chunk_text in enumerate(initial_chunks):
        if debug_mode:
            print(f"Summarizing initial chunk {i+1}/{len(initial_chunks)}...")
        summary = generate_summary_ollama(chunk_text, ollama_url, ollama_model,
                                          initial_pass_ctx, initial_pass_max_gen,
                                          prompt_template_initial_chunk,
                                          debug_mode=debug_mode)
        if debug_mode:
            save_debug_file(original_input_filepath, ollama_model, f"S1-Chunk{i+1}", summary) 
        if not (summary.startswith("[Error") or summary.startswith("[Ollama")):
            initial_summaries.append(summary)
        else:
            if debug_mode:
                print(f"Warning (Stage 1): Failed to summarize initial chunk {i+1}. Error: {summary}")
            initial_summaries.append(f"[Summary error for initial chunk {i+1}: {summary}]") 

    valid_initial_summaries = [s for s in initial_summaries if not (s.startswith("[Error") or s.startswith("[Ollama") or ("[Summary error for initial chunk" in s and "Ollama returned empty summary" not in s) )] 
    if not valid_initial_summaries:
        all_errors_stage1 = "\n".join(initial_summaries)
        return f"[Error (Stage 1): All initial chunk summarizations failed. Errors:\n{all_errors_stage1}]"
    
    texts_for_next_stage = valid_initial_summaries
    combined_initial_summaries_text = "\n\n---\n\n".join(texts_for_next_stage)
    combined_initial_summaries_text = remove_thinking_tokens(combined_initial_summaries_text)
    if debug_mode:
        save_debug_file(original_input_filepath, ollama_model, "S1-CombinedAllChunks", combined_initial_summaries_text) 

    tokens_after_initial_pass = len(tokenizer_for_counting.encode(combined_initial_summaries_text))
    max_input_tokens_for_final_pass_prompt = get_max_text_tokens_for_prompt_input(final_pass_ctx, final_pass_max_gen)

    needs_intermediate_stage = False
    if tokens_after_initial_pass > max_input_tokens_for_final_pass_prompt:
        print(f"Combined initial summaries ({tokens_after_initial_pass} tokens) are too long for direct final pass (max input: {max_input_tokens_for_final_pass_prompt}). Intermediate stage needed.")
        needs_intermediate_stage = True
    else:
        print(f"Combined initial summaries ({tokens_after_initial_pass} tokens) are short enough for direct final pass. Skipping intermediate stage.")

    if needs_intermediate_stage:
        print(f"\n--- Stage 2: Intermediate Combination (Ctx: {intermediate_pass_ctx}, MaxGen: {intermediate_pass_max_gen}) ---")
        current_texts_for_intermediate_processing = texts_for_next_stage
        intermediate_loop_count = 1
        max_intermediate_loops = 5 

        while intermediate_loop_count <= max_intermediate_loops:
            combined_text_for_this_intermediate_iter = "\n\n---\n\n".join(current_texts_for_intermediate_processing)
            combined_text_for_this_intermediate_iter = remove_thinking_tokens(combined_text_for_this_intermediate_iter)
            tokens_for_this_intermediate_iter = len(tokenizer_for_counting.encode(combined_text_for_this_intermediate_iter))
            print(f"Intermediate Loop {intermediate_loop_count}: Processing {tokens_for_this_intermediate_iter} tokens.")

            if tokens_for_this_intermediate_iter <= max_input_tokens_for_final_pass_prompt:
                print("Texts from intermediate stage are now short enough for final pass.")
                texts_for_next_stage = current_texts_for_intermediate_processing
                break 

            max_content_tokens_for_intermediate_chunk = int(intermediate_pass_ctx * target_chunk_tokens_ratio)
            safe_max_content_for_intermediate_chunk = get_max_text_tokens_for_prompt_input(intermediate_pass_ctx, intermediate_pass_max_gen)

            if max_content_tokens_for_intermediate_chunk > safe_max_content_for_intermediate_chunk:
                print(f"Warning (Stage 2 Loop {intermediate_loop_count}): Calculated chunk content tokens ({max_content_tokens_for_intermediate_chunk}) > safe max ({safe_max_content_for_intermediate_chunk}). Adjusting.")
                max_content_tokens_for_intermediate_chunk = safe_max_content_for_intermediate_chunk
            
            if max_content_tokens_for_intermediate_chunk <= 0:
                print(f"[Config Error (Stage 2 Loop {intermediate_loop_count}): intermediate_pass_ctx ({intermediate_pass_ctx}) too small. Using previous texts.]")
                texts_for_next_stage = current_texts_for_intermediate_processing 
                break

            intermediate_chunks_to_summarize = split_text_into_chunks(combined_text_for_this_intermediate_iter,
                                                                    tokenizer_for_counting,
                                                                    max_content_tokens_for_intermediate_chunk,
                                                                    overlap_tokens,
                                                                    debug_mode=debug_mode)
            if not intermediate_chunks_to_summarize:
                print(f"Error (Stage 2 Loop {intermediate_loop_count}): No chunks created. Using previous texts.")
                texts_for_next_stage = current_texts_for_intermediate_processing 
                break
            
            summaries_from_this_intermediate_iter = []
            for i, text_to_summarize in enumerate(intermediate_chunks_to_summarize):
                print(f"Summarizing intermediate chunk {i+1}/{len(intermediate_chunks_to_summarize)} (Loop {intermediate_loop_count})...")
                summary = generate_summary_ollama(text_to_summarize, ollama_url, ollama_model,
                                                  intermediate_pass_ctx, intermediate_pass_max_gen,
                                                  prompt_template_combine_summaries,
                                                  debug_mode=debug_mode) 
                if debug_mode:
                    save_debug_file(original_input_filepath, ollama_model, f"S2L{intermediate_loop_count}-Chunk{i+1}", summary) 
                if not (summary.startswith("[Error") or summary.startswith("[Ollama")):
                    summaries_from_this_intermediate_iter.append(summary)
                else:
                    print(f"Warning (Stage 2 Loop {intermediate_loop_count}): Failed to summarize intermediate chunk {i+1}. Error: {summary}")
                    summaries_from_this_intermediate_iter.append(f"[Summary error for intermediate chunk {i+1} L{intermediate_loop_count}: {summary}]")
            
            valid_summaries_this_iter = [s for s in summaries_from_this_intermediate_iter if not (s.startswith("[Error") or s.startswith("[Ollama") or ("[Summary error for intermediate chunk" in s and "Ollama returned empty summary" not in s))]
            if not valid_summaries_this_iter:
                all_errors_stage2_iter = "\n".join(summaries_from_this_intermediate_iter)
                print(f"[Error (Stage 2 Loop {intermediate_loop_count}): All intermediate summarizations failed. Using previous texts. Errors:\n{all_errors_stage2_iter}]")
                texts_for_next_stage = current_texts_for_intermediate_processing 
                break 
            
            current_texts_for_intermediate_processing = valid_summaries_this_iter
            combined_output_this_intermediate_loop = "\n\n---\n\n".join(current_texts_for_intermediate_processing)
            combined_output_this_intermediate_loop = remove_thinking_tokens(combined_output_this_intermediate_loop)
            if debug_mode:
                save_debug_file(original_input_filepath, ollama_model, f"S2L{intermediate_loop_count}-CombinedAllChunks", combined_output_this_intermediate_loop) 

            intermediate_loop_count += 1
            if intermediate_loop_count > max_intermediate_loops:
                print(f"Reached max intermediate loops ({max_intermediate_loops}). Proceeding with current texts.")
                texts_for_next_stage = current_texts_for_intermediate_processing
                break
        else: 
            texts_for_next_stage = current_texts_for_intermediate_processing

    print(f"\n--- Stage 3: Final Combination Summarization (Ctx: {final_pass_ctx}, MaxGen: {final_pass_max_gen}) ---")
    combined_text_for_final_summary = "\n\n---\n\n".join(texts_for_next_stage)
    combined_text_for_final_summary = remove_thinking_tokens(combined_text_for_final_summary)
    final_input_tokens = len(tokenizer_for_counting.encode(combined_text_for_final_summary))
    print(f"Generating final summary from {len(texts_for_next_stage)} text(s), totaling {final_input_tokens} tokens.")

    if final_input_tokens > max_input_tokens_for_final_pass_prompt:
        print(f"Warning (Stage 3): Final input text ({final_input_tokens} tokens) still exceeds recommended max ({max_input_tokens_for_final_pass_prompt}) for the prompt. Summary quality may be affected or generation might fail.")

    final_summary = generate_summary_ollama(combined_text_for_final_summary, ollama_url, ollama_model,
                                            final_pass_ctx, final_pass_max_gen,
                                            prompt_template_combine_summaries,
                                            debug_mode=debug_mode) 
    return final_summary

def main():
    parser = argparse.ArgumentParser(description="Summarize a long text file using multiple Ollama models with a multi-stage MapReduce approach.")
    parser.add_argument("file_path", help="Path to the long text file.")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL, help=f"URL of the Ollama server (default: {DEFAULT_OLLAMA_URL}).")
    parser.add_argument("--ollama_model", help=f"Optional: Specify a single model to use. If not provided, will run all models in MODELS_TO_RUN_LIST: {', '.join(MODELS_TO_RUN_LIST)}")
    
    parser.add_argument("--initial_pass_ollama_num_ctx", type=int, default=DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX, help=f"Context window for initial chunk summarization (default: {DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX}).")
    parser.add_argument("--initial_pass_max_new_tokens", type=int, default=DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS, help=f"Max new tokens for initial chunk summaries (default: {DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS}).")
    parser.add_argument("--intermediate_pass_ollama_num_ctx", type=int, default=DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX, help=f"Context window for intermediate combination (default: {DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX}).")
    parser.add_argument("--intermediate_pass_max_new_tokens", type=int, default=DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS, help=f"Max new tokens for intermediate summaries (default: {DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS}).")
    parser.add_argument("--final_pass_ollama_num_ctx", type=int, default=DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX, help=f"Context window for final summary (default: {DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX}).")
    parser.add_argument("--final_pass_max_new_tokens", type=int, default=DEFAULT_FINAL_PASS_MAX_NEW_TOKENS, help=f"Max new tokens for the final summary (default: {DEFAULT_FINAL_PASS_MAX_NEW_TOKENS}).")

    parser.add_argument("--output_file", help="Optional base path to save the final summaries (will be .md). Model names will be appended. If not provided, summaries are saved in the same directory as the input file.")
    parser.add_argument("--target_chunk_ratio", type=float, default=DEFAULT_TARGET_CHUNK_TOKENS_RATIO, help=f"Ratio of a stage's ollama_num_ctx to use for text content in each chunk for that stage (default: {DEFAULT_TARGET_CHUNK_TOKENS_RATIO}).")
    parser.add_argument("--overlap_tokens", type=int, default=DEFAULT_TOKEN_OVERLAP, help=f"Token overlap between chunks (default: {DEFAULT_TOKEN_OVERLAP}).")
    # New DEBUG flag
    parser.add_argument("--DEBUG", action='store_true', help="Enable saving of intermediate debug files.")


    args = parser.parse_args()

    if not check_ollama_status(args.ollama_url, args.DEBUG):
        return

    try:
        tokenizer_for_counting = get_tokenizer_for_counting()
    except Exception:
        print("Failed to initialize tokenizer for counting. Exiting.")
        return

    try:
        with open(args.file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {args.file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not full_text.strip():
        print("Error: Input file is empty.")
        return

    # Determine which models to run
    models_to_run = [args.ollama_model] if args.ollama_model else MODELS_TO_RUN_LIST

    for model_name_for_iteration in models_to_run:
        if args.DEBUG:
            print(f"\n\n{'='*20} PROCESSING WITH MODEL: {model_name_for_iteration} {'='*20}")
            print(f"Input file: {args.file_path}")
            print(f"Ollama URL: {args.ollama_url}, Model: {model_name_for_iteration}")
            print(f"Config: Initial Pass (Ctx: {args.initial_pass_ollama_num_ctx}, MaxGen: {args.initial_pass_max_new_tokens})")
            print(f"        Intermediate Pass (Ctx: {args.intermediate_pass_ollama_num_ctx}, MaxGen: {args.intermediate_pass_max_new_tokens})")
            print(f"        Final Pass (Ctx: {args.final_pass_ollama_num_ctx}, MaxGen: {args.final_pass_max_new_tokens})")
            print(f"        Chunk Ratio: {args.target_chunk_ratio}, Overlap: {args.overlap_tokens}")
            print(f"        DEBUG mode enabled: Intermediate files will be saved.")

        final_summary = summarize_transcript(
            full_text, tokenizer_for_counting, 
            args.ollama_url, model_name_for_iteration,
            args.initial_pass_ollama_num_ctx, args.initial_pass_max_new_tokens,
            args.intermediate_pass_ollama_num_ctx, args.intermediate_pass_max_new_tokens,
            args.final_pass_ollama_num_ctx, args.final_pass_max_new_tokens,
            args.target_chunk_ratio, args.overlap_tokens,
            args.file_path,
            args.DEBUG # Pass the debug flag
        )

        if args.DEBUG:
            print(f"\n--- FINAL SUMMARY ({model_name_for_iteration}) ---")
            print(final_summary)

        output_file_path_to_use = None
        sanitized_model_name = model_name_for_iteration.replace(":", "_").replace("/", "_")
        output_extension = ".md" 

        if args.output_file:
            path_part, original_filename_with_ext = os.path.split(args.output_file)
            base_name = os.path.splitext(original_filename_with_ext)[0] 
            new_filename = f"{base_name}_{sanitized_model_name}{output_extension}"
            output_file_path_to_use = os.path.join(path_part, new_filename)
        else:
            input_dir = os.path.dirname(args.file_path) 
            input_filename_base = os.path.splitext(os.path.basename(args.file_path))[0]
            new_filename = f"{input_filename_base}_summary_{sanitized_model_name}{output_extension}"
            output_file_path_to_use = os.path.join(input_dir, new_filename)

        try:
            output_dir_for_saving = os.path.dirname(output_file_path_to_use)
            if output_dir_for_saving: 
                os.makedirs(output_dir_for_saving, exist_ok=True)
            
            with open(output_file_path_to_use, "w", encoding="utf-8") as f:
                f.write(f"# Summary from model: {model_name_for_iteration}\n")
                f.write(f"# Config: Initial (Ctx: {args.initial_pass_ollama_num_ctx}, MaxGen: {args.initial_pass_max_new_tokens}), "
                        f"Intermediate (Ctx: {args.intermediate_pass_ollama_num_ctx}, MaxGen: {args.intermediate_pass_max_new_tokens}), "
                        f"Final (Ctx: {args.final_pass_ollama_num_ctx}, MaxGen: {args.final_pass_max_new_tokens})\n\n")
                f.write(final_summary)
            if args.DEBUG:
                print(f"\nFinal summary for {model_name_for_iteration} saved to {output_file_path_to_use}")
        except Exception as e:
            print(f"Error saving summary for {model_name_for_iteration} to file '{output_file_path_to_use}': {e}")
        
        if args.DEBUG:
            print(f"{'='*20} FINISHED PROCESSING MODEL: {model_name_for_iteration} {'='*20}\n")

if __name__ == "__main__":
    main()

