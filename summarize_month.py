#!/usr/bin/env python3
import requests
import argparse
import os
import json
import tiktoken
import re
import logging
from logging_config import setup_logging
from datetime import datetime, timedelta
import glob

# --- Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "phi4-mini:latest"

# Initialize logger
logger = setup_logging('summarize_month', False)  # Default to non-debug mode

MODELS_TO_RUN_LIST = [
    "phi4-mini:latest"               
]

# Context window sizes for different summarization passes
DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX = 8192
DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX = 12288
DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX = 16384

# Max new tokens to generate for each pass's output
DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS = 1000
DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS = 3000
DEFAULT_FINAL_PASS_MAX_NEW_TOKENS = 6000

# Other chunking parameters
DEFAULT_TARGET_CHUNK_TOKENS_RATIO = 0.7
DEFAULT_TOKEN_OVERLAP = 100
TIKTOKEN_ENCODING_FOR_CHUNKNG = "o100k_base"
BUFFER_FOR_PROMPT_AND_GENERATION_MARGIN = 300

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
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama server at {ollama_url}: {e}")
        print("Please ensure Ollama is running and accessible.")
        return False

def split_text_into_chunks(text, tokenizer, max_content_tokens_for_chunk, overlap_tokens, debug_mode=False):
    """Splits text into chunks."""
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
        if advance_by <= 0:
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
        "stream": True,
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
        
        summary = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        chunk = json_response['response']
                        print(chunk, end='', flush=True)
                        summary += chunk
                    
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

        print()
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

def remove_thinking_tokens(text):
    """Remove any text between <think> and </think> tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\s*[tT][hH][iI][nN][kK]\s*>.*?</\s*[tT][hH][iI][nN][kK]\s*>', '', text, flags=re.DOTALL)
    return text

def get_month_folder_path(base_dir, date):
    """Generate the month folder path based on the date."""
    year = str(date.year)
    month = date.strftime("%B")
    return os.path.join(base_dir, "monthly", year, month)

def collect_weekly_summaries(month_folder, model_name):
    """Collect all weekly summaries for the month."""
    summaries = []
    
    # Get the month's date range
    month_folder_name = os.path.basename(month_folder)
    year = os.path.basename(os.path.dirname(month_folder))
    
    try:
        # Convert month name to number
        month_num = datetime.strptime(month_folder_name, "%B").month
        year_num = int(year)
        
        # Validate year is between 2024 and 2099
        if year_num < 2024 or year_num > 2099:
            print(f"Error: Invalid year: {year_num}. Year must be between 2024 and 2099.")
            return summaries
        
        # Calculate the first and last day of the month
        first_day = datetime(year_num, month_num, 1)
        if month_num == 12:
            last_day = datetime(year_num + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year_num, month_num + 1, 1) - timedelta(days=1)
        
        print(f"\nProcessing month from {first_day.strftime('%Y-%m-%d')} through {last_day.strftime('%Y-%m-%d')}")
        print(f"Month folder: {month_folder}")
        
        # Get the base directory (journals root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(month_folder)))
        print(f"Base directory: {base_dir}")
        
        # Look for weekly summaries in the weekly directory
        weekly_dir = os.path.join(base_dir, "weekly", year)
        if not os.path.exists(weekly_dir):
            print(f"Weekly directory not found: {weekly_dir}")
            return summaries
            
        # Find all week folders in the month
        week_pattern = os.path.join(weekly_dir, month_folder_name, "WeekEnding*")
        week_folders = glob.glob(week_pattern)
        print(f"Found {len(week_folders)} week folders in {weekly_dir}/{month_folder_name}")
        
        for week_folder in week_folders:
            try:
                # Extract the Saturday date from the folder name
                week_folder_name = os.path.basename(week_folder)
                if not week_folder_name.startswith("WeekEnding"):
                    continue
                    
                saturday_date_str = week_folder_name[11:]  # Remove "WeekEnding" prefix
                saturday_date = datetime.strptime(saturday_date_str, "%Y%m%d")
                
                # Check if this week is within our target month
                if saturday_date.month != month_num or saturday_date.year != year_num:
                    continue
                
                # Look for weekly summary files
                pattern = os.path.join(week_folder, "weekly-summary-*.md")
                summary_files = glob.glob(pattern)
                print(f"Found {len(summary_files)} summary files in {week_folder}")
                
                for file_path in summary_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract the week's date range from the content
                            week_date_match = re.search(r"Week of Sunday, (.*?) through Saturday, (.*?)\n", content)
                            if week_date_match:
                                week_start = week_date_match.group(1)
                                week_end = week_date_match.group(2)
                                date_range = f"{week_start} - {week_end}"
                            else:
                                date_range = saturday_date.strftime("%Y-%m-%d")
                            summaries.append((date_range, content))
                            print(f"Added summary from: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Error reading summary file {file_path}: {e}")
            
            except Exception as e:
                print(f"Error processing week folder {week_folder}: {e}")
    
    except Exception as e:
        print(f"Error processing month folder {month_folder}: {e}")
    
    # Sort summaries by date
    summaries.sort(key=lambda x: x[0])
    print(f"\nTotal summaries collected: {len(summaries)}")
    if not summaries:
        print(f"No weekly summaries found in {month_folder_name} {year}")
    return summaries

def summarize_month(weekly_summaries, tokenizer_for_counting, ollama_url, ollama_model,
                   initial_pass_ctx, initial_pass_max_gen,
                   intermediate_pass_ctx, intermediate_pass_max_gen,
                   final_pass_ctx, final_pass_max_gen,
                   target_chunk_tokens_ratio, overlap_tokens,
                   original_input_filepath,
                   debug_mode):
    """Summarizes a month's worth of weekly summaries using a multi-stage approach."""

    prompt_template_initial_chunk = """
You are analyzing a collection of weekly summaries from a month. Your task is to identify and extract the most significant information, patterns, and recurring themes across these summaries.

IMPORTANT GUIDELINES:
* Focus on real-world events, activities, and discussions
* Exclude content that is clearly from entertainment media (TV shows, movies, songs)
* Pay special attention to recurring themes or topics that appear across multiple weeks
* Preserve all reminders, notes-to-self, and lookups from the original summaries
* Identify any patterns in decision-making or problem-solving approaches

Your response MUST be in well-formatted Markdown. Use:
* **Bold text** for emphasis
* Bullet points (`*` or `-`) for lists
* Headers (`##` for main sections, `###` for subsections)
* Code blocks (```) for any technical content
* Tables where appropriate

Please analyze this excerpt and provide a detailed summary focusing on:

## Major Themes & Patterns
* What topics, concerns, or activities appear consistently across multiple weeks?

## Key Decisions & Their Evolution
* How have important decisions evolved or been refined over the month?

## Recurring Questions & Concerns
* What questions or uncertainties have persisted or evolved?

## Action Items & Progress
* What tasks were completed, started, or carried forward? Track their progress.

## Notable Insights & Breakthroughs
* What significant realizations or breakthroughs occurred?

## Critical Information & Facts
* What key information emerged that's important to remember?

## REMINDERS & NOTES-TO-SELF
* CRITICALLY IMPORTANT: Preserve all items explicitly labeled as reminders, notes-to-self, or lookups. These must be passed on.

Weekly Summaries Excerpt:
---
{text_chunk}
---
Summary of Excerpt (including any REMINDER/NOTE-TO-SELF/LOOKUP items):
"""

    prompt_template_combine_summaries = """
You are creating a comprehensive monthly summary by synthesizing multiple summaries of weekly summaries. Your goal is to create a coherent, well-structured overview of the entire month that highlights patterns, progress, and important information.

The output MUST be in well-formatted Markdown, designed for maximum readability and visual appeal. Use:
* **Main Headings (##)** for major sections
* **Sub-headings (###)** for subsections
* **Bullet points (`*` or `-`)** for lists
* **Bold text (`**text**`)** for emphasis
* **Numbered lists** for sequential items
* **Tables** for structured data
* **Code blocks (```)** for any technical content
* **Horizontal rules (---)** to separate major sections

In this monthly synthesis, focus on:

## Monthly Narrative & Flow
* How did the month progress? What was the overall arc?

## Cross-Week Themes & Patterns
* What topics, concerns, or activities appeared consistently?

## Decision Evolution
* How did key decisions evolve or get refined?

## Progress Tracking
* What was started, completed, or carried forward?

## Recurring Questions & Concerns
* What issues persisted or evolved?

## Breakthroughs & Insights
* What significant realizations occurred?

## Reminders & Follow-ups
* CRITICAL: Create a dedicated section for all reminders, notes-to-self, and lookups. Preserve these exactly as they appear in the weekly summaries.

## Pattern Recognition
* Identify any patterns in how problems were approached or solved.

## Information Synthesis
* Combine related information from different weeks into coherent insights.

The aim is to produce a high-level yet detailed digest that allows someone to quickly understand the month's key events, patterns, and important outcomes, while preserving all critical reminders and follow-up items.

IMPORTANT: Your response MUST be only the Markdown summary itself. Do NOT include any commentary or analysis of these instructions.

Combined Summaries:
---
{text_chunk}
---
Comprehensive Monthly Summary (Markdown, including a dedicated section for Reminders/Notes-to-Self/Lookups):
"""

    def get_max_text_tokens_for_prompt_input(context_window, max_generation_tokens):
        return context_window - max_generation_tokens - BUFFER_FOR_PROMPT_AND_GENERATION_MARGIN

    # Combine all weekly summaries with clear date markers
    combined_text = "\n\n".join([f"=== {date} ===\n{content}" for date, content in weekly_summaries])
    
    max_text_for_direct_final_summary = get_max_text_tokens_for_prompt_input(final_pass_ctx, final_pass_max_gen)
    full_text_tokens = len(tokenizer_for_counting.encode(combined_text))

    if full_text_tokens <= max_text_for_direct_final_summary:
        if debug_mode:
            print(f"Combined text ({full_text_tokens} tokens) is short enough for a single final pass. Max input: {max_text_for_direct_final_summary}.")
        return generate_summary_ollama(combined_text, ollama_url, ollama_model,
                                     final_pass_ctx, final_pass_max_gen,
                                     prompt_template_initial_chunk,
                                     debug_mode=debug_mode)

    # Multi-stage summarization process
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

    initial_chunks = split_text_into_chunks(combined_text, tokenizer_for_counting,
                                          max_content_tokens_for_initial_chunk, overlap_tokens,
                                          debug_mode=debug_mode)
    if not initial_chunks:
        return "[Error (Stage 1): No initial chunks created]"

    initial_summaries = []
    for i, chunk_text in enumerate(initial_chunks):
        if debug_mode:
            print(f"Summarizing initial chunk {i+1}/{len(initial_chunks)}...")
        summary = generate_summary_ollama(chunk_text, ollama_url, ollama_model,
                                        initial_pass_ctx, initial_pass_max_gen,
                                        prompt_template_initial_chunk,
                                        debug_mode=debug_mode)
        if not (summary.startswith("[Error") or summary.startswith("[Ollama")):
            initial_summaries.append(summary)
        else:
            if debug_mode:
                print(f"Warning (Stage 1): Failed to summarize initial chunk {i+1}. Error: {summary}")
            initial_summaries.append(f"[Summary error for initial chunk {i+1}: {summary}]")

    valid_initial_summaries = [s for s in initial_summaries if not (s.startswith("[Error") or s.startswith("[Ollama") or ("[Summary error for initial chunk" in s and "Ollama returned empty summary" not in s))]
    if not valid_initial_summaries:
        all_errors_stage1 = "\n".join(initial_summaries)
        return f"[Error (Stage 1): All initial chunk summarizations failed. Errors:\n{all_errors_stage1}]"

    texts_for_next_stage = valid_initial_summaries
    combined_initial_summaries_text = "\n\n---\n\n".join(texts_for_next_stage)
    combined_initial_summaries_text = remove_thinking_tokens(combined_initial_summaries_text)

    tokens_after_initial_pass = len(tokenizer_for_counting.encode(combined_initial_summaries_text))
    max_input_tokens_for_final_pass_prompt = get_max_text_tokens_for_prompt_input(final_pass_ctx, final_pass_max_gen)

    needs_intermediate_stage = tokens_after_initial_pass > max_input_tokens_for_final_pass_prompt
    if needs_intermediate_stage:
        print(f"Combined initial summaries ({tokens_after_initial_pass} tokens) are too long for direct final pass (max input: {max_input_tokens_for_final_pass_prompt}). Intermediate stage needed.")
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
    parser = argparse.ArgumentParser(description="Generate a monthly summary from weekly summaries using multiple Ollama models.")
    parser.add_argument("date", help="Date in YYYY-MM-DD format to determine the month to summarize.")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the weekly summaries.")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL, help=f"URL of the Ollama server (default: {DEFAULT_OLLAMA_URL}).")
    parser.add_argument("--ollama_model", help=f"Optional: Specify a single model to use. If not provided, will run all models in MODELS_TO_RUN_LIST: {', '.join(MODELS_TO_RUN_LIST)}")
    
    parser.add_argument("--initial_pass_ollama_num_ctx", type=int, default=DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX)
    parser.add_argument("--initial_pass_max_new_tokens", type=int, default=DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS)
    parser.add_argument("--intermediate_pass_ollama_num_ctx", type=int, default=DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX)
    parser.add_argument("--intermediate_pass_max_new_tokens", type=int, default=DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS)
    parser.add_argument("--final_pass_ollama_num_ctx", type=int, default=DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX)
    parser.add_argument("--final_pass_max_new_tokens", type=int, default=DEFAULT_FINAL_PASS_MAX_NEW_TOKENS)

    parser.add_argument("--target_chunk_ratio", type=float, default=DEFAULT_TARGET_CHUNK_TOKENS_RATIO)
    parser.add_argument("--overlap_tokens", type=int, default=DEFAULT_TOKEN_OVERLAP)
    parser.add_argument("--DEBUG", action='store_true', help="Enable saving of intermediate debug files.")

    args = parser.parse_args()

    # Update logger with debug mode
    logger = setup_logging('summarize_month', args.DEBUG)

    if not check_ollama_status(args.ollama_url, args.DEBUG):
        return

    try:
        input_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD format.")
        return

    try:
        tokenizer_for_counting = get_tokenizer_for_counting()
    except Exception:
        print("Failed to initialize tokenizer for counting. Exiting.")
        return

    # Determine which models to run
    models_to_run = [args.ollama_model] if args.ollama_model else MODELS_TO_RUN_LIST

    # Get the month folder path
    month_folder = get_month_folder_path(args.base_dir, input_date)
    print(f"\nCreating monthly summary in: {month_folder}")
    os.makedirs(month_folder, exist_ok=True)

    # Collect weekly summaries once for all models
    weekly_summaries = collect_weekly_summaries(month_folder, None)
    if not weekly_summaries:
        print(f"No weekly summaries found in {input_date.strftime('%B %Y')}")
        return

    for model_name_for_iteration in models_to_run:
        if args.DEBUG:
            print(f"\n\n{'='*20} PROCESSING WITH MODEL: {model_name_for_iteration} {'='*20}")
            print(f"Month folder: {month_folder}")
            print(f"Ollama URL: {args.ollama_url}, Model: {model_name_for_iteration}")
            print(f"Config: Initial Pass (Ctx: {args.initial_pass_ollama_num_ctx}, MaxGen: {args.initial_pass_max_new_tokens})")
            print(f"        Intermediate Pass (Ctx: {args.intermediate_pass_ollama_num_ctx}, MaxGen: {args.intermediate_pass_max_new_tokens})")
            print(f"        Final Pass (Ctx: {args.final_pass_ollama_num_ctx}, MaxGen: {args.final_pass_max_new_tokens})")
            print(f"        Chunk Ratio: {args.target_chunk_ratio}, Overlap: {args.overlap_tokens}")
            print(f"        DEBUG mode enabled: Intermediate files will be saved.")

        final_summary = summarize_month(
            weekly_summaries, tokenizer_for_counting,
            args.ollama_url, model_name_for_iteration,
            args.initial_pass_ollama_num_ctx, args.initial_pass_max_new_tokens,
            args.intermediate_pass_ollama_num_ctx, args.intermediate_pass_max_new_tokens,
            args.final_pass_ollama_num_ctx, args.final_pass_max_new_tokens,
            args.target_chunk_ratio, args.overlap_tokens,
            month_folder,
            args.DEBUG
        )

        if args.DEBUG:
            print(f"\n--- FINAL MONTHLY SUMMARY ({model_name_for_iteration}) ---")
            print(final_summary)

        # Save the monthly summary
        sanitized_model_name = model_name_for_iteration.replace(":", "_").replace("/", "_")
        output_filename = f"monthly-summary-{sanitized_model_name}.md"
        output_path = os.path.join(month_folder, output_filename)
        print(f"\nSaving monthly summary to: {output_path}")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Monthly Summary from model: {model_name_for_iteration}\n\n")
                f.write(f"# Monthly Summary of {input_date.strftime('%B %Y')}\n\n")
                cleaned_summary = remove_thinking_tokens(final_summary)
                f.write(cleaned_summary)
            print(f"Successfully saved monthly summary to: {output_path}")
        except Exception as e:
            print(f"Error saving monthly summary for {model_name_for_iteration} to file '{output_path}': {e}")

        if args.DEBUG:
            print(f"{'='*20} FINISHED PROCESSING MODEL: {model_name_for_iteration} {'='*20}\n")

if __name__ == "__main__":
    main() 