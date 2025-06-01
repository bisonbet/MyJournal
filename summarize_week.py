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
DEFAULT_OLLAMA_MODEL = "qwen3:30b-a3b"

# Initialize logger
logger = setup_logging('summarize_week', False)  # Default to non-debug mode

MODELS_TO_RUN_LIST = [
    "qwen3:30b-a3b"               
]

# Context window sizes for different summarization passes
DEFAULT_INITIAL_PASS_OLLAMA_NUM_CTX = 4096
DEFAULT_INTERMEDIATE_PASS_OLLAMA_NUM_CTX = 8192
DEFAULT_FINAL_PASS_OLLAMA_NUM_CTX = 16000

# Max new tokens to generate for each pass's output
DEFAULT_INITIAL_PASS_MAX_NEW_TOKENS = 750
DEFAULT_INTERMEDIATE_PASS_MAX_NEW_TOKENS = 2500
DEFAULT_FINAL_PASS_MAX_NEW_TOKENS = 5000

# Other chunking parameters
DEFAULT_TARGET_CHUNK_TOKENS_RATIO = 0.6
DEFAULT_TOKEN_OVERLAP = 200
TIKTOKEN_ENCODING_FOR_CHUNKNG = "o200k_base"
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

def get_week_folder_path(base_dir, date):
    """Generate the week folder path based on the date."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    # Find the Sunday of the week
    sunday = date - timedelta(days=date.weekday() + 1)
    week_folder = f"WeekOf{sunday.strftime('%Y%m%d')}"
    return os.path.join(base_dir, "weekly", year, month, week_folder)

def collect_daily_summaries(week_folder, model_name):
    """Collect all daily summaries for the week for a specific model."""
    summaries = []
    # Look for daily summaries in the format YYYY-MM-DD-{model_name}-daily-summary.md
    pattern = os.path.join(week_folder, "..", "..", "..", "*", f"*-{model_name}-daily-summary.md")
    summary_files = glob.glob(pattern)
    
    # Sort files by date
    summary_files.sort()
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the date from the filename
                filename = os.path.basename(file_path)
                date_str = filename.split('-')[0]
                summaries.append((date_str, content))
        except Exception as e:
            print(f"Error reading summary file {file_path}: {e}")
    
    return summaries

def summarize_week(daily_summaries, tokenizer_for_counting, ollama_url, ollama_model,
                   initial_pass_ctx, initial_pass_max_gen,
                   intermediate_pass_ctx, intermediate_pass_max_gen,
                   final_pass_ctx, final_pass_max_gen,
                   target_chunk_tokens_ratio, overlap_tokens,
                   original_input_filepath,
                   debug_mode):
    """Summarizes a week's worth of daily summaries using a multi-stage approach."""

    prompt_template_initial_chunk = """
You are analyzing a collection of daily summaries from a week. Your task is to identify and extract the most significant information, patterns, and recurring themes across these summaries.

IMPORTANT GUIDELINES:
* Focus on real-world events, activities, and discussions
* Exclude content that is clearly from entertainment media (TV shows, movies, songs)
* Pay special attention to recurring themes or topics that appear across multiple days
* Preserve all reminders, notes-to-self, and lookups from the original summaries
* Identify any patterns in decision-making or problem-solving approaches

Please analyze this excerpt and provide a detailed summary focusing on:

* **Major Themes & Patterns:** What topics, concerns, or activities appear consistently across multiple days?
* **Key Decisions & Their Evolution:** How have important decisions evolved or been refined over the week?
* **Recurring Questions & Concerns:** What questions or uncertainties have persisted or evolved?
* **Action Items & Progress:** What tasks were completed, started, or carried forward? Track their progress.
* **Notable Insights & Breakthroughs:** What significant realizations or breakthroughs occurred?
* **Critical Information & Facts:** What key information emerged that's important to remember?
* **REMINDERS & NOTES-TO-SELF:** CRITICALLY IMPORTANT: Preserve all items explicitly labeled as reminders, notes-to-self, or lookups. These must be passed on.

Daily Summaries Excerpt:
---
{text_chunk}
---
Summary of Excerpt (including any REMINDER/NOTE-TO-SELF/LOOKUP items):
"""

    prompt_template_combine_summaries = """
You are creating a comprehensive weekly summary by synthesizing multiple summaries of daily summaries. Your goal is to create a coherent, well-structured overview of the entire week that highlights patterns, progress, and important information.

The output MUST be in well-formatted Markdown, designed for maximum readability and visual appeal. Use:
* **Main Headings (e.g., ## Weekly Overview, ## Major Themes)**
* **Sub-headings (e.g., ### Progress on Project X, ### Recurring Concerns)**
* **Bullet points (`*` or `-`)** for lists
* **Bold text (`**text**`)** for emphasis
* **Numbered lists** for sequential items

In this weekly synthesis, focus on:

* **Weekly Narrative & Flow:** How did the week progress? What was the overall arc?
* **Cross-Day Themes & Patterns:** What topics, concerns, or activities appeared consistently?
* **Decision Evolution:** How did key decisions evolve or get refined?
* **Progress Tracking:** What was started, completed, or carried forward?
* **Recurring Questions & Concerns:** What issues persisted or evolved?
* **Breakthroughs & Insights:** What significant realizations occurred?
* **CRITICAL: Reminders & Follow-ups:** Create a dedicated section for all reminders, notes-to-self, and lookups. Preserve these exactly as they appear in the daily summaries.
* **Pattern Recognition:** Identify any patterns in how problems were approached or solved.
* **Information Synthesis:** Combine related information from different days into coherent insights.

The aim is to produce a high-level yet detailed digest that allows someone to quickly understand the week's key events, patterns, and important outcomes, while preserving all critical reminders and follow-up items.

IMPORTANT: Your response MUST be only the Markdown summary itself. Do NOT include any commentary or analysis of these instructions.

Combined Summaries:
---
{text_chunk}
---
Comprehensive Weekly Summary (Markdown, including a dedicated section for Reminders/Notes-to-Self/Lookups):
"""

    def get_max_text_tokens_for_prompt_input(context_window, max_generation_tokens):
        return context_window - max_generation_tokens - BUFFER_FOR_PROMPT_AND_GENERATION_MARGIN

    # Combine all daily summaries with clear date markers
    combined_text = "\n\n".join([f"=== {date} ===\n{content}" for date, content in daily_summaries])
    
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
    parser = argparse.ArgumentParser(description="Generate a weekly summary from daily summaries using multiple Ollama models.")
    parser.add_argument("date", help="Date in YYYY-MM-DD format to determine the week to summarize.")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the daily summaries.")
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
    logger = setup_logging('summarize_week', args.DEBUG)

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

    # Get the week folder path
    week_folder = get_week_folder_path(args.base_dir, input_date)
    os.makedirs(week_folder, exist_ok=True)

    for model_name_for_iteration in models_to_run:
        if args.DEBUG:
            print(f"\n\n{'='*20} PROCESSING WITH MODEL: {model_name_for_iteration} {'='*20}")
            print(f"Week folder: {week_folder}")
            print(f"Ollama URL: {args.ollama_url}, Model: {model_name_for_iteration}")
            print(f"Config: Initial Pass (Ctx: {args.initial_pass_ollama_num_ctx}, MaxGen: {args.initial_pass_max_new_tokens})")
            print(f"        Intermediate Pass (Ctx: {args.intermediate_pass_ollama_num_ctx}, MaxGen: {args.intermediate_pass_max_new_tokens})")
            print(f"        Final Pass (Ctx: {args.final_pass_ollama_num_ctx}, MaxGen: {args.final_pass_max_new_tokens})")
            print(f"        Chunk Ratio: {args.target_chunk_ratio}, Overlap: {args.overlap_tokens}")
            print(f"        DEBUG mode enabled: Intermediate files will be saved.")

        # Collect daily summaries for this model
        daily_summaries = collect_daily_summaries(week_folder, model_name_for_iteration)
        if not daily_summaries:
            print(f"No daily summaries found for model {model_name_for_iteration} in the week of {input_date.strftime('%Y-%m-%d')}")
            continue

        final_summary = summarize_week(
            daily_summaries, tokenizer_for_counting,
            args.ollama_url, model_name_for_iteration,
            args.initial_pass_ollama_num_ctx, args.initial_pass_max_new_tokens,
            args.intermediate_pass_ollama_num_ctx, args.intermediate_pass_max_new_tokens,
            args.final_pass_ollama_num_ctx, args.final_pass_max_new_tokens,
            args.target_chunk_ratio, args.overlap_tokens,
            week_folder,
            args.DEBUG
        )

        if args.DEBUG:
            print(f"\n--- FINAL WEEKLY SUMMARY ({model_name_for_iteration}) ---")
            print(final_summary)

        # Save the weekly summary
        sanitized_model_name = model_name_for_iteration.replace(":", "_").replace("/", "_")
        output_filename = f"weekly-summary-{sanitized_model_name}.md"
        output_path = os.path.join(week_folder, output_filename)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Weekly Summary from model: {model_name_for_iteration}\n\n")
                f.write(f"# Week of {input_date.strftime('%Y-%m-%d')}\n\n")
                cleaned_summary = remove_thinking_tokens(final_summary)
                f.write(cleaned_summary)
            if args.DEBUG:
                print(f"\nWeekly summary for {model_name_for_iteration} saved to {output_path}")
        except Exception as e:
            print(f"Error saving weekly summary for {model_name_for_iteration} to file '{output_path}': {e}")

        if args.DEBUG:
            print(f"{'='*20} FINISHED PROCESSING MODEL: {model_name_for_iteration} {'='*20}\n")

if __name__ == "__main__":
    main() 