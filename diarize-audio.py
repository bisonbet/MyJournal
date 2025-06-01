#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
import glob
import logging
from datetime import datetime
from pathlib import Path
from logging_config import setup_logging

# --- Configuration ---
SCRIPT1_NAME = "transcribeTHIS.py"
SCRIPT2_NAME = "summarizeTHIS.py"
SCRIPT3_NAME = "summarize_week.py"

# Initialize logger with DEBUG level by default since this is the orchestrator
logger = setup_logging('diarize', True)  # Default to debug mode

def check_script_exists(script_name):
    """Checks if a script exists in the PWD."""
    script_path = os.path.join(os.getcwd(), script_name)
    if not os.path.exists(script_path):
        logger.error(f"Script '{script_name}' not found in the current directory ({os.getcwd()}).")
        logger.error(f"Please ensure {SCRIPT1_NAME}, {SCRIPT2_NAME}, and {SCRIPT3_NAME} are in the same directory as this orchestrator (diarize-audio.py).")
        return False
    return True

def run_script1(abs_wav_directory, hf_token, is_orchestrator_debug_mode):
    """
    Runs transcribeTHIS.py to process WAV files and create a concatenated transcript.
    Returns the path to the concatenated transcript file if successful, None otherwise.
    """
    logger.info(f"\n--- Running Transcription Script ({SCRIPT1_NAME}) ---")
    logger.info(f"Input WAV Directory: {abs_wav_directory}")

    script1_path = os.path.join(os.getcwd(), SCRIPT1_NAME)
    
    command_to_execute = [
        sys.executable,
        "-u",  # Force Python to run unbuffered
        script1_path,
        abs_wav_directory,
        "--hf_token", hf_token 
    ]

    if is_orchestrator_debug_mode:
        command_to_execute.append("--DEBUG")
    
    command_for_display = list(command_to_execute) 
    try:
        token_flag_index = command_for_display.index("--hf_token")
        if token_flag_index + 1 < len(command_for_display) and command_for_display[token_flag_index + 1] == hf_token:
            token_to_mask_in_display = command_for_display[token_flag_index + 1]
            if token_to_mask_in_display and len(token_to_mask_in_display) > 4:
                masked_token_str = '*' * (len(token_to_mask_in_display) - 4) + token_to_mask_in_display[-4:]
            elif token_to_mask_in_display:
                masked_token_str = '*' * len(token_to_mask_in_display)
            else: 
                masked_token_str = "[EMPTY_OR_INVALID_TOKEN]"
            command_for_display[token_flag_index + 1] = masked_token_str
    except ValueError:
        pass 
    
    logger.info(f"Executing: {' '.join(command_for_display)}")
    
    try:
        # Create process with real-time output
        process = subprocess.Popen(
            command_to_execute,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy(),
            universal_newlines=True
        )

        # Read both stdout and stderr in real-time using select
        import select
        import time

        # Set up file descriptors for select
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        
        while True:
            # Check if process has finished
            if process.poll() is not None:
                break
                
            # Use select to check for available output
            reads = [stdout_fd, stderr_fd]
            ret = select.select(reads, [], [], 0.1)  # 0.1 second timeout
            
            if stdout_fd in ret[0]:
                line = process.stdout.readline()
                if line:
                    print(line.strip(), flush=True)
                    logger.info(line.strip())
                    
            if stderr_fd in ret[0]:
                line = process.stderr.readline()
                if line:
                    print(line.strip(), flush=True)
                    logger.warning(line.strip())
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

        # Read any remaining output
        for line in process.stdout:
            print(line.strip(), flush=True)
            logger.info(line.strip())
            
        for line in process.stderr:
            print(line.strip(), flush=True)
            logger.warning(line.strip())

        return_code = process.poll()

        if return_code == 0:
            logger.info(f"Script '{SCRIPT1_NAME}' completed successfully.")
            dir_basename = os.path.basename(abs_wav_directory)
            expected_filename_suffix = "_transcription_summary.txt"
            concatenated_txt_path = os.path.join(abs_wav_directory, f"{dir_basename}{expected_filename_suffix}")

            if os.path.exists(concatenated_txt_path):
                logger.info(f"Concatenated transcript FOUND by orchestrator: {concatenated_txt_path}")
                return concatenated_txt_path
            else:
                logger.error(f"Error: {SCRIPT1_NAME} finished, but expected output file '{concatenated_txt_path}' was not found.")
                return None
        else:
            logger.error(f"Error: {SCRIPT1_NAME} failed with return code {return_code}.")
            return None

    except FileNotFoundError:
        logger.error(f"Error: Could not find the Python interpreter or the script '{script1_path}'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to run {SCRIPT1_NAME}: {e}")
        return None

def run_script2(transcript_file_path, is_orchestrator_debug_mode, script2_extra_args):
    """
    Runs summarizeTHIS.py with the provided transcript file and extra arguments.
    Returns True if successful, False otherwise.
    """
    logger.info(f"\n--- Running Summarization Script ({SCRIPT2_NAME}) ---")
    logger.info(f"Input Transcript File: {transcript_file_path}")

    script2_path = os.path.join(os.getcwd(), SCRIPT2_NAME)
    command = [
        sys.executable,
        script2_path,
        transcript_file_path
    ]

    if is_orchestrator_debug_mode:
        command.append("--DEBUG")
    
    if script2_extra_args:
        command.extend(script2_extra_args)

    logger.info(f"Executing: {' '.join(command)}") 
    try:
        # Create process with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy()
        )

        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Print immediately
                logger.info(output.strip())

        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print(stderr_output.strip())  # Print immediately
            logger.warning(stderr_output.strip())

        return_code = process.poll()

        if return_code == 0:
            logger.info(f"Script '{SCRIPT2_NAME}' completed successfully.")
            return True
        else:
            logger.error(f"Error: {SCRIPT2_NAME} failed with return code {return_code}.")
            return False

    except FileNotFoundError:
        logger.error(f"Error: Could not find the Python interpreter or the script '{script2_path}'.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to run {SCRIPT2_NAME}: {e}")
        return False

def convert_wavs_to_mp3(abs_wav_directory, is_orchestrator_debug_mode):
    """
    Converts all .WAV files in the given directory to .mp3.
    Deletes original .WAV files if not in debug mode.
    Requires ffmpeg.
    """
    logger.info(f"\n--- Converting WAV files to MP3 in {abs_wav_directory} ---")
    if not shutil.which("ffmpeg"):
        logger.error("Error: ffmpeg command not found. Cannot convert WAV to MP3.")
        logger.error("Please install ffmpeg and ensure it is in your PATH.")
        return False

    wav_files_found = []
    for ext in ("*.WAV", "*.wav"):
        wav_files_found.extend(glob.glob(os.path.join(abs_wav_directory, ext)))

    if not wav_files_found:
        logger.info("No .WAV files found to convert in this directory.")
        return True

    success_count = 0
    failure_count = 0

    for wav_file_path in wav_files_found:
        wav_filename = os.path.basename(wav_file_path)
        base_name, _ = os.path.splitext(wav_filename)
        mp3_filename = base_name + ".mp3"
        mp3_file_path = os.path.join(abs_wav_directory, mp3_filename)

        logger.info(f"Converting {wav_filename} to {mp3_filename}...")
        ffmpeg_command = [
            "ffmpeg", "-i", wav_file_path, "-y", "-vn",
            "-ar", "16000", "-ac", "1", "-b:a", "32k",
            mp3_file_path
        ]

        try:
            # Create process with real-time output
            process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=os.environ.copy()
            )

            # Read output in real-time
            while True:
                output = process.stderr.readline()  # ffmpeg outputs progress to stderr
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print immediately
                    logger.info(output.strip())

            return_code = process.poll()

            if return_code == 0:
                logger.info(f"Successfully converted {wav_filename} to {mp3_filename}.")
                success_count += 1
                if not is_orchestrator_debug_mode:
                    try:
                        os.remove(wav_file_path)
                        logger.info(f"Deleted original WAV file (production mode): {wav_filename}")
                    except Exception as e_del:
                        logger.error(f"Error deleting WAV file {wav_filename}: {e_del}")
            else:
                logger.error(f"Error converting {wav_filename}")
                failure_count += 1

        except Exception as e_conv:
            logger.error(f"An unexpected error occurred during conversion of {wav_filename}: {e_conv}")
            failure_count += 1

    logger.info(f"\nMP3 Conversion Summary: {success_count} succeeded, {failure_count} failed.")
    if failure_count > 0:
        logger.warning("Warning: Some WAV files failed to convert.")
        return False
    return True

def run_script3(date_str, is_orchestrator_debug_mode, script3_extra_args):
    """
    Runs summarize_week.py to generate a weekly summary.
    Returns True if successful, False otherwise.
    """
    logger.info(f"\n--- Running Weekly Summary Script ({SCRIPT3_NAME}) ---")
    logger.info(f"Date for weekly summary: {date_str}")

    script3_path = os.path.join(os.getcwd(), SCRIPT3_NAME)
    command = [
        sys.executable,
        script3_path,
        date_str
    ]

    if is_orchestrator_debug_mode:
        command.append("--DEBUG")
    
    if script3_extra_args:
        command.extend(script3_extra_args)

    logger.info(f"Executing: {' '.join(command)}")
    try:
        # Create process with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy()
        )

        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Print immediately
                logger.info(output.strip())

        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print(stderr_output.strip())  # Print immediately
            logger.warning(stderr_output.strip())

        return_code = process.poll()

        if return_code == 0:
            logger.info(f"Script '{SCRIPT3_NAME}' completed successfully.")
            return True
        else:
            logger.error(f"Error: {SCRIPT3_NAME} failed with return code {return_code}.")
            return False

    except FileNotFoundError:
        logger.error(f"Error: Could not find the Python interpreter or the script '{script3_path}'.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to run {SCRIPT3_NAME}: {e}")
        return False

def main():
    """Main function to orchestrate the script execution."""
    parser = argparse.ArgumentParser(
        description=f"Orchestrates WAV processing ({SCRIPT1_NAME}), summarization ({SCRIPT2_NAME}), and MP3 conversion.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("wav_directory", help="Path to the directory containing .WAV files.")
    parser.add_argument(
        "hf_token", nargs='?', default=os.environ.get("HFTOKEN"),
        help="Hugging Face token. Can also be via HFTOKEN env var."
    )
    parser.add_argument(
        '--production', action='store_true', default=False,
        help="Run in production mode (deletes WAVs after MP3 conversion, no --DEBUG to child scripts). Default is DEBUG mode."
    )
    parser.add_argument(
        '--generate-weekly', action='store_true', default=False,
        help="Generate weekly summary after processing the daily summaries."
    )

    args, remaining_args = parser.parse_known_args()
    is_orchestrator_debug_mode = not args.production

    # Update logger with debug mode (will be DEBUG unless --production is used)
    logger = setup_logging('diarize', is_orchestrator_debug_mode)

    if not check_script_exists(SCRIPT1_NAME) or not check_script_exists(SCRIPT2_NAME):
        sys.exit(1)
    
    if args.generate_weekly and not check_script_exists(SCRIPT3_NAME):
        logger.error(f"Error: {SCRIPT3_NAME} not found. Cannot generate weekly summary.")
        sys.exit(1)

    if not shutil.which("whisperx"):
        logger.warning("Warning: 'whisperx' command not found. Transcription will likely fail.")
    if not shutil.which("ffmpeg"):
        logger.warning("Warning: 'ffmpeg' command not found. MP3 conversion will fail.")

    if not args.hf_token:
        logger.error("Error: Hugging Face token not found as an argument or HFTOKEN environment variable.")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.isdir(args.wav_directory):
        logger.error(f"Error: Specified WAV directory not found: {args.wav_directory}")
        sys.exit(1)

    abs_wav_directory = os.path.abspath(args.wav_directory)

    logger.info("Starting diarization, summarization, and MP3 conversion orchestration...")
    logger.info(f"Mode: {'DEBUG' if is_orchestrator_debug_mode else 'PRODUCTION'}")
    logger.info(f"WAV Directory (Absolute): {abs_wav_directory}")
    hf_token_display = f"'{'*' * (len(args.hf_token) - 4) + args.hf_token[-4:]}'" if args.hf_token and len(args.hf_token) > 4 else "'Provided'"
    logger.info(f"Hugging Face Token: {hf_token_display}")

    concatenated_transcript_file = run_script1(abs_wav_directory, args.hf_token, is_orchestrator_debug_mode)
    if not concatenated_transcript_file:
        logger.error(f"\nOrchestration failed: {SCRIPT1_NAME} did not produce the expected output file correctly.")
        sys.exit(1)

    script2_success = run_script2(concatenated_transcript_file, is_orchestrator_debug_mode, remaining_args)
    if not script2_success:
        logger.error(f"\nOrchestration partially failed: {SCRIPT2_NAME} reported errors.")
        logger.info("Proceeding with MP3 conversion despite summarization issues...")

    mp3_conversion_success = convert_wavs_to_mp3(abs_wav_directory, is_orchestrator_debug_mode)
    if not mp3_conversion_success:
        logger.warning("\nOrchestration warning: MP3 conversion step encountered issues.")

    # Generate weekly summary if requested
    weekly_summary_success = True
    if args.generate_weekly:
        # Extract date from the wav directory name (assuming it's in MMDDYYYY format)
        try:
            dir_name = os.path.basename(abs_wav_directory)
            date_obj = datetime.strptime(dir_name, "%m%d%Y")
            date_str = date_obj.strftime("%Y-%m-%d")
            weekly_summary_success = run_script3(date_str, is_orchestrator_debug_mode, remaining_args)
            if not weekly_summary_success:
                logger.warning("\nOrchestration warning: Weekly summary generation encountered issues.")
        except ValueError:
            logger.error(f"Error: Could not extract date from directory name '{dir_name}'. Weekly summary generation skipped.")
            weekly_summary_success = False

    if not script2_success or not mp3_conversion_success or not weekly_summary_success:
        logger.warning("\n--- Orchestration Finished with warnings/errors ---")
        if not script2_success: sys.exit(2)
        elif not mp3_conversion_success: sys.exit(3)
        elif not weekly_summary_success: sys.exit(4)
    else:
        logger.info("\n--- Orchestration Finished Successfully ---")

if __name__ == "__main__":
    main()

