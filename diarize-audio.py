#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
import glob # Added for finding WAV files

# --- Configuration ---
SCRIPT1_NAME = "transcribeTHIS.py"
SCRIPT2_NAME = "summarizeTHIS.py"

def check_script_exists(script_name):
    """Checks if a script exists in the PWD."""
    script_path = os.path.join(os.getcwd(), script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_name}' not found in the current directory ({os.getcwd()}).")
        print(f"Please ensure {SCRIPT1_NAME} and {SCRIPT2_NAME} are in the same directory as this orchestrator (diarize-audio.py).")
        return False
    return True

def run_script1(abs_wav_directory, hf_token, is_orchestrator_debug_mode):
    """
    Runs transcribeTHIS.py to process WAV files and create a concatenated transcript.
    Returns the path to the concatenated transcript file if successful, None otherwise.
    """
    print(f"\n--- Running Transcription Script ({SCRIPT1_NAME}) ---")
    print(f"Input WAV Directory: {abs_wav_directory}")

    script1_path = os.path.join(os.getcwd(), SCRIPT1_NAME)
    
    command_to_execute = [
        sys.executable,
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
    
    print(f"Executing: {' '.join(command_for_display)}")
    
    try:
        process = subprocess.run(command_to_execute, capture_output=True, text=True, check=False, env=os.environ.copy())

        if process.returncode == 0:
            print(f"Script '{SCRIPT1_NAME}' completed successfully.") # This doesn't mean the file exists yet, just that the script exited 0
            # We rely on the script's own output to know the filename if it's dynamic,
            # OR we hardcode the expected name if it's fixed.
            # Based on user's latest error, transcribeTHIS.py now outputs a file named
            # {directory_basename}_transcription_summary.txt

            if process.stdout:
                print(f"\n{SCRIPT1_NAME} STDOUT:")
                print(process.stdout)
            if process.stderr:
                print(f"\n{SCRIPT1_NAME} STDERR (may include info from whisperx):")
                print(process.stderr)

            dir_basename = os.path.basename(abs_wav_directory)
            # MODIFICATION: Update expected filename based on new information
            expected_filename_suffix = "_transcription_summary.txt"
            concatenated_txt_path = os.path.join(abs_wav_directory, f"{dir_basename}{expected_filename_suffix}")

            if os.path.exists(concatenated_txt_path):
                print(f"Concatenated transcript FOUND by orchestrator: {concatenated_txt_path}")
                return concatenated_txt_path
            else:
                # Attempt to find the filename from transcribeTHIS.py's stdout as a fallback
                # This is a bit fragile but can help if the naming convention is slightly off or changes.
                stdout_last_line = ""
                if process.stdout:
                    lines = process.stdout.strip().split('\n')
                    if lines:
                        # Example line: "All relevant .txt files concatenated into /path/to/file.txt"
                        search_phrase = "concatenated into "
                        for line in reversed(lines):
                            if search_phrase in line:
                                stdout_last_line = line
                                break
                
                if stdout_last_line:
                    try:
                        # Extract the path from the stdout line
                        actual_path_from_stdout = stdout_last_line.split(search_phrase, 1)[1].strip()
                        if os.path.exists(actual_path_from_stdout):
                            print(f"Warning: Expected file '{concatenated_txt_path}' not found.")
                            print(f"However, found file mentioned in {SCRIPT1_NAME}'s STDOUT: '{actual_path_from_stdout}'")
                            print(f"Proceeding with file: {actual_path_from_stdout}")
                            return actual_path_from_stdout
                        else:
                            print(f"Error: {SCRIPT1_NAME} finished, but expected output file '{concatenated_txt_path}' was not found.")
                            print(f"Also, the path mentioned in {SCRIPT1_NAME}'s STDOUT ('{actual_path_from_stdout}') does not exist or could not be parsed correctly.")
                            return None
                    except Exception as e_parse:
                         print(f"Error: {SCRIPT1_NAME} finished, but expected output file '{concatenated_txt_path}' was not found.")
                         print(f"Tried to parse filename from {SCRIPT1_NAME} stdout but failed: {e_parse}")
                         return None       
                else:
                    print(f"Error: {SCRIPT1_NAME} finished, but expected output file '{concatenated_txt_path}' was not found.")
                    print(f"Could not determine actual output filename from {SCRIPT1_NAME}'s STDOUT.")
                    return None
        else:
            print(f"Error: {SCRIPT1_NAME} failed with return code {process.returncode}.")
            if process.stdout: print(f"\n{SCRIPT1_NAME} STDOUT:\n{process.stdout}")
            if process.stderr: print(f"\n{SCRIPT1_NAME} STDERR:\n{process.stderr}")
            return None
    except FileNotFoundError:
        print(f"Error: Could not find the Python interpreter or the script '{script1_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {SCRIPT1_NAME}: {e}")
        return None

def run_script2(transcript_file_path, is_orchestrator_debug_mode, script2_extra_args):
    """
    Runs summarizeTHIS.py with the provided transcript file and extra arguments.
    Returns True if successful, False otherwise.
    """
    print(f"\n--- Running Summarization Script ({SCRIPT2_NAME}) ---")
    print(f"Input Transcript File: {transcript_file_path}")

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

    print(f"Executing: {' '.join(command)}") 
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, env=os.environ.copy())

        print(f"\n{SCRIPT2_NAME} Output:")
        if process.stdout: print(f"STDOUT:\n{process.stdout}")
        if process.stderr: print(f"STDERR:\n{process.stderr}")

        if process.returncode == 0:
            print(f"Script '{SCRIPT2_NAME}' completed successfully.")
            return True
        else:
            print(f"Error: {SCRIPT2_NAME} failed with return code {process.returncode}.")
            return False
    except FileNotFoundError:
        print(f"Error: Could not find the Python interpreter or the script '{script2_path}'.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {SCRIPT2_NAME}: {e}")
        return False

def convert_wavs_to_mp3(abs_wav_directory, is_orchestrator_debug_mode):
    """
    Converts all .WAV files in the given directory to .mp3.
    Deletes original .WAV files if not in debug mode.
    Requires ffmpeg.
    """
    print(f"\n--- Converting WAV files to MP3 in {abs_wav_directory} ---")
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg command not found. Cannot convert WAV to MP3.")
        print("Please install ffmpeg and ensure it is in your PATH.")
        return False

    wav_files_found = []
    for ext in ("*.WAV", "*.wav"):
        wav_files_found.extend(glob.glob(os.path.join(abs_wav_directory, ext)))

    if not wav_files_found:
        print("No .WAV files found to convert in this directory.")
        return True

    success_count = 0
    failure_count = 0

    for wav_file_path in wav_files_found:
        wav_filename = os.path.basename(wav_file_path)
        base_name, _ = os.path.splitext(wav_filename)
        mp3_filename = base_name + ".mp3"
        mp3_file_path = os.path.join(abs_wav_directory, mp3_filename)

        print(f"Converting {wav_filename} to {mp3_filename}...")
        ffmpeg_command = [
            "ffmpeg", "-i", wav_file_path, "-y", "-vn",
            "-ar", "16000", "-ac", "1", "-b:a", "32k",
            mp3_file_path
        ]

        try:
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(f"Successfully converted {wav_filename} to {mp3_filename}.")
                success_count += 1
                if not is_orchestrator_debug_mode:
                    try:
                        os.remove(wav_file_path)
                        print(f"Deleted original WAV file (production mode): {wav_filename}")
                    except Exception as e_del:
                        print(f"Error deleting WAV file {wav_filename}: {e_del}")
            else:
                print(f"Error converting {wav_filename}:")
                if result.stderr: print(f"ffmpeg STDERR:\n{result.stderr}")
                else: print("No STDERR from ffmpeg.")
                failure_count += 1
        except Exception as e_conv:
            print(f"An unexpected error occurred during conversion of {wav_filename}: {e_conv}")
            failure_count += 1

    print(f"\nMP3 Conversion Summary: {success_count} succeeded, {failure_count} failed.")
    if failure_count > 0:
        print("Warning: Some WAV files failed to convert.")
        return False
    return True

def main():
    """Main function to orchestrate the script execution."""
    if not check_script_exists(SCRIPT1_NAME) or not check_script_exists(SCRIPT2_NAME):
        sys.exit(1)

    if not shutil.which("whisperx"):
        print("Warning: 'whisperx' command not found. Transcription will likely fail.")
    if not shutil.which("ffmpeg"):
        print("Warning: 'ffmpeg' command not found. MP3 conversion will fail.")

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

    args, script2_extra_args = parser.parse_known_args()
    is_orchestrator_debug_mode = not args.production

    if not args.hf_token:
        print("Error: Hugging Face token not found as an argument or HFTOKEN environment variable.")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.isdir(args.wav_directory):
        print(f"Error: Specified WAV directory not found: {args.wav_directory}")
        sys.exit(1)

    abs_wav_directory = os.path.abspath(args.wav_directory)

    print("Starting diarization, summarization, and MP3 conversion orchestration...")
    print(f"Mode: {'DEBUG' if is_orchestrator_debug_mode else 'PRODUCTION'}")
    print(f"WAV Directory (Absolute): {abs_wav_directory}")
    hf_token_display = f"'{'*' * (len(args.hf_token) - 4) + args.hf_token[-4:]}'" if args.hf_token and len(args.hf_token) > 4 else "'Provided'"
    print(f"Hugging Face Token: {hf_token_display}")

    concatenated_transcript_file = run_script1(abs_wav_directory, args.hf_token, is_orchestrator_debug_mode)
    if not concatenated_transcript_file:
        print(f"\nOrchestration failed: {SCRIPT1_NAME} did not produce the expected output file correctly.")
        sys.exit(1)

    script2_success = run_script2(concatenated_transcript_file, is_orchestrator_debug_mode, script2_extra_args)
    if not script2_success:
        print(f"\nOrchestration partially failed: {SCRIPT2_NAME} reported errors.")
        print("Proceeding with MP3 conversion despite summarization issues...")

    mp3_conversion_success = convert_wavs_to_mp3(abs_wav_directory, is_orchestrator_debug_mode)
    if not mp3_conversion_success:
        print("\nOrchestration warning: MP3 conversion step encountered issues.")

    if not script2_success or not mp3_conversion_success:
        print("\n--- Orchestration Finished with warnings/errors ---")
        if not script2_success : sys.exit(2)
        elif not mp3_conversion_success: sys.exit(3) 
    else:
        print("\n--- Orchestration Finished Successfully ---")

if __name__ == "__main__":
    main()

