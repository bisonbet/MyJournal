#!/usr/bin/env python3
import os
import subprocess
import sys
import glob
import shutil
import argparse
import tempfile
import re
from datetime import datetime
import json

def debug_print(message, debug_mode=False):
    """Print message only if debug mode is enabled."""
    if debug_mode:
        print(message)

# Attempt to import optional dependencies and provide guidance if missing
try:
    import soundfile as sf
    import numpy as np
    # from scipy.io import wavfile # Not directly used, but often a dependency for audio tasks
    import noisereduce
except ImportError:
    print("INFO: For noise reduction, please install 'noisereduce', 'soundfile', 'numpy', and 'scipy'.")
    print("INFO: You can typically install them using: pip install noisereduce soundfile numpy scipy")
    noisereduce = None # Set to None if import fails

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntEncodeError
except ImportError:
    print("INFO: For audio normalization, please install 'pydub'.")
    print("INFO: You can typically install it using: pip install pydub")
    print("INFO: 'pydub' also requires ffmpeg or libav to be installed on your system.")
    AudioSegment = None # Set to None if import fails


def apply_noise_reduction(input_path, output_path):
    """
    Applies noise reduction to an audio file.
    Requires 'noisereduce', 'soundfile', 'numpy', 'scipy'.
    """
    if not noisereduce or not sf: # Check for soundfile as well
        print(f"Skipping noise reduction for {os.path.basename(input_path)} as 'noisereduce' or 'soundfile' library is not available.")
        return False
    try:
        print(f"Applying noise reduction to {os.path.basename(input_path)}...")
        data, rate = sf.read(input_path)
        
        if data.ndim > 1:
            print("INFO: Audio is stereo, averaging channels to mono for noise reduction.")
            data_mono = np.mean(data, axis=1)
        else:
            data_mono = data

        reduced_noise_data = noisereduce.reduce_noise(y=data_mono, sr=rate, stationary=False, prop_decrease=0.75)
        
        sf.write(output_path, reduced_noise_data, rate)
        print(f"Noise reduction complete. Saved to: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"Error during noise reduction for {os.path.basename(input_path)}: {e}")
        return False

def apply_normalization(input_path, output_path, target_dbfs=-20.0):
    """
    Normalizes an audio file to a target dBFS.
    Requires 'pydub' and ffmpeg/libav.
    """
    if not AudioSegment:
        print(f"Skipping normalization for {os.path.basename(input_path)} as 'pydub' library is not available.")
        return False
    try:
        print(f"Normalizing {os.path.basename(input_path)} to {target_dbfs} dBFS...")
        sound = AudioSegment.from_file(input_path) # pydub usually auto-detects format
        
        change_in_dbfs = target_dbfs - sound.dBFS
        normalized_sound = sound.apply_gain(change_in_dbfs)
        
        normalized_sound.export(output_path, format="wav")
        print(f"Normalization complete. Saved to: {os.path.basename(output_path)}")
        return True
    except CouldntEncodeError:
        print(f"Error: Could not export normalized audio for {os.path.basename(input_path)}. "
              "Ensure ffmpeg or libav is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"Error during normalization for {os.path.basename(input_path)}: {e}")
        return False

def process_wav_files_in_directory(directory_path, hf_token, args):
    """
    Processes all .WAV files in a given directory using whisperx,
    with optional pre-processing and timestamp-based sorting,
    then concatenates the resulting .txt files.
    """
    if args.DEBUG:
        debug_print("DEBUG mode enabled.", args.DEBUG)

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return

    if not hf_token:
        print("Error: Hugging Face token (HFTOKEN) not provided or found.")
        return

    if not shutil.which("whisperx"):
        print("Error: whisperx command not found. Please ensure it is installed and")
        print("that you have activated the correct Python environment.")
        return

    debug_print(f"Processing directory: {os.path.abspath(directory_path)}", args.DEBUG)
    if args.DEBUG:
        debug_print(f"DEBUG: Absolute directory path: {os.path.abspath(directory_path)}", args.DEBUG)

    # --- File discovery and sorting based on timestamp ---
    initial_wav_files = []
    for ext in ("*.WAV", "*.wav"):
        initial_wav_files.extend(glob.glob(os.path.join(directory_path, ext)))

    if not initial_wav_files:
        print(f"No .WAV or .wav files found in {directory_path}")
        return
    
    if args.DEBUG:
        debug_print(f"DEBUG: Found {len(initial_wav_files)} initial WAV files.", args.DEBUG)

    files_to_process_info = []
    if args.filename_timestamp_format and args.filename_timestamp_regex:
        debug_print(f"Attempting to sort WAV files by timestamp using format '{args.filename_timestamp_format}' and regex '{args.filename_timestamp_regex}'...", args.DEBUG)
        if args.DEBUG:
            debug_print(f"DEBUG: Timestamp format: {args.filename_timestamp_format}", args.DEBUG)
            debug_print(f"DEBUG: Timestamp regex: {args.filename_timestamp_regex}", args.DEBUG)
            
        sortable_files = []
        unsortable_files = []

        for wav_path in initial_wav_files:
            filename = os.path.basename(wav_path)
            try:
                match = re.search(args.filename_timestamp_regex, filename)
                if match and match.group(1):
                    timestamp_str = match.group(1)
                    if args.DEBUG: debug_print(f"DEBUG: File '{filename}', extracted timestamp_str: '{timestamp_str}'", args.DEBUG)
                    dt_obj = datetime.strptime(timestamp_str, args.filename_timestamp_format)
                    sortable_files.append({'datetime': dt_obj, 'path': wav_path, 'original_filename': filename})
                else:
                    debug_print(f"Warning: Timestamp pattern not found or regex group 1 empty in '{filename}'. Will process later.", args.DEBUG)
                    if args.DEBUG: debug_print(f"DEBUG: No match or empty group 1 for '{filename}' with regex '{args.filename_timestamp_regex}'", args.DEBUG)
                    unsortable_files.append({'path': wav_path, 'original_filename': filename})
            except ValueError as ve:
                debug_print(f"Warning: Could not parse timestamp from '{filename}' (extracted: '{timestamp_str if 'timestamp_str' in locals() and match and match.group(1) else 'N/A'}', attempted format: '{args.filename_timestamp_format}'). Error: {ve}. Will process later.", args.DEBUG)
                if args.DEBUG: debug_print(f"DEBUG: ValueError for '{filename}'. Extracted: '{timestamp_str if 'timestamp_str' in locals() and match and match.group(1) else 'N/A'}'. Format: '{args.filename_timestamp_format}'. Error: {ve}", args.DEBUG)
                unsortable_files.append({'path': wav_path, 'original_filename': filename})
            except Exception as e_parse:
                debug_print(f"Warning: Error parsing timestamp for '{filename}': {e_parse}. Will process later.", args.DEBUG)
                if args.DEBUG: debug_print(f"DEBUG: Generic parsing error for '{filename}': {e_parse}", args.DEBUG)
                unsortable_files.append({'path': wav_path, 'original_filename': filename})
        
        sortable_files.sort(key=lambda x: x['datetime'])
        files_to_process_info = sortable_files + unsortable_files 
        if args.DEBUG:
            debug_print(f"DEBUG: {len(sortable_files)} sortable files, {len(unsortable_files)} unsortable files.", args.DEBUG)
    else:
        debug_print("INFO: Timestamp sorting not enabled (format or regex not provided). Processing files in default glob order.", args.DEBUG)
        for wav_path in initial_wav_files:
             files_to_process_info.append({'path': wav_path, 'original_filename': os.path.basename(wav_path)})
    
    debug_print("\nOrder of processing WAV files:", args.DEBUG)
    for i, file_info in enumerate(files_to_process_info):
        debug_print(f"{i+1}. {file_info['original_filename']}", args.DEBUG)
    debug_print("-" * 30, args.DEBUG)

    # Always concatenate WAV files
    debug_print("\nConcatenating WAV files...", args.DEBUG)
    wav_paths = [file_info['path'] for file_info in files_to_process_info]
    concatenated_output = os.path.join(directory_path, f"{os.path.basename(directory_path)}_concatenated.wav")
    concatenated_files = concatenate_wav_files(wav_paths, concatenated_output, args.max_chunk_size, args.DEBUG)
    
    if not concatenated_files:
        print("Error: WAV concatenation failed or produced no output files.")
        return
    
    # Update files_to_process_info with concatenated files
    files_to_process_info = []
    for i, concat_file in enumerate(concatenated_files):
        files_to_process_info.append({
            'path': concat_file,
            'original_filename': os.path.basename(concat_file)
        })
    
    debug_print(f"Created {len(concatenated_files)} concatenated WAV chunks.", args.DEBUG)

    generated_txt_files = []

    for file_info in files_to_process_info:
        wav_file_path = file_info['path']
        original_wav_filename = file_info['original_filename'] 
        original_base_name_no_ext = os.path.splitext(original_wav_filename)[0]
        
        current_audio_path = wav_file_path
        
        debug_print(f"\n--- Processing: {original_wav_filename} ---", args.DEBUG)
        if args.DEBUG: debug_print(f"DEBUG: Starting processing for {original_wav_filename}, original path: {wav_file_path}", args.DEBUG)

        with tempfile.TemporaryDirectory(prefix=f"{original_base_name_no_ext}_processed_", dir=directory_path) as temp_proc_dir:
            if args.DEBUG: debug_print(f"DEBUG: Created temporary processing directory: {temp_proc_dir}", args.DEBUG)
            if args.enable_noise_reduction:
                if not noisereduce or not sf:
                    debug_print(f"INFO: Skipping noise reduction for {original_wav_filename} due to missing libraries.", args.DEBUG)
                else:
                    temp_nr_path = os.path.join(temp_proc_dir, f"{original_base_name_no_ext}_nr.wav")
                    if args.DEBUG: debug_print(f"DEBUG: Attempting noise reduction. Input: {current_audio_path}, Output: {temp_nr_path}", args.DEBUG)
                    if apply_noise_reduction(current_audio_path, temp_nr_path):
                        current_audio_path = temp_nr_path
            else:
                debug_print(f"INFO: Noise reduction disabled by user for {original_wav_filename}.", args.DEBUG)

            if args.enable_normalization:
                if not AudioSegment:
                     debug_print(f"INFO: Skipping normalization for {original_wav_filename} due to missing pydub or ffmpeg/libav.", args.DEBUG)
                else:
                    temp_norm_path = os.path.join(temp_proc_dir, f"{original_base_name_no_ext}_norm.wav")
                    if args.DEBUG: debug_print(f"DEBUG: Attempting normalization. Input: {current_audio_path}, Output: {temp_norm_path}, Target dBFS: {args.normalization_target_dbfs}", args.DEBUG)
                    if apply_normalization(current_audio_path, temp_norm_path, args.normalization_target_dbfs):
                        current_audio_path = temp_norm_path
            else:
                debug_print(f"INFO: Normalization disabled by user for {original_wav_filename}.", args.DEBUG)
            
            debug_print(f"Running WhisperX on: {os.path.basename(current_audio_path)} (derived from {original_wav_filename})...", args.DEBUG)
            command = [
                "whisperx", current_audio_path,
                "--model", "large-v3", "--diarize", "--language", "en",
                "--highlight_words", "True", "--hf_token", hf_token,
                "--output_dir", directory_path, 
                "--vad_onset", str(args.vad_onset), "--vad_offset", str(args.vad_offset)
            ]
            if args.compute_type:
                command.extend(["--compute_type", args.compute_type])
            
            if args.DEBUG: debug_print(f"DEBUG: WhisperX command: {' '.join(command)}", args.DEBUG)

            try:
                process_env = os.environ.copy()
                process_env["HFTOKEN"] = hf_token 
                result = subprocess.run(command, capture_output=True, text=True, env=process_env, check=False)

                if args.DEBUG:
                    debug_print(f"DEBUG: WhisperX process completed. Return code: {result.returncode}", args.DEBUG)
                    if result.stdout: debug_print(f"DEBUG: WhisperX Stdout:\n{result.stdout}", args.DEBUG)
                    if result.stderr: debug_print(f"DEBUG: WhisperX Stderr:\n{result.stderr}", args.DEBUG)

                if result.returncode == 0:
                    debug_print(f"WhisperX completed for {original_wav_filename}.", args.DEBUG)
                    
                    whisperx_input_basename_for_txt = os.path.splitext(os.path.basename(current_audio_path))[0]
                    actual_txt_generated_by_whisperx = os.path.join(directory_path, f"{whisperx_input_basename_for_txt}.txt")
                    target_final_txt_path = os.path.join(directory_path, f"{original_base_name_no_ext}.txt")
                    if args.DEBUG:
                        debug_print(f"DEBUG: WhisperX input basename for TXT: {whisperx_input_basename_for_txt}", args.DEBUG)
                        debug_print(f"DEBUG: Actual TXT generated by WhisperX (expected): {actual_txt_generated_by_whisperx}", args.DEBUG)
                        debug_print(f"DEBUG: Target final TXT path: {target_final_txt_path}", args.DEBUG)

                    if os.path.exists(actual_txt_generated_by_whisperx):
                        if actual_txt_generated_by_whisperx != target_final_txt_path:
                            try:
                                if os.path.exists(target_final_txt_path):
                                    debug_print(f"Warning: Target file {os.path.basename(target_final_txt_path)} already exists. Overwriting.", args.DEBUG)
                                    if args.DEBUG: debug_print(f"DEBUG: Overwriting existing target file: {target_final_txt_path}", args.DEBUG)
                                    os.remove(target_final_txt_path)
                                shutil.move(actual_txt_generated_by_whisperx, target_final_txt_path)
                                debug_print(f"Renamed WhisperX output {os.path.basename(actual_txt_generated_by_whisperx)} to {os.path.basename(target_final_txt_path)}", args.DEBUG)
                                if args.DEBUG: debug_print(f"DEBUG: Successfully renamed {os.path.basename(actual_txt_generated_by_whisperx)} to {os.path.basename(target_final_txt_path)}", args.DEBUG)
                            except Exception as e_rename:
                                debug_print(f"Error renaming {os.path.basename(actual_txt_generated_by_whisperx)} to {os.path.basename(target_final_txt_path)}: {e_rename}. Using original WhisperX output name.", args.DEBUG)
                                if args.DEBUG: debug_print(f"DEBUG: Renaming failed. Error: {e_rename}", args.DEBUG)
                                target_final_txt_path = actual_txt_generated_by_whisperx 
                        
                        generated_txt_files.append(target_final_txt_path)
                        debug_print(f"Successfully processed and prepared transcription: {os.path.basename(target_final_txt_path)}", args.DEBUG)
                    else:
                        debug_print(f"Warning: WhisperX reported success, but expected output file {os.path.basename(actual_txt_generated_by_whisperx)} not found.", args.DEBUG)
                else:
                    debug_print(f"Error processing {original_wav_filename} with WhisperX:", args.DEBUG)
                    if not args.DEBUG: # If not in debug, print stdout/stderr here for errors
                        if result.stdout: debug_print(f"WhisperX Stdout:\n{result.stdout}", args.DEBUG)
                        if result.stderr: debug_print(f"WhisperX Stderr:\n{result.stderr}", args.DEBUG)
            except Exception as e:
                debug_print(f"An unexpected error occurred while running WhisperX for {original_wav_filename}: {e}", args.DEBUG)
                if args.DEBUG: debug_print(f"DEBUG: Exception during WhisperX subprocess run: {e}", args.DEBUG)
            
            if args.DEBUG: debug_print(f"DEBUG: Exiting temporary directory context for {original_wav_filename}. Temp dir {temp_proc_dir} will be cleaned up.", args.DEBUG)

    if not generated_txt_files:
        debug_print("\nNo .txt files were generated by WhisperX. Skipping concatenation.", args.DEBUG)
        return

    abs_directory_path = os.path.abspath(directory_path)
    directory_basename = os.path.basename(abs_directory_path)
    concatenated_file_name = os.path.join(abs_directory_path, f"{directory_basename}_transcription_summary.txt")

    debug_print(f"\nConcatenating {len(generated_txt_files)} .txt files into: {concatenated_file_name}", args.DEBUG)
    if args.DEBUG: debug_print(f"DEBUG: Concatenating files: {generated_txt_files}", args.DEBUG)
    
    with open(concatenated_file_name, "w", encoding="utf-8") as outfile:
        for txt_file_path in generated_txt_files: 
            try:
                with open(txt_file_path, "r", encoding="utf-8") as infile:
                    base_txt_filename = os.path.basename(txt_file_path)
                    outfile.write(f"--- Content from: {base_txt_filename} ---\n\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n" + "="*80 + "\n\n")
                debug_print(f"Added content from {base_txt_filename}", args.DEBUG)
            except Exception as e:
                debug_print(f"Error reading or writing {txt_file_path}: {e}", args.DEBUG)
    
    debug_print("\nScript finished.", args.DEBUG)
    debug_print(f"All relevant .txt files concatenated into {concatenated_file_name}", args.DEBUG)

def run_ffmpeg_command(cmd, debug_mode=False):
    """
    Runs an ffmpeg command and prints output in real-time.
    Returns True if successful, False otherwise.
    """
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        if return_code != 0:
            print(f"Error: ffmpeg command failed with return code {return_code}")
            return False
            
        return True
    except Exception as e:
        print(f"Error running ffmpeg command: {e}")
        return False

def get_wav_settings(wav_file):
    """
    Gets the audio settings from a WAV file using ffmpeg.
    Returns a tuple of (sample_rate, channels, codec, bitrate).
    Defaults to 24kHz, stereo, 192kbit if settings can't be detected.
    """
    try:
        # Run ffprobe to get audio stream information
        cmd = [
            "ffprobe", 
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,codec_name,bit_rate",
            "-of", "json",
            wav_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract settings from the first audio stream
        stream = info['streams'][0]
        codec = stream.get('codec_name', 'pcm_s16le')
        
        # If the input is ADPCM, we'll use pcm_s16le for output
        if codec == 'adpcm_ima_wav':
            codec = 'pcm_s16le'
            
        return (
            str(stream.get('sample_rate', '24000')),  # Default to 24kHz
            str(stream.get('channels', '2')),         # Default to stereo
            codec,
            str(stream.get('bit_rate', '192000'))     # Default to 192kbit
        )
    except Exception as e:
        print(f"Warning: Could not get WAV settings from {wav_file}, using defaults: {e}")
        return ('24000', '2', 'pcm_s16le', '192000')  # Default values: 24kHz, stereo, 192kbit

def concatenate_wav_files(wav_files, output_path, max_size_gb=3.5, debug_mode=False):
    """
    Concatenates WAV files into chunks that don't exceed max_size_gb.
    Uses original WAV file settings for output, or defaults to 24kHz stereo 192kbit.
    Returns a list of paths to the concatenated files.
    """
    if not AudioSegment:
        print("Error: pydub library not available. Cannot concatenate WAV files.")
        return []

    if not wav_files:
        print("Error: No WAV files provided for concatenation.")
        return []

    debug_print(f"Starting WAV concatenation. Max chunk size: {max_size_gb}GB", debug_mode)
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
    # Get settings from the first WAV file
    sample_rate, channels, codec, bitrate = get_wav_settings(wav_files[0])
    debug_print(f"Using audio settings: {sample_rate}Hz, {channels} channels, {codec}, {int(bitrate)/1000}kbit", debug_mode)
    
    concatenated_files = []
    current_chunk = None
    current_chunk_size = 0
    chunk_index = 0
    
    for wav_file in wav_files:
        try:
            debug_print(f"Processing {os.path.basename(wav_file)}", debug_mode)
            # Use ffmpeg directly to read the file
            temp_wav = os.path.join(os.path.dirname(wav_file), f"temp_{os.path.basename(wav_file)}")
            try:
                # First convert to standard WAV format
                convert_cmd = [
                    "ffmpeg", "-y",
                    "-i", wav_file,
                    "-acodec", "pcm_s16le",
                    "-ar", sample_rate,
                    "-ac", channels,
                    temp_wav
                ]
                if not run_ffmpeg_command(convert_cmd, debug_mode):
                    raise Exception("Failed to convert WAV file")
                
                # Now load the converted file
                audio = AudioSegment.from_wav(temp_wav)
                
                # If this is the first file or adding this file would exceed max size
                if current_chunk is None or (current_chunk_size + len(audio.raw_data) > max_size_bytes):
                    # Save previous chunk if it exists
                    if current_chunk is not None:
                        chunk_path = os.path.join(os.path.dirname(output_path), 
                                                f"{os.path.splitext(os.path.basename(output_path))[0]}_chunk{chunk_index}.wav")
                        # Export using detected or default settings
                        current_chunk.export(chunk_path, 
                                           format="wav",
                                           parameters=["-acodec", "pcm_s16le",
                                                     "-ar", sample_rate,
                                                     "-ac", channels,
                                                     "-b:a", bitrate])
                        concatenated_files.append(chunk_path)
                        debug_print(f"Created chunk {chunk_index}: {os.path.basename(chunk_path)}", debug_mode)
                        chunk_index += 1
                    
                    # Start new chunk
                    current_chunk = audio
                    current_chunk_size = len(audio.raw_data)
                else:
                    # Add to current chunk
                    current_chunk += audio
                    current_chunk_size += len(audio.raw_data)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    # Save the last chunk if it exists
    if current_chunk is not None:
        chunk_path = os.path.join(os.path.dirname(output_path), 
                                f"{os.path.splitext(os.path.basename(output_path))[0]}_chunk{chunk_index}.wav")
        current_chunk.export(chunk_path, 
                           format="wav",
                           parameters=["-acodec", "pcm_s16le",
                                     "-ar", sample_rate,
                                     "-ac", channels,
                                     "-b:a", bitrate])
        concatenated_files.append(chunk_path)
        debug_print(f"Created final chunk {chunk_index}: {os.path.basename(chunk_path)}", debug_mode)
    
    return concatenated_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process .WAV files with WhisperX. Sorts by filename timestamp if format/regex provided. Noise reduction and normalization are ON by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("directory_path", help="Path to the directory containing .WAV files.")
    parser.add_argument("--hf_token", help="Hugging Face token. Can also be set via HFTOKEN environment variable.", default=os.environ.get("HFTOKEN"))
    
    # Debug Flag
    parser.add_argument("--DEBUG", action="store_true", help="Enable debug mode for verbose logging and diagnostics.")

    # Filename timestamp parsing arguments
    timestamp_group = parser.add_argument_group('Timestamp Sorting Options')
    timestamp_group.add_argument("--filename-timestamp-format", type=str, default="%m_%d_%y-%H_%M_%S",
                        help="datetime.strptime format for timestamps in filenames (e.g., '%%m_%%d_%%y-%%H_%%M_%%S').")
    timestamp_group.add_argument("--filename-timestamp-regex", type=str, default=r"(\d{2}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})",
                        help="Regex to extract timestamp from filename. Must have one capture group for the timestamp string.")

    # Pre-processing arguments
    preprocessing_group = parser.add_argument_group('Pre-processing Options (ON by default)')
    preprocessing_group.add_argument("--disable-noise-reduction", action="store_false", dest="enable_noise_reduction", help="Disable noise reduction.")
    preprocessing_group.add_argument("--disable-normalization", action="store_false", dest="enable_normalization", help="Disable audio normalization.")
    preprocessing_group.add_argument("--normalization-target-dbfs", type=float, default=-20.0, help="Target dBFS for normalization.")
    parser.set_defaults(enable_noise_reduction=True, enable_normalization=True)
    
    # WhisperX specific arguments
    whisperx_group = parser.add_argument_group('WhisperX Options')
    whisperx_group.add_argument("--vad-onset", type=float, default=0.5, help="WhisperX VAD onset threshold.")
    whisperx_group.add_argument("--vad-offset", type=float, default=0.363, help="WhisperX VAD offset threshold.")
    whisperx_group.add_argument("--compute_type", type=str, default="float16", help="Compute type for WhisperX (e.g., float16, int8).")

    # WAV concatenation arguments
    concat_group = parser.add_argument_group('WAV Concatenation Options')
    concat_group.add_argument("--max-chunk-size", type=float, default=3.5, help="Maximum size of concatenated WAV chunks in GB (default: 3.5GB).")

    args = parser.parse_args()

    if args.DEBUG:
        debug_print("--- Parsed Arguments ---", args.DEBUG)
        for arg, value in vars(args).items():
            debug_print(f"{arg}: {value}", args.DEBUG)
        debug_print("------------------------", args.DEBUG)


    if not args.hf_token:
        print("Error: Hugging Face token not found. Set HFTOKEN env var or use --hf_token.")
        sys.exit(1)
    
    # --- How to use this script ---
    # (Installation instructions for dependencies remain the same)
    #
    # Example (default timestamp format 'MM_DD_YY-HH_MM_SS', e.g., prefix_05_25_25-04_30_53_suffix.wav, pre-processing ON):
    # ./script.py /path/to/wavs --hf_token YOUR_TOKEN
    #
    # Example (with DEBUG mode):
    # ./script.py /path/to/wavs --hf_token YOUR_TOKEN --DEBUG
    #
    # Example (disabling noise reduction, keeping default timestamping, pre-processing ON for norm):
    # ./script.py /path/to/wavs --hf_token YOUR_TOKEN --disable-noise-reduction

    process_wav_files_in_directory(args.directory_path, args.hf_token, args)

