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
import logging
from pathlib import Path
from logging_config import setup_logging
import platform  # Add platform import for OS detection

# Force Python to run unbuffered
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
sys.stderr.reconfigure(line_buffering=True)  # Python 3.7+

# Force flush on all output
def force_flush():
    sys.stdout.flush()
    sys.stderr.flush()

# Initialize logger
logger = setup_logging('transcribe', False)  # Default to non-debug mode

# Ensure logging is unbuffered
for handler in logger.handlers:
    handler.setLevel(logging.INFO)
    if isinstance(handler, logging.StreamHandler):
        handler.flush = handler.stream.flush

def debug_print(message, debug_mode=False):
    """Print debug messages if debug mode is enabled."""
    if debug_mode:
        logger.debug(message)
        force_flush()

# Determine default device and compute type based on OS
if platform.system() == "Darwin":
    DEFAULT_DEVICE = "mps"
    DEFAULT_COMPUTE_TYPE = "float32"  # Changed from 'auto' to 'float32' for macOS
else:
    DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    DEFAULT_COMPUTE_TYPE = "int8"  # Or "float32" if int8 doesn't work

# Attempt to import optional dependencies and provide guidance if missing
try:
    import torch
    import torchaudio
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    # Set device for torch operations
    DEVICE = torch.device("mps" if platform.system() == "Darwin" else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {DEVICE} for audio processing")
except ImportError:
    logger.warning("INFO: For GPU-accelerated audio processing, please install 'torch' and 'torchaudio'.")
    logger.warning("INFO: You can typically install them using: pip install torch torchaudio")
    TORCH_AVAILABLE = False

try:
    import soundfile as sf
    import numpy as np
    import noisereduce
except ImportError:
    logger.warning("INFO: For noise reduction, please install 'noisereduce', 'soundfile', 'numpy', and 'scipy'.")
    logger.warning("INFO: You can typically install them using: pip install noisereduce soundfile numpy scipy")
    noisereduce = None

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntEncodeError
except ImportError:
    logger.warning("INFO: For audio normalization, please install 'pydub'.")
    logger.warning("INFO: You can typically install it using: pip install pydub")
    logger.warning("INFO: 'pydub' also requires ffmpeg or libav to be installed on your system.")
    AudioSegment = None

def apply_noise_reduction(input_path, output_path):
    """
    Applies noise reduction to an audio file using GPU acceleration if available.
    Processes large files in chunks to avoid memory issues.
    """
    if TORCH_AVAILABLE:
        try:
            logger.info(f"Applying GPU-accelerated noise reduction to {os.path.basename(input_path)}...")
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(input_path)
            
            # Check if file is too large for GPU memory
            total_samples = waveform.shape[-1]
            chunk_size = 30 * sample_rate  # 30 seconds per chunk
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            if num_chunks > 1:
                logger.info(f"File is large ({total_samples/sample_rate:.1f} seconds). Processing in {num_chunks} chunks...")
                
                # Process in chunks
                processed_chunks = []
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, total_samples)
                    logger.info(f"Processing chunk {i+1}/{num_chunks} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)...")
                    
                    # Extract chunk
                    chunk = waveform[:, start:end].to(DEVICE)
                    
                    # Convert to mono if stereo
                    if chunk.shape[0] > 1:
                        chunk = torch.mean(chunk, dim=0, keepdim=True)
                    
                    # Apply noise reduction using spectral gating
                    stft = torch.stft(chunk, n_fft=2048, hop_length=512, return_complex=True)
                    magnitude = torch.abs(stft)
                    phase = torch.angle(stft)
                    
                    # Estimate noise floor
                    noise_floor = torch.mean(magnitude, dim=-1, keepdim=True)
                    threshold = noise_floor * 1.5
                    
                    # Apply spectral gating
                    mask = (magnitude > threshold).float()
                    magnitude_filtered = magnitude * mask
                    
                    # Convert back to time domain
                    stft_filtered = magnitude_filtered * torch.exp(1j * phase)
                    chunk_filtered = torch.istft(stft_filtered, n_fft=2048, hop_length=512, length=chunk.shape[-1])
                    
                    # Move back to CPU and store
                    processed_chunks.append(chunk_filtered.cpu())
                    
                    # Clear GPU memory
                    del chunk, stft, magnitude, phase, stft_filtered, chunk_filtered
                    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
                
                # Concatenate processed chunks
                waveform_filtered = torch.cat(processed_chunks, dim=-1)
                
            else:
                # Process entire file at once
                waveform = waveform.to(DEVICE)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    logger.info("INFO: Audio is stereo, averaging channels to mono for noise reduction.")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Apply noise reduction using spectral gating
                stft = torch.stft(waveform, n_fft=2048, hop_length=512, return_complex=True)
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                
                # Estimate noise floor
                noise_floor = torch.mean(magnitude, dim=-1, keepdim=True)
                threshold = noise_floor * 1.5
                
                # Apply spectral gating
                mask = (magnitude > threshold).float()
                magnitude_filtered = magnitude * mask
                
                # Convert back to time domain
                stft_filtered = magnitude_filtered * torch.exp(1j * phase)
                waveform_filtered = torch.istft(stft_filtered, n_fft=2048, hop_length=512, length=waveform.shape[-1])
                
                # Move back to CPU
                waveform_filtered = waveform_filtered.cpu()
            
            # Save using torchaudio
            torchaudio.save(output_path, waveform_filtered, sample_rate)
            logger.info(f"GPU-accelerated noise reduction complete. Saved to: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error during GPU noise reduction: {e}")
            logger.info("Falling back to CPU-based noise reduction...")
    
    # Fallback to CPU-based noise reduction
    if not noisereduce or not sf:
        logger.warning(f"Skipping noise reduction for {os.path.basename(input_path)} as required libraries are not available.")
        return False
    
    try:
        logger.info(f"Applying CPU-based noise reduction to {os.path.basename(input_path)}...")
        data, rate = sf.read(input_path)
        
        if data.ndim > 1:
            logger.info("INFO: Audio is stereo, averaging channels to mono for noise reduction.")
            data_mono = np.mean(data, axis=1)
        else:
            data_mono = data

        reduced_noise_data = noisereduce.reduce_noise(y=data_mono, sr=rate, stationary=False, prop_decrease=0.75)
        
        sf.write(output_path, reduced_noise_data, rate)
        logger.info(f"Noise reduction complete. Saved to: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        logger.error(f"Error during noise reduction for {os.path.basename(input_path)}: {e}")
        return False

def apply_normalization(input_path, output_path, target_dbfs=-20.0):
    """
    Normalizes an audio file using GPU acceleration if available.
    """
    if TORCH_AVAILABLE:
        try:
            logger.info(f"Applying GPU-accelerated normalization to {os.path.basename(input_path)}...")
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(input_path)
            
            # Move to GPU if available
            waveform = waveform.to(DEVICE)
            
            # Calculate current RMS
            rms = torch.sqrt(torch.mean(waveform ** 2))
            current_dbfs = 20 * torch.log10(rms)
            
            # Calculate gain needed
            gain_db = target_dbfs - current_dbfs
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply gain
            normalized_waveform = waveform * gain_linear
            
            # Move back to CPU for saving
            normalized_waveform = normalized_waveform.cpu()
            
            # Save using torchaudio
            torchaudio.save(output_path, normalized_waveform, sample_rate)
            logger.info(f"GPU-accelerated normalization complete. Saved to: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error during GPU normalization: {e}")
            logger.info("Falling back to CPU-based normalization...")
    
    # Fallback to CPU-based normalization
    if not AudioSegment:
        logger.warning(f"Skipping normalization for {os.path.basename(input_path)} as 'pydub' library is not available.")
        return False
    
    try:
        logger.info(f"Normalizing {os.path.basename(input_path)} to {target_dbfs} dBFS...")
        sound = AudioSegment.from_file(input_path)
        
        change_in_dbfs = target_dbfs - sound.dBFS
        normalized_sound = sound.apply_gain(change_in_dbfs)
        
        normalized_sound.export(output_path, format="wav")
        logger.info(f"Normalization complete. Saved to: {os.path.basename(output_path)}")
        return True
    except CouldntEncodeError:
        logger.error(f"Error: Could not export normalized audio for {os.path.basename(input_path)}. "
              "Ensure ffmpeg or libav is installed and in your system's PATH.")
        return False
    except Exception as e:
        logger.error(f"Error during normalization for {os.path.basename(input_path)}: {e}")
        return False

def process_wav_files_in_directory(directory_path, hf_token, args):
    """
    Processes all .WAV files in a given directory using whisperx,
    with optional pre-processing and timestamp-based sorting,
    then concatenates the resulting .txt files.
    """
    logger.info("\n=== Starting Audio Processing Pipeline ===")
    force_flush()
    logger.info(f"Processing directory: {os.path.abspath(directory_path)}")
    force_flush()
    logger.info(f"Using device: {args.device}")
    force_flush()
    logger.info(f"Using compute type: {args.compute_type}")
    force_flush()
    logger.info(f"Noise reduction: {'Enabled' if args.enable_noise_reduction else 'Disabled'}")
    force_flush()
    logger.info(f"Normalization: {'Enabled' if args.enable_normalization else 'Disabled'}")
    force_flush()
    if args.enable_normalization:
        logger.info(f"Normalization target dBFS: {args.normalization_target_dbfs}")
        force_flush()

    if args.DEBUG:
        debug_print("DEBUG mode enabled.", args.DEBUG)

    if not os.path.isdir(directory_path):
        logger.error(f"Error: Directory not found: {directory_path}")
        return

    if not hf_token:
        logger.error("Error: Hugging Face token (HFTOKEN) not provided or found.")
        return

    if not shutil.which("whisperx"):
        logger.error("Error: whisperx command not found. Please ensure it is installed and")
        logger.error("that you have activated the correct Python environment.")
        return

    # --- File discovery and sorting based on timestamp ---
    logger.info("\n=== Scanning for WAV Files ===")
    initial_wav_files = []
    for ext in ("*.WAV", "*.wav"):
        initial_wav_files.extend(glob.glob(os.path.join(directory_path, ext)))

    if not initial_wav_files:
        logger.warning(f"No .WAV or .wav files found in {directory_path}")
        return
    
    logger.info(f"Found {len(initial_wav_files)} WAV files")
    for wav_file in initial_wav_files:
        logger.info(f"  - {os.path.basename(wav_file)}")

    files_to_process_info = []
    if args.filename_timestamp_format and args.filename_timestamp_regex:
        logger.info("\n=== Sorting Files by Timestamp ===")
        logger.info(f"Using timestamp format: {args.filename_timestamp_format}")
        logger.info(f"Using regex pattern: {args.filename_timestamp_regex}")
        
        sortable_files = []
        unsortable_files = []

        for wav_path in initial_wav_files:
            filename = os.path.basename(wav_path)
            try:
                match = re.search(args.filename_timestamp_regex, filename)
                if match and match.group(1):
                    timestamp_str = match.group(1)
                    dt_obj = datetime.strptime(timestamp_str, args.filename_timestamp_format)
                    sortable_files.append({'datetime': dt_obj, 'path': wav_path, 'original_filename': filename})
                    logger.info(f"  ✓ Sorted: {filename} -> {dt_obj}")
                else:
                    logger.warning(f"  ⚠ No timestamp found in: {filename}")
                    unsortable_files.append({'path': wav_path, 'original_filename': filename})
            except ValueError as ve:
                logger.warning(f"  ⚠ Could not parse timestamp in: {filename} - {ve}")
                unsortable_files.append({'path': wav_path, 'original_filename': filename})
            except Exception as e_parse:
                logger.warning(f"  ⚠ Error processing: {filename} - {e_parse}")
                unsortable_files.append({'path': wav_path, 'original_filename': filename})
        
        sortable_files.sort(key=lambda x: x['datetime'])
        files_to_process_info = sortable_files + unsortable_files
        logger.info(f"\nSorted {len(sortable_files)} files by timestamp")
        logger.info(f"Found {len(unsortable_files)} files without valid timestamps")
    else:
        logger.info("\n=== Processing Files in Default Order ===")
        for wav_path in initial_wav_files:
            files_to_process_info.append({'path': wav_path, 'original_filename': os.path.basename(wav_path)})

    # Always concatenate WAV files
    logger.info("\n=== Concatenating WAV Files ===")
    wav_paths = [file_info['path'] for file_info in files_to_process_info]
    concatenated_output = os.path.join(directory_path, f"{os.path.basename(directory_path)}_concatenated.wav")
    logger.info(f"Creating concatenated file: {os.path.basename(concatenated_output)}")
    concatenated_files = concatenate_wav_files(wav_paths, concatenated_output, args.max_chunk_size, args.DEBUG)
    
    if not concatenated_files:
        logger.error("Error: WAV concatenation failed or produced no output files.")
        return
    
    logger.info(f"Created {len(concatenated_files)} concatenated chunks")
    for chunk in concatenated_files:
        logger.info(f"  - {os.path.basename(chunk)}")
    
    # Update files_to_process_info with concatenated files
    files_to_process_info = []
    for i, concat_file in enumerate(concatenated_files):
        files_to_process_info.append({
            'path': concat_file,
            'original_filename': os.path.basename(concat_file)
        })

    generated_txt_files = []

    for file_info in files_to_process_info:
        wav_file_path = file_info['path']
        original_wav_filename = file_info['original_filename'] 
        original_base_name_no_ext = os.path.splitext(original_wav_filename)[0]
        
        current_audio_path = wav_file_path
        
        logger.info(f"\n=== Processing: {original_wav_filename} ===")
        if args.DEBUG: debug_print(f"DEBUG: Starting processing for {original_wav_filename}, original path: {wav_file_path}", args.DEBUG)

        with tempfile.TemporaryDirectory(prefix=f"{original_base_name_no_ext}_processed_", dir=directory_path) as temp_proc_dir:
            logger.info(f"Created temporary processing directory: {temp_proc_dir}")
            
            if args.enable_noise_reduction:
                if not noisereduce or not sf:
                    logger.warning(f"Skipping noise reduction for {original_wav_filename} due to missing libraries.")
                else:
                    temp_nr_path = os.path.join(temp_proc_dir, f"{original_base_name_no_ext}_nr.wav")
                    logger.info(f"Applying noise reduction: {os.path.basename(current_audio_path)} -> {os.path.basename(temp_nr_path)}")
                    if apply_noise_reduction(current_audio_path, temp_nr_path):
                        current_audio_path = temp_nr_path
                        logger.info("✓ Noise reduction completed successfully")
                    else:
                        logger.warning("⚠ Noise reduction failed, using original audio")
            else:
                logger.info("Noise reduction disabled by user")

            if args.enable_normalization:
                if not AudioSegment:
                    logger.warning(f"Skipping normalization for {original_wav_filename} due to missing pydub or ffmpeg/libav.")
                else:
                    temp_norm_path = os.path.join(temp_proc_dir, f"{original_base_name_no_ext}_norm.wav")
                    logger.info(f"Applying normalization: {os.path.basename(current_audio_path)} -> {os.path.basename(temp_norm_path)}")
                    if apply_normalization(current_audio_path, temp_norm_path, args.normalization_target_dbfs):
                        current_audio_path = temp_norm_path
                        logger.info("✓ Normalization completed successfully")
                    else:
                        logger.warning("⚠ Normalization failed, using previous audio")
            else:
                logger.info("Normalization disabled by user")
            
            logger.info(f"\n=== Running WhisperX Transcription ===")
            logger.info(f"Input file: {os.path.basename(current_audio_path)}")
            command = [
                "whisperx", current_audio_path,
                "--model", "large-v3", "--diarize", "--language", "en",
                "--highlight_words", "True", "--hf_token", hf_token,
                "--output_dir", directory_path, 
                "--vad_onset", str(args.vad_onset), "--vad_offset", str(args.vad_offset),
                "--device", args.device,
                "--compute_type", args.compute_type
            ]
            
            if args.DEBUG: debug_print(f"DEBUG: WhisperX command: {' '.join(command)}", args.DEBUG)

            try:
                process_env = os.environ.copy()
                process_env["HFTOKEN"] = hf_token 
                result = subprocess.run(command, capture_output=True, text=True, env=process_env, check=False)

                if result.returncode == 0:
                    logger.info("✓ WhisperX transcription completed successfully")
                    
                    whisperx_input_basename_for_txt = os.path.splitext(os.path.basename(current_audio_path))[0]
                    actual_txt_generated_by_whisperx = os.path.join(directory_path, f"{whisperx_input_basename_for_txt}.txt")
                    target_final_txt_path = os.path.join(directory_path, f"{original_base_name_no_ext}.txt")
                    
                    if os.path.exists(actual_txt_generated_by_whisperx):
                        if actual_txt_generated_by_whisperx != target_final_txt_path:
                            try:
                                if os.path.exists(target_final_txt_path):
                                    logger.warning(f"Target file {os.path.basename(target_final_txt_path)} already exists. Overwriting.")
                                    os.remove(target_final_txt_path)
                                shutil.move(actual_txt_generated_by_whisperx, target_final_txt_path)
                                logger.info(f"Renamed output: {os.path.basename(actual_txt_generated_by_whisperx)} -> {os.path.basename(target_final_txt_path)}")
                            except Exception as e_rename:
                                logger.error(f"Error renaming output file: {e_rename}")
                                target_final_txt_path = actual_txt_generated_by_whisperx
                        
                        generated_txt_files.append(target_final_txt_path)
                        logger.info(f"✓ Transcription saved: {os.path.basename(target_final_txt_path)}")
                    else:
                        logger.error(f"Expected output file not found: {os.path.basename(actual_txt_generated_by_whisperx)}")
                else:
                    logger.error(f"WhisperX transcription failed with return code {result.returncode}")
                    if result.stdout: logger.error(f"WhisperX Stdout:\n{result.stdout}")
                    if result.stderr: logger.error(f"WhisperX Stderr:\n{result.stderr}")
            except Exception as e:
                logger.error(f"Unexpected error during WhisperX processing: {e}")

    if not generated_txt_files:
        logger.error("\nNo transcription files were generated. Processing failed.")
        return

    logger.info("\n=== Concatenating Transcription Files ===")
    abs_directory_path = os.path.abspath(directory_path)
    directory_basename = os.path.basename(abs_directory_path)
    concatenated_file_name = os.path.join(abs_directory_path, f"{directory_basename}_transcription_summary.txt")
    
    logger.info(f"Creating summary file: {os.path.basename(concatenated_file_name)}")
    logger.info(f"Combining {len(generated_txt_files)} transcription files")
    
    with open(concatenated_file_name, "w", encoding="utf-8") as outfile:
        for txt_file_path in generated_txt_files:
            try:
                with open(txt_file_path, "r", encoding="utf-8") as infile:
                    base_txt_filename = os.path.basename(txt_file_path)
                    outfile.write(f"--- Content from: {base_txt_filename} ---\n\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n" + "="*80 + "\n\n")
                logger.info(f"✓ Added content from: {base_txt_filename}")
            except Exception as e:
                logger.error(f"Error processing {txt_file_path}: {e}")
    
    logger.info("\n=== Processing Complete ===")
    logger.info(f"All transcriptions combined into: {os.path.basename(concatenated_file_name)}")

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
                logger.info(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        if return_code != 0:
            logger.error(f"Error: ffmpeg command failed with return code {return_code}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error running ffmpeg command: {e}")
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
        logger.warning(f"Warning: Could not get WAV settings from {wav_file}, using defaults: {e}")
        return ('24000', '2', 'pcm_s16le', '192000')  # Default values: 24kHz, stereo, 192kbit

def concatenate_wav_files(wav_files, output_path, max_size_gb=3.5, debug_mode=False):
    """
    Concatenates WAV files into chunks that don't exceed max_size_gb.
    Uses original WAV file settings for output, or defaults to 24kHz stereo 192kbit.
    Returns a list of paths to the concatenated files.
    """
    if not AudioSegment:
        logger.error("Error: pydub library not available. Cannot concatenate WAV files.")
        return []

    if not wav_files:
        logger.error("Error: No WAV files provided for concatenation.")
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
            logger.error(f"Error processing {wav_file}: {e}")
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    whisperx_group.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Device to use for inference (e.g., cuda, cpu, mps). Defaults to {DEFAULT_DEVICE} on {platform.system()}.")
    whisperx_group.add_argument("--compute_type", type=str, default=DEFAULT_COMPUTE_TYPE, 
                        help=f"Compute type for WhisperX (e.g., auto, float16, float32, int8). Defaults to {DEFAULT_COMPUTE_TYPE} on {platform.system()}.")

    # WAV concatenation arguments
    concat_group = parser.add_argument_group('WAV Concatenation Options')
    concat_group.add_argument("--max-chunk-size", type=float, default=3.5, help="Maximum size of concatenated WAV chunks in GB (default: 3.5GB).")

    args = parser.parse_args()

    # Update logger with debug mode from args
    logger = setup_logging('transcribe', args.DEBUG)

    if args.DEBUG:
        logger.debug("--- Parsed Arguments ---")
        for arg, value in vars(args).items():
            logger.debug(f"{arg}: {value}")
        logger.debug("------------------------")

    if not args.hf_token:
        logger.error("Error: Hugging Face token not found. Set HFTOKEN env var or use --hf_token.")
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

