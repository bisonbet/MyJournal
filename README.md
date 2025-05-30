# MyJournal

A way to take audio files recorded during the way and extract the data to give a summary of your day.  This includes the ability to get notes, find tasks you forgot, or even for posterity!

This system provides a comprehensive solution for transcribing audio files, generating summaries, and converting WAV files to MP3 format. It consists of five main components:

1. `create_dirs.py` - Creates the hierarchical directory structure for organizing journal entries
2. `diarize-audio.py` - Main orchestration script
3. `transcribeTHIS.py` - Handles audio transcription using WhisperX
4. `summarizeTHIS.py` - Generates summaries of the transcriptions
5. `journal_gui.py` - Graphical user interface for easy interaction

## Setup

### Prerequisites

- Python 3.x
- ffmpeg (for audio processing)
- CUDA-compatible GPU (recommended for faster processing)

### Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

## Usage

### Directory Structure Setup

First, create the necessary directory structure for organizing your journal entries:

```bash
python create_dirs.py
```

This will create a hierarchical structure for:
- Daily entries: `./daily/YYYY/MonthName/MMDDYYYY`
- Weekly entries: `./weekly/YYYY/MonthName/WeekOfMMDDYYYY`
- Monthly entries: `./monthly/YYYY/MonthName`

The structure spans the next 12 months from the current date.

### Graphical User Interface (GUI)

The easiest way to use MyJournal is through the GUI:

```bash
python journal_gui.py
```

The GUI provides a user-friendly interface to:
- Upload and process audio files
- View transcriptions and summaries
- Manage your journal entries
- Configure processing options

### Command Line Interface

#### Main Orchestration Script (`diarize-audio.py`)

The main script orchestrates the entire process:

```bash
python diarize-audio.py <wav_directory> [hf_token] [--production] [additional_summarize_options]
```

#### Required Arguments:
- `wav_directory`: Path to directory containing WAV files
- `hf_token`: Hugging Face token (can also be set via HFTOKEN environment variable)

#### Optional Arguments:
- `--production`: Run in production mode (deletes WAVs after MP3 conversion, no debug mode)
- Additional arguments for summarizeTHIS.py can be passed after the main arguments

### Transcription Script (`transcribeTHIS.py`)

Handles the audio transcription process with WhisperX.

#### Key Features:
- Noise reduction (optional)
- Audio normalization (optional)
- Timestamp-based file sorting
- VAD (Voice Activity Detection) configuration

#### Options:
- `--disable-noise-reduction`: Disable noise reduction
- `--disable-normalization`: Disable audio normalization
- `--normalization-target-dbfs`: Set target dBFS for normalization (default: -20.0)
- `--vad-onset`: Set VAD onset threshold (default: 0.5)
- `--vad-offset`: Set VAD offset threshold (default: 0.363)
- `--compute_type`: Set compute type for WhisperX (default: float16)
- `--DEBUG`: Enable debug mode

### Summarization Script (`summarizeTHIS.py`)

Generates summaries of the transcriptions using multiple models.

#### Key Features:
- Multi-stage summarization process
- Support for multiple models
- Configurable context windows and token limits

#### Options:
- `--ollama_url`: URL of Ollama server (default: http://localhost:11434)
- `--ollama_model`: Default Ollama model
- `--output_file`: Base path for saving summaries
- `--target_chunk_ratio`: Ratio for text content in chunks (default: 0.6)
- `--overlap_tokens`: Token overlap between chunks (default: 200)
- `--DEBUG`: Enable debug mode

## Example Usage

1. Using the GUI:
```bash
python journal_gui.py
```

2. Basic usage with debug mode:
```bash
python diarize-audio.py /path/to/wavs YOUR_HF_TOKEN
```

3. Production mode with custom summarization options:
```bash
python diarize-audio.py /path/to/wavs YOUR_HF_TOKEN --production --output_file custom_output
```

4. With noise reduction disabled:
```bash
python diarize-audio.py /path/to/wavs YOUR_HF_TOKEN --disable-noise-reduction
```

## Output

The system generates:
1. Transcribed text files for each WAV file
2. A concatenated transcript file
3. Summary files for each model used
4. MP3 versions of the original WAV files (in production mode)

## Notes

- The system requires a Hugging Face token for WhisperX operation
- Debug mode is enabled by default unless `--production` is specified
- WAV files are preserved in debug mode and deleted after MP3 conversion in production mode
- The system supports various audio preprocessing options for optimal transcription quality
- The GUI provides an intuitive interface for all features available in the command-line tools
