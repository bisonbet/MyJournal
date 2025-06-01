#!/usr/bin/env python3
import os
import gradio as gr
import markdown
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from weasyprint import HTML, CSS
import tempfile
import base64
import subprocess
import sys
import logging
import logging.handlers
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a rotating file handler
    log_file = log_dir / "journal_gui.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Set up logging at module level
logger = setup_logging()

class JournalViewer:
    def __init__(self, root_dir="journals"):
        self.root_dir = Path(root_dir)
        self.current_file = None
        self.journal_types = ["daily", "weekly", "monthly"]
        self._ensure_directory_structure()
        logger.info(f"JournalViewer initialized with root directory: {self.root_dir}")
        
    def _get_yesterdays_path(self):
        """Get the path for yesterday's journal entry."""
        yesterday = datetime.now() - timedelta(days=1)
        year = str(yesterday.year)
        month = yesterday.strftime("%B")
        day = yesterday.strftime("%m%d%Y")
        return f"daily/{year}/{month}/{day}"
        
    def _get_path_for_date(self, selected_date):
        """Get the appropriate path(s) for a selected date."""
        # Convert string date to datetime if needed
        if isinstance(selected_date, str):
            selected_date = datetime.strptime(selected_date, "%Y-%m-%d")
            
        year = str(selected_date.year)
        month = selected_date.strftime("%B")
        day = selected_date.strftime("%m%d%Y")
        
        paths = []
        
        # Always add daily path
        daily_path = f"daily/{year}/{month}/{day}"
        paths.append(("Daily", daily_path))
        
        # If it's Sunday, add weekly path
        if selected_date.weekday() == 6:  # Sunday is 6
            week_start = selected_date
            week_path = f"weekly/{year}/{month}/WeekOf{week_start.strftime('%m%d%Y')}"
            paths.append(("Weekly", week_path))
        
        # If it's first day of month, add monthly path
        if selected_date.day == 1:
            monthly_path = f"monthly/{year}/{month}"
            paths.append(("Monthly", monthly_path))
        
        return paths
        
    def _ensure_directory_structure(self):
        """Ensure the journal directory structure exists."""
        for journal_type in self.journal_types:
            (self.root_dir / journal_type).mkdir(parents=True, exist_ok=True)
            
        # Ensure yesterday's directory exists
        yesterday_path = self.root_dir / self._get_yesterdays_path()
        yesterday_path.mkdir(parents=True, exist_ok=True)
    
    def create_new_journal_entry(self, current_path, title):
        """Create a new journal entry in the current directory."""
        if not current_path:
            return "Please select a directory first"
            
        try:
            # Create the full path
            entry_dir = self.root_dir / current_path
            entry_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on date and title
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}-{title}.md"
            file_path = entry_dir / filename
            
            if file_path.exists():
                return f"File {filename} already exists"
                
            # Create the file with initial content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"Date: {date_str}\n\n")
                
            return f"Created new journal entry: {filename}"
        except Exception as e:
            return f"Error creating journal entry: {str(e)}"
    
    def _get_immediate_subdirectories(self, current_path=None):
        """Get immediate subdirectories of the current path."""
        if current_path is None:
            # At root level, return journal types
            return self.journal_types
            
        try:
            search_dir = self.root_dir / current_path
            subdirs = []
            for item in search_dir.iterdir():
                if item.is_dir():
                    # Just return the directory name, not the full path
                    subdirs.append(item.name)
            return sorted(subdirs)
        except Exception as e:
            return []
    
    def _get_markdown_files(self, path=None):
        """Get all markdown files in the specified directory path."""
        md_files = []
        try:
            if not path:
                # At root level, search all journal types
                for journal_type in self.journal_types:
                    search_dir = self.root_dir / journal_type
                    for file_path in search_dir.glob("*.md"):  # Only current directory
                        # Skip files with "DEBUG" in the name
                        if "DEBUG" in file_path.name.upper():
                            continue
                        rel_path = file_path.relative_to(self.root_dir)
                        md_files.append(str(rel_path))
            else:
                # Search in specific directory
                search_dir = self.root_dir / path
                for file_path in search_dir.glob("*.md"):  # Only current directory
                    # Skip files with "DEBUG" in the name
                    if "DEBUG" in file_path.name.upper():
                        continue
                    rel_path = file_path.relative_to(self.root_dir)
                    md_files.append(str(rel_path))
        except Exception as e:
            pass
            
        return sorted(md_files)
    
    def create_new_journal(self, journal_type, title):
        """Create a new journal entry."""
        if not journal_type or not title:
            return "Please select a journal type and provide a title"
            
        # Create filename based on date and title
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}-{title}.md"
        file_path = self.root_dir / journal_type / filename
        
        if file_path.exists():
            return f"File {filename} already exists"
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"Date: {date_str}\n\n")
            return f"Created new journal: {filename}"
        except Exception as e:
            return f"Error creating journal: {str(e)}"
    
    def load_file(self, filename):
        """Load and return the contents of a markdown file."""
        if not filename:
            return "", ""
        try:
            file_path = self.root_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.current_file = file_path
            return content, self._render_markdown(content)
        except Exception as e:
            return f"Error loading file: {str(e)}", ""
    
    def save_file(self, filename, content):
        """Save content to a markdown file."""
        if not filename:
            return "No file selected"
        try:
            file_path = self.root_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Saved to {filename}"
        except Exception as e:
            return f"Error saving file: {str(e)}"
    
    def _render_markdown(self, content):
        """Convert markdown content to HTML."""
        return markdown.markdown(content, extensions=['extra', 'codehilite'])
    
    def export_to_pdf(self, file_paths):
        """Export selected markdown files to PDF."""
        if not file_paths:
            return "No files selected for export"
            
        try:
            # Create a temporary directory for the PDF
            temp_dir = tempfile.mkdtemp()
            pdf_path = Path(temp_dir) / "journal_export.pdf"
            
            # Combine all markdown content
            combined_html = []
            for file_path in file_paths:
                if not file_path:
                    continue
                    
                full_path = self.root_dir / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(content, extensions=['extra', 'codehilite'])
                combined_html.append(html_content)
            
            # Combine all HTML content with page breaks
            full_html = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 2cm; }}
                        h1 {{ color: #2c3e50; }}
                        pre {{ background-color: #f8f9fa; padding: 1em; border-radius: 4px; }}
                        hr {{ page-break-after: always; }}
                    </style>
                </head>
                <body>
                    {"<hr>".join(combined_html)}
                </body>
            </html>
            """
            
            # Generate PDF
            HTML(string=full_html).write_pdf(pdf_path)
            
            # Return both the path and the temp directory
            return str(pdf_path), temp_dir
                
        except Exception as e:
            return f"Error exporting to PDF: {str(e)}", None

    def generate_word_cloud(self, current_path):
        """Generate a word cloud from the transcription summary file in the current directory."""
        try:
            # Find the transcription summary file
            summary_file = None
            for file in (self.root_dir / current_path).glob("*_transcription_summary.txt"):
                summary_file = file
                break

            if not summary_file:
                return "No transcription summary file found in the current directory", None

            # Read the summary file
            with open(summary_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean the text
            text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
            text = ' '.join(text.split())  # Normalize whitespace

            # Create a temporary directory for the word cloud
            temp_dir = tempfile.mkdtemp()
            wordcloud_path = Path(temp_dir) / "wordcloud.png"

            # Generate the word cloud
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                max_words=200,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text)

            # Save the word cloud
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(wordcloud_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Convert to PDF using absolute path for the image
            pdf_path = Path(temp_dir) / "wordcloud.pdf"
            HTML(string=f'<img src="file://{wordcloud_path.absolute()}" style="width: 100%;">').write_pdf(pdf_path)

            return str(pdf_path), temp_dir

        except Exception as e:
            error_msg = f"Error generating word cloud: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, None

    def get_script_description(self, script_name, current_path):
        """Generate a description for the selected script based on the current path."""
        if not current_path:
            return "Please select a directory first"

        try:
            # Check if we're in the correct directory type for the selected script
            path_parts = current_path.split('/')
            if len(path_parts) < 2:
                return "Invalid path format"

            directory_type = path_parts[0]  # daily, weekly, or monthly

            if script_name == "Generate Daily Summary":
                if directory_type != "daily":
                    return "Please select a Daily directory"
                # Extract date from path (format: daily/YYYY/Month/MMDDYYYY)
                if len(path_parts) >= 4:
                    date_str = path_parts[-1]  # Get the MMDDYYYY part
                    try:
                        date = datetime.strptime(date_str, "%m%d%Y")
                        return f"Transcribe, Diarize and Summarize sound files for {date.strftime('%B %d, %Y')}"
                    except ValueError:
                        return "Invalid date format in path"
                return "Invalid path format for daily summary"

            elif script_name == "Generate Weekly Summary":
                if directory_type != "weekly":
                    return "Please select a Weekly directory"
                # Extract week from path (format: weekly/YYYY/Month/WeekOfMMDDYYYY)
                if len(path_parts) >= 4:
                    week_str = path_parts[-1]  # Get the WeekOfMMDDYYYY part
                    if week_str.startswith("WeekOf"):
                        try:
                            date_str = week_str[6:]  # Remove "WeekOf" prefix
                            date = datetime.strptime(date_str, "%m%d%Y")
                            return f"Create Summary of Week of {date.strftime('%B %d, %Y')}"
                        except ValueError:
                            return "Invalid date format in week path"
                return "Invalid path format for weekly summary"

            elif script_name == "Generate Monthly Summary":
                if directory_type != "monthly":
                    return "Please select a Monthly directory"
                # Extract month from path (format: monthly/YYYY/Month)
                if len(path_parts) >= 3:
                    month = path_parts[-1]  # Get the Month name
                    year = path_parts[-2]   # Get the Year
                    return f"Create Summary of {month} {year}"
                return "Invalid path format for monthly summary"

            elif script_name == "Export to PDF":
                return "Export selected files to PDF"

            elif script_name == "Word Cloud Analysis":
                if directory_type != "daily":
                    return "Please select a Daily directory"
                # Extract date from path (format: daily/YYYY/Month/MMDDYYYY)
                if len(path_parts) >= 4:
                    date_str = path_parts[-1]  # Get the MMDDYYYY part
                    try:
                        date = datetime.strptime(date_str, "%m%d%Y")
                        return f"Generate word cloud visualization from transcription for {date.strftime('%B %d, %Y')}"
                    except ValueError:
                        return "Invalid date format in path"
                return "Invalid path format for word cloud analysis"

            return "No description available for this script"

        except Exception as e:
            logger.error(f"Error generating script description: {str(e)}")
            return "Error generating description"

    def _find_most_recent_date_with_files(self):
        """Find the most recent date that has files in its directory."""
        today = datetime.now()
        for i in range(365):  # Look back up to a year
            check_date = today - timedelta(days=i)
            paths = self._get_path_for_date(check_date)
            for _, path in paths:
                full_path = self.root_dir / path
                if full_path.exists():
                    if any(full_path.glob("*.md")) or any(full_path.glob("*.wav")) or \
                       any(full_path.glob("*.mp3")) or any(full_path.glob("*.txt")):
                        return check_date
        return None

def run_script(script_name, current_path, file1, file2=None, journal_viewer=None):
    if not script_name:
        return "No script selected", None
        
    try:
        if script_name == "Export to PDF":
            # Filter out None values from file paths
            files_to_export = [f for f in [file1, file2] if f]
            logger.info(f"Exporting to PDF: {files_to_export}")
            result = journal_viewer.export_to_pdf(files_to_export)
            
            if isinstance(result, tuple) and len(result) == 2:
                pdf_path, temp_dir = result
                if pdf_path.startswith("Error"):
                    logger.error(f"PDF export error: {pdf_path}")
                    return pdf_path, None
                else:
                    # Store the temp directory in a global variable to prevent garbage collection
                    run_script.temp_dir = temp_dir
                    logger.info(f"PDF exported successfully to: {pdf_path}")
                    return "PDF exported successfully", pdf_path
            else:
                logger.error(f"PDF export error: {result}")
                return result, None
                
        elif script_name == "Generate Daily Summary":
            if not current_path:
                logger.warning("No directory selected for summary generation")
                return "No directory selected", None
                
            try:
                # Convert relative path to full path
                full_path = str(journal_viewer.root_dir / current_path)
                logger.info(f"Running diarize-audio.py with path: {full_path}")
                
                # Run diarize-audio.py with the full path
                result = subprocess.run(
                    [sys.executable, "diarize-audio.py", full_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Log all output
                if result.stdout:
                    logger.info(f"Script stdout:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Script stderr:\n{result.stderr}")
                    
                return f"Daily summary generated successfully:\n{result.stdout}", None
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Error generating daily summary:\nCommand: {' '.join(e.cmd)}\nExit code: {e.returncode}\nOutput: {e.stdout}\nError: {e.stderr}"
                logger.error(error_msg)
                return error_msg, None
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}\nType: {type(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, None

        elif script_name == "Generate Weekly Summary":
            if not current_path:
                logger.warning("No directory selected for weekly summary generation")
                return "No directory selected", None
                
            try:
                # Extract date from the path (format: weekly/YYYY/Month/WeekOfMMDDYYYY)
                path_parts = current_path.split('/')
                if len(path_parts) >= 4:
                    week_str = path_parts[-1]  # Get the WeekOfMMDDYYYY part
                    if week_str.startswith("WeekOf"):
                        date_str = week_str[6:]  # Remove "WeekOf" prefix
                        try:
                            # Convert MMDDYYYY to YYYY-MM-DD format
                            date_obj = datetime.strptime(date_str, "%m%d%Y")
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                            
                            # Run summarize_week.py directly with the date
                            logger.info(f"Running summarize_week.py for date: {formatted_date}")
                            result = subprocess.run(
                                [sys.executable, "summarize_week.py", formatted_date, "--base_dir", str(journal_viewer.root_dir)],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            
                            # Log all output
                            if result.stdout:
                                logger.info(f"Script stdout:\n{result.stdout}")
                            if result.stderr:
                                logger.warning(f"Script stderr:\n{result.stderr}")
                                
                            return f"Weekly summary generated successfully:\n{result.stdout}", None
                            
                        except ValueError as e:
                            error_msg = f"Invalid date format in path: {e}"
                            logger.error(error_msg)
                            return error_msg, None
                    else:
                        error_msg = "Invalid week folder format"
                        logger.error(error_msg)
                        return error_msg, None
                else:
                    error_msg = "Invalid path format for weekly summary"
                    logger.error(error_msg)
                    return error_msg, None
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Error generating weekly summary:\nCommand: {' '.join(e.cmd)}\nExit code: {e.returncode}\nOutput: {e.stdout}\nError: {e.stderr}"
                logger.error(error_msg)
                return error_msg, None
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}\nType: {type(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, None

        elif script_name == "Generate Monthly Summary":
            if not current_path:
                logger.warning("No directory selected for monthly summary generation")
                return "No directory selected", None
            logger.info("Monthly summary generation requested (not implemented)")
            return "Monthly summary generation will be implemented", None
                
        elif script_name == "Word Cloud Analysis":
            if not current_path:
                logger.warning("No directory selected for word cloud analysis")
                return "No directory selected", None
                
            logger.info(f"Generating word cloud for path: {current_path}")
            result = journal_viewer.generate_word_cloud(current_path)
            
            if isinstance(result, tuple) and len(result) == 2:
                pdf_path, temp_dir = result
                if pdf_path.startswith("Error"):
                    logger.error(f"Word cloud generation error: {pdf_path}")
                    return pdf_path, None
                else:
                    # Store the temp directory in a global variable to prevent garbage collection
                    run_script.temp_dir = temp_dir
                    logger.info(f"Word cloud generated successfully: {pdf_path}")
                    return "Word cloud generated successfully", pdf_path
            else:
                logger.error(f"Word cloud generation error: {result}")
                return result, None
                
        return f"Running {script_name}...", None
        
    except Exception as e:
        error_msg = f"Error running script: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, None

# Initialize the temp_dir attribute for the run_script function
run_script.temp_dir = None

def create_interface(journal_viewer):
    with gr.Blocks(title="MyJournal", theme=gr.themes.Soft(), css="""
        .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--background-fill-primary);
            padding: 10px;
            z-index: 1000;
            border-bottom: 1px solid var(--border-color-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .toggle-button {
            margin: 0;
            border-radius: 4px;
            border: 1px solid var(--border-color-primary);
            background: var(--background-fill-primary);
            color: var(--body-text-color);
            cursor: pointer;
            transition: all 0.2s;
            z-index: 1000;
            width: 40px;
            height: 40px;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .toggle-button:hover {
            background: var(--background-fill-secondary);
        }
        .toggle-button::before {
            content: "Toggle Sidebar";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--background-fill-primary);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
            margin-top: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toggle-button:hover::before {
            opacity: 1;
        }
        .sidebar-hidden .toggle-button {
            margin-left: 0;
        }
        .main-content {
            margin-top: 60px;
        }
        .calendar-container {
            padding: 10px;
            background: var(--background-fill-secondary);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .directory-options {
            margin-top: 10px;
            padding: 10px;
            background: var(--background-fill-secondary);
            border-radius: 8px;
        }
    """) as interface:
        # Top bar with title and toggle
        with gr.Row(elem_classes="top-bar"):
            with gr.Column(scale=1, min_width=50):
                toggle_button = gr.Button(
                    "☰",
                    size="sm",
                    elem_classes="toggle-button"
                )
            with gr.Column(scale=20):
                gr.Markdown("# MyJournal")
        
        # Add state for sidebar visibility
        sidebar_visible = gr.State(True)
        
        # Main content area
        with gr.Row(elem_classes="main-content"):
            # Left column in a collapsible container
            with gr.Column(scale=1, min_width=200) as left_column:
                # Calendar picker
                with gr.Group(elem_classes="calendar-container"):
                    with gr.Row():
                        # Find most recent date with files
                        most_recent_date = journal_viewer._find_most_recent_date_with_files()
                        logger.info(f"Most recent date with files: {most_recent_date}")
                        if most_recent_date is None:
                            gr.Markdown("No journal entries found in the last year")
                            return interface

                        # Define months list for reference
                        months = ["January", "February", "March", "April", "May", "June", 
                                "July", "August", "September", "October", "November", "December"]

                        # Find available years and months
                        available_years = set()
                        available_months = set()
                        current_year = datetime.now().year
                        
                        logger.info("Checking for available years and months...")
                        # Check last 2 years for files
                        for year in range(current_year - 2, current_year + 1):
                            year_has_files = False
                            for month in range(1, 13):
                                month_has_files = False
                                # Check each day in the month
                                for day in range(1, 32):
                                    try:
                                        date = datetime(year, month, day)
                                        paths = journal_viewer._get_path_for_date(date)
                                        for _, path in paths:
                                            full_path = journal_viewer.root_dir / path
                                            if full_path.exists():
                                                # Check for any relevant files with case-insensitive extensions
                                                for ext in ['*.[mM][dD]', '*.[wW][aA][vV]', '*.[mM][pP]3', '*.[tT][xX][tT]']:
                                                    files = list(full_path.glob(ext))
                                                    # Filter out DEBUG files
                                                    files = [f for f in files if "DEBUG" not in f.name.upper()]
                                                    if files:
                                                        year_has_files = True
                                                        month_has_files = True
                                                        break
                                                if month_has_files:
                                                    break
                                        if month_has_files:
                                            break
                                    except ValueError:
                                        # Skip invalid dates (e.g., Feb 30)
                                        continue
                                
                                if month_has_files:
                                    available_months.add(months[month - 1])
                            
                            if year_has_files:
                                available_years.add(str(year))
                        
                        logger.info(f"Available years: {available_years}")
                        logger.info(f"Available months: {available_months}")
                                        
                        # Generate list of years that have files
                        years = sorted(list(available_years))
                        year_dropdown = gr.Dropdown(
                            label="Year",
                            choices=years,
                            value=str(most_recent_date.year) if str(most_recent_date.year) in years else years[0] if years else None,
                            interactive=True
                        )
                        
                        # Months that have files
                        months_list = sorted(list(available_months), key=lambda x: months.index(x))
                        month_dropdown = gr.Dropdown(
                            label="Month",
                            choices=months_list,
                            value=months[most_recent_date.month - 1] if months[most_recent_date.month - 1] in months_list else months_list[0] if months_list else None,
                            interactive=True
                        )
                        
                        # Days (will be updated based on year/month)
                        days = [str(day).zfill(2) for day in range(1, 32)]
                        day_dropdown = gr.Dropdown(
                            label="Day",
                            choices=[],
                            value=None,
                            interactive=True
                        )
                
                # Directory options (will be populated based on date selection)
                with gr.Group(elem_classes="directory-options"):
                    directory_radio = gr.Radio(
                        label="Select Directory Type",
                        choices=[],
                        interactive=True
                    )
                
                # File browser
                file_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select File",
                    interactive=True
                )
                
                # Side-by-side toggle
                side_by_side = gr.Checkbox(
                    label="Side-by-Side View",
                    value=False,
                    interactive=True
                )
                
                # Second file browser (for side-by-side)
                file_dropdown2 = gr.Dropdown(
                    choices=[],
                    label="Select Second File",
                    interactive=True,
                    visible=False
                )
                
                # Save controls
                with gr.Group():
                    save_button = gr.Button("Save Changes")
                    script_dropdown = gr.Dropdown(
                        choices=["Export to PDF", "Generate Daily Summary", "Generate Weekly Summary", "Generate Monthly Summary", "Word Cloud Analysis"],
                        label="Select Script",
                        interactive=True
                    )
                    script_description = gr.Textbox(
                        label="Script Description",
                        interactive=False,
                        value="Select a script to see its description",
                        show_label=True
                    )
                    run_script_button = gr.Button("Run Script")
                    script_status = gr.Textbox(label="Script Status", interactive=False)
                    pdf_download = gr.File(label="Download PDF", visible=False)
            
            # Main content column
            with gr.Column(scale=2) as main_column:
                with gr.Tabs():
                    with gr.TabItem("Edit"):
                        with gr.Group():
                            with gr.Row():
                                with gr.Column(scale=1):
                                    editor = gr.Textbox(
                                        label="Editor",
                                        lines=20,
                                        show_label=False,
                                        interactive=True
                                    )
                                    with gr.Row():
                                        find_text = gr.Textbox(
                                            label="Find",
                                            placeholder="Text to find",
                                            scale=2
                                        )
                                        replace_text = gr.Textbox(
                                            label="Replace",
                                            placeholder="Text to replace with",
                                            scale=2
                                        )
                                        find_replace_button = gr.Button("Find & Replace", scale=1)
                                
                                editor2 = gr.Textbox(
                                    label="Second Editor",
                                    lines=20,
                                    show_label=False,
                                    interactive=True,
                                    visible=False
                                )
                    
                    with gr.TabItem("Preview"):
                        with gr.Row():
                            preview = gr.HTML(label="Preview")
                            preview2 = gr.HTML(label="Second Preview", visible=False)

        # Event handlers
        def update_days(year, month):
            """Update the days dropdown based on selected year and month."""
            try:
                if not year or not month:
                    logger.info("No year or month selected")
                    return gr.update(choices=[], value=None)
                    
                # Convert month name to number (1-12)
                month_num = months.index(month) + 1
                # Get the last day of the month
                last_day = (datetime(int(year), month_num + 1, 1) - timedelta(days=1)).day
                available_days = []
                
                logger.info(f"Checking days for {year}-{month_num}")
                
                # Check each day to see if it has files
                for day in range(1, last_day + 1):
                    try:
                        date = datetime(int(year), month_num, day)
                        paths = journal_viewer._get_path_for_date(date)
                        has_files = False
                        
                        # Check each path type (daily, weekly, monthly) for files
                        for label, path in paths:
                            full_path = journal_viewer.root_dir / path
                            if full_path.exists():
                                # Check for any relevant files with case-insensitive extensions
                                for ext in ['*.[mM][dD]', '*.[wW][aA][vV]', '*.[mM][pP]3', '*.[tT][xX][tT]']:
                                    files = list(full_path.glob(ext))
                                    # Filter out DEBUG files
                                    files = [f for f in files if "DEBUG" not in f.name.upper()]
                                    if files:
                                        logger.info(f"Found files in {path}: {files}")
                                        has_files = True
                                        break
                                if has_files:
                                    break
                                    
                        if has_files:
                            available_days.append(str(day).zfill(2))
                            logger.info(f"Added day {day} to available days")
                    except ValueError as e:
                        logger.error(f"Invalid date: {e}")
                        continue
                
                logger.info(f"Available days: {available_days}")
                
                # If no days are available, return empty choices
                if not available_days:
                    logger.info("No days with files found")
                    return gr.update(choices=[], value=None)
                    
                # Try to keep the current day if it's still valid
                current_day = day_dropdown.value
                if current_day in available_days:
                    logger.info(f"Keeping current day: {current_day}")
                    return gr.update(choices=available_days, value=current_day)
                logger.info(f"Using first available day: {available_days[0]}")
                return gr.update(choices=available_days, value=available_days[0])
                
            except Exception as e:
                logger.error(f"Error updating days: {str(e)}")
                return gr.update(choices=[], value=None)

        def get_selected_date(year, month, day):
            """Convert selected year, month, day to a datetime object."""
            try:
                if not all([year, month, day]):
                    return datetime.now() - timedelta(days=1)
                    
                month_num = months.index(month) + 1
                return datetime(int(year), month_num, int(day))
            except Exception as e:
                logger.error(f"Error creating date: {str(e)}")
                return datetime.now() - timedelta(days=1)

        def update_file_list(selected_path):
            """Update the file list based on the selected directory path."""
            if not selected_path:
                return gr.update(choices=[])
            files = journal_viewer._get_markdown_files(selected_path)
            return gr.update(choices=files, value=files[0] if files else None)

        def get_initial_directory_options():
            """Get initial directory options for the most recent date."""
            paths = journal_viewer._get_path_for_date(most_recent_date)
            choices = [(label, path) for label, path in paths]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        # Update days when year or month changes
        year_dropdown.change(
            fn=update_days,
            inputs=[year_dropdown, month_dropdown],
            outputs=[day_dropdown]
        )
        month_dropdown.change(
            fn=update_days,
            inputs=[year_dropdown, month_dropdown],
            outputs=[day_dropdown]
        )

        # Initial day dropdown update
        def initialize_day_dropdown():
            return update_days(str(most_recent_date.year), months[most_recent_date.month - 1])

        # Add a load event to initialize the day dropdown
        interface.load(
            fn=initialize_day_dropdown,
            inputs=[],
            outputs=[day_dropdown]
        )

        # Update directory options when any date component changes
        def update_directory_options_from_components(year, month, day):
            selected_date = get_selected_date(year, month, day)
            paths = journal_viewer._get_path_for_date(selected_date)
            choices = [(label, path) for label, path in paths]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        # Connect date components to directory options
        year_dropdown.change(
            fn=update_directory_options_from_components,
            inputs=[year_dropdown, month_dropdown, day_dropdown],
            outputs=[directory_radio]
        )
        month_dropdown.change(
            fn=update_directory_options_from_components,
            inputs=[year_dropdown, month_dropdown, day_dropdown],
            outputs=[directory_radio]
        )
        day_dropdown.change(
            fn=update_directory_options_from_components,
            inputs=[year_dropdown, month_dropdown, day_dropdown],
            outputs=[directory_radio]
        )

        # Connect directory selection to file list
        directory_radio.change(
            fn=update_file_list,
            inputs=[directory_radio],
            outputs=[file_dropdown]
        )

        # Connect file selection to editor
        file_dropdown.change(
            fn=journal_viewer.load_file,
            inputs=[file_dropdown],
            outputs=[editor, preview]
        )

        # Connect the script description update
        def update_script_description(script_name, selected_path):
            return journal_viewer.get_script_description(script_name, selected_path)

        script_dropdown.change(
            fn=update_script_description,
            inputs=[script_dropdown, directory_radio],
            outputs=[script_description]
        )

        directory_radio.change(
            fn=update_script_description,
            inputs=[script_dropdown, directory_radio],
            outputs=[script_description]
        )

        # Connect the script runner
        run_script_button.click(
            fn=lambda *args: run_script(*args, journal_viewer=journal_viewer),
            inputs=[script_dropdown, directory_radio, file_dropdown, file_dropdown2],
            outputs=[script_status, pdf_download]
        )

        # Show/hide PDF download based on script status
        def update_pdf_download(status, pdf_path):
            if pdf_path and os.path.exists(pdf_path):
                return gr.update(visible=True, value=pdf_path)
            return gr.update(visible=False)
        
        script_status.change(
            fn=update_pdf_download,
            inputs=[script_status, pdf_download],
            outputs=[pdf_download]
        )
        
        # Clean up temporary directory after download
        def cleanup_temp_dir(pdf_path):
            if hasattr(run_script, 'temp_dir') and run_script.temp_dir:
                try:
                    import shutil
                    shutil.rmtree(run_script.temp_dir)
                    run_script.temp_dir = None
                except Exception:
                    pass
            return pdf_path
        
        pdf_download.change(
            fn=cleanup_temp_dir,
            inputs=[pdf_download],
            outputs=[pdf_download]
        )
        
        return interface

def main():
    parser = argparse.ArgumentParser(description="Journal Viewer Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=45459, help="Port to listen on (default: 45459)")
    parser.add_argument("--root-dir", default="journals", help="Root directory for journals (default: journals)")
    args = parser.parse_args()
    
    logger.info(f"Starting Journal Viewer on {args.host}:{args.port}")
    logger.info(f"Using root directory: {args.root_dir}")
    
    journal_viewer = JournalViewer(root_dir=args.root_dir)
    interface = create_interface(journal_viewer)
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        pwa=True
    )

if __name__ == "__main__":
    main() 