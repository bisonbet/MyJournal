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

class JournalViewer:
    def __init__(self, root_dir="journals"):
        self.root_dir = Path(root_dir)
        self.current_file = None
        self.journal_types = ["daily", "weekly", "monthly"]
        self._ensure_directory_structure()
        
    def _get_yesterdays_path(self):
        """Get the path for yesterday's journal entry."""
        yesterday = datetime.now() - timedelta(days=1)
        year = str(yesterday.year)
        month = yesterday.strftime("%B")
        day = yesterday.strftime("%m%d%Y")
        return f"daily/{year}/{month}/{day}"
        
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
                        rel_path = file_path.relative_to(self.root_dir)
                        md_files.append(str(rel_path))
            else:
                # Search in specific directory
                search_dir = self.root_dir / path
                for file_path in search_dir.glob("*.md"):  # Only current directory
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
                # Current path display
                current_path = gr.Textbox(
                    label="Current Path",
                    interactive=False,
                    value=journal_viewer._get_yesterdays_path()
                )
                
                # Directory level selector
                directory_dropdown = gr.Dropdown(
                    choices=journal_viewer._get_immediate_subdirectories(journal_viewer._get_yesterdays_path()),
                    value=None,
                    label="Select Directory",
                    interactive=True
                )
                
                # Navigation buttons
                with gr.Row():
                    up_button = gr.Button("⬆️ Up")
                    home_button = gr.Button("🏠 Home")
                
                # File browser
                file_dropdown = gr.Dropdown(
                    choices=journal_viewer._get_markdown_files(journal_viewer._get_yesterdays_path()),
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
                    choices=journal_viewer._get_markdown_files(journal_viewer._get_yesterdays_path()),
                    label="Select Second File",
                    interactive=True,
                    visible=False
                )
                
                # Save controls
                save_button = gr.Button("Save Changes")
                script_dropdown = gr.Dropdown(
                    choices=["Export to PDF", "Generate Summary", "Word Cloud Analysis"],
                    label="Select Script",
                    interactive=True
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
        def update_directory_list(current_path, selected_dir):
            if selected_dir is None:
                subdirs = journal_viewer._get_immediate_subdirectories(current_path)
                files = journal_viewer._get_markdown_files(current_path)
                # If there are files, select the first one
                first_file = files[0] if files else None
                return current_path, gr.update(choices=subdirs), gr.update(choices=files, value=first_file), gr.update(choices=files, value=None)
            
            new_path = f"{current_path}/{selected_dir}" if current_path else selected_dir
            subdirs = journal_viewer._get_immediate_subdirectories(new_path)
            files = journal_viewer._get_markdown_files(new_path)
            # If there are files, select the first one
            first_file = files[0] if files else None
            
            return new_path, gr.update(choices=subdirs), gr.update(choices=files, value=first_file), gr.update(choices=files, value=None)
        
        def go_up(current_path):
            if not current_path:
                subdirs = journal_viewer._get_immediate_subdirectories()
                files = journal_viewer._get_markdown_files()
                # If there are files, select the first one
                first_file = files[0] if files else None
                return current_path, gr.update(choices=subdirs), gr.update(choices=files, value=first_file), gr.update(choices=files, value=None)
            
            parent_path = str(Path(current_path).parent)
            if parent_path == ".":
                parent_path = ""
            
            subdirs = journal_viewer._get_immediate_subdirectories(parent_path)
            files = journal_viewer._get_markdown_files(parent_path)
            # If there are files, select the first one
            first_file = files[0] if files else None
            
            return parent_path, gr.update(choices=subdirs), gr.update(choices=files, value=first_file), gr.update(choices=files, value=None)
        
        def go_home():
            subdirs = journal_viewer._get_immediate_subdirectories()
            files = journal_viewer._get_markdown_files()
            # If there are files, select the first one
            first_file = files[0] if files else None
            return "", gr.update(choices=subdirs), gr.update(choices=files, value=first_file), gr.update(choices=files, value=None)
        
        def find_replace(text, find, replace):
            if not find:
                return text
            return text.replace(find, replace)
        
        def toggle_side_by_side(show_side_by_side):
            return {
                file_dropdown2: gr.update(visible=show_side_by_side),
                editor2: gr.update(visible=show_side_by_side),
                preview2: gr.update(visible=show_side_by_side)
            }
        
        def toggle_sidebar(current_state):
            new_state = not current_state
            return {
                sidebar_visible: new_state,
                left_column: gr.update(visible=new_state),
                main_column: gr.update(scale=1 if new_state else 3),
                toggle_button: gr.update(elem_classes=["toggle-button", "sidebar-hidden"] if not new_state else ["toggle-button"])
            }
        
        def run_script(script_name, current_path, file1, file2=None):
            if not script_name:
                return "No script selected", None
                
            try:
                if script_name == "Export to PDF":
                    # Filter out None values from file paths
                    files_to_export = [f for f in [file1, file2] if f]
                    result = journal_viewer.export_to_pdf(files_to_export)
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        pdf_path, temp_dir = result
                        if pdf_path.startswith("Error"):
                            return pdf_path, None
                        else:
                            # Store the temp directory in a global variable to prevent garbage collection
                            run_script.temp_dir = temp_dir
                            return "PDF exported successfully", pdf_path
                    else:
                        return result, None
                        
                elif script_name == "Generate Summary":
                    if not current_path:
                        return "No directory selected", None
                        
                    try:
                        # Run diarize-audio.py with the current path
                        result = subprocess.run(
                            [sys.executable, "diarize-audio.py", current_path],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        return f"Summary generated successfully:\n{result.stdout}", None
                    except subprocess.CalledProcessError as e:
                        return f"Error generating summary: {e.stderr}", None
                        
                elif script_name == "Word Cloud Analysis":
                    return "Word cloud analysis will be implemented", None
                    
                return f"Running {script_name}...", None
            except Exception as e:
                return f"Error running script: {str(e)}", None
        
        # Initialize the temp_dir attribute for the run_script function
        run_script.temp_dir = None
        
        # Connect the directory navigation
        directory_dropdown.change(
            fn=update_directory_list,
            inputs=[current_path, directory_dropdown],
            outputs=[current_path, directory_dropdown, file_dropdown, file_dropdown2]
        )
        
        up_button.click(
            fn=go_up,
            inputs=[current_path],
            outputs=[current_path, directory_dropdown, file_dropdown, file_dropdown2]
        )
        
        home_button.click(
            fn=go_home,
            inputs=[],
            outputs=[current_path, directory_dropdown, file_dropdown, file_dropdown2]
        )
        
        file_dropdown.change(
            fn=journal_viewer.load_file,
            inputs=[file_dropdown],
            outputs=[editor, preview]
        )
        
        file_dropdown2.change(
            fn=journal_viewer.load_file,
            inputs=[file_dropdown2],
            outputs=[editor2, preview2]
        )
        
        editor.change(
            fn=journal_viewer._render_markdown,
            inputs=[editor],
            outputs=[preview]
        )
        
        editor2.change(
            fn=journal_viewer._render_markdown,
            inputs=[editor2],
            outputs=[preview2]
        )
        
        save_button.click(
            fn=journal_viewer.save_file,
            inputs=[file_dropdown, editor],
            outputs=[script_status]
        )
        
        find_replace_button.click(
            fn=find_replace,
            inputs=[editor, find_text, replace_text],
            outputs=[editor]
        )
        
        side_by_side.change(
            fn=toggle_side_by_side,
            inputs=[side_by_side],
            outputs=[file_dropdown2, editor2, preview2]
        )
        
        toggle_button.click(
            fn=toggle_sidebar,
            inputs=[sidebar_visible],
            outputs=[sidebar_visible, left_column, main_column, toggle_button]
        )
        
        # Connect the script runner
        run_script_button.click(
            fn=run_script,
            inputs=[script_dropdown, current_path, file_dropdown, file_dropdown2],
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