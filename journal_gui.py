#!/usr/bin/env python3
import os
import gradio as gr
import markdown
from pathlib import Path
import argparse
from datetime import datetime

class JournalViewer:
    def __init__(self, root_dir="journals"):
        self.root_dir = Path(root_dir)
        self.current_file = None
        self.journal_types = ["daily", "weekly", "monthly"]
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self):
        """Ensure the journal directory structure exists."""
        for journal_type in self.journal_types:
            (self.root_dir / journal_type).mkdir(parents=True, exist_ok=True)
    
    def _get_markdown_files(self, journal_type=None):
        """Get all markdown files in the specified journal type directory."""
        md_files = []
        if journal_type:
            search_dir = self.root_dir / journal_type
        else:
            search_dir = self.root_dir
            
        for path in search_dir.rglob("*.md"):
            # Get relative path from the root directory
            rel_path = path.relative_to(self.root_dir)
            md_files.append(str(rel_path))
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

def create_interface(journal_viewer):
    with gr.Blocks(title="Journal Viewer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Journal Viewer")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Journal type selector
                journal_type = gr.Dropdown(
                    choices=["All"] + journal_viewer.journal_types,
                    value="All",
                    label="Journal Type",
                    interactive=True
                )
                
                # File browser
                file_dropdown = gr.Dropdown(
                    choices=journal_viewer._get_markdown_files(),
                    label="Select File",
                    interactive=True
                )
                
                # New journal creation
                with gr.Group():
                    gr.Markdown("### Create New Journal")
                    new_journal_type = gr.Dropdown(
                        choices=journal_viewer.journal_types,
                        label="Type",
                        interactive=True
                    )
                    new_journal_title = gr.Textbox(
                        label="Title",
                        placeholder="Enter journal title"
                    )
                    create_button = gr.Button("Create New Journal")
                    create_status = gr.Textbox(label="Status", interactive=False)
                
                # Save controls
                save_button = gr.Button("Save Changes")
                save_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Edit"):
                        editor = gr.Textbox(
                            label="Editor",
                            lines=20,
                            show_label=False,
                            interactive=True
                        )
                    with gr.TabItem("Preview"):
                        preview = gr.HTML(label="Preview")
        
        # Event handlers
        def update_file_list(journal_type):
            if journal_type == "All":
                return gr.Dropdown.update(choices=journal_viewer._get_markdown_files())
            return gr.Dropdown.update(choices=journal_viewer._get_markdown_files(journal_type))
        
        journal_type.change(
            fn=update_file_list,
            inputs=[journal_type],
            outputs=[file_dropdown]
        )
        
        file_dropdown.change(
            fn=journal_viewer.load_file,
            inputs=[file_dropdown],
            outputs=[editor, preview]
        )
        
        editor.change(
            fn=journal_viewer._render_markdown,
            inputs=[editor],
            outputs=[preview]
        )
        
        save_button.click(
            fn=journal_viewer.save_file,
            inputs=[file_dropdown, editor],
            outputs=[save_status]
        )
        
        create_button.click(
            fn=journal_viewer.create_new_journal,
            inputs=[new_journal_type, new_journal_title],
            outputs=[create_status]
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
        share=False
    )

if __name__ == "__main__":
    main() 