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
        }
        .toggle-button {
            position: fixed;
            left: 200px;
            top: 70px;
            margin: 0;
            border-radius: 0 4px 4px 0;
            border: none;
            background: var(--background-fill-primary);
            color: var(--body-text-color);
            cursor: pointer;
            transition: all 0.2s;
            z-index: 1000;
            width: 30px;
            height: 30px;
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
            bottom: 100%;
            left: 0;
            background: var(--background-fill-primary);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
        }
        .toggle-button:hover::before {
            opacity: 1;
        }
        .sidebar-hidden .toggle-button {
            left: 0;
        }
        .main-content {
            margin-top: 60px;
        }
    """) as interface:
        # Top bar with title and toggle
        with gr.Row(elem_classes="top-bar"):
            with gr.Column(scale=20):
                gr.Markdown("# MyJournal")
            with gr.Column(scale=1, min_width=50):
                toggle_button = gr.Button(
                    "☰",
                    size="sm",
                    elem_classes="toggle-button"
                )
        
        # Add state for sidebar visibility
        sidebar_visible = gr.State(True)
        
        # Main content area
        with gr.Row(elem_classes="main-content"):
            # Left column in a collapsible container
            with gr.Column(scale=1, min_width=200) as left_column:
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
                
                # Side-by-side toggle
                side_by_side = gr.Checkbox(
                    label="Side-by-Side View",
                    value=False,
                    interactive=True
                )
                
                # Second file browser (for side-by-side)
                file_dropdown2 = gr.Dropdown(
                    choices=journal_viewer._get_markdown_files(),
                    label="Select Second File",
                    interactive=True,
                    visible=False
                )
                
                # Save controls
                save_button = gr.Button("Save Changes")
                save_status = gr.Textbox(label="Status", interactive=False)
            
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
        def update_file_list(journal_type):
            files = journal_viewer._get_markdown_files(journal_type if journal_type != "All" else None)
            return gr.update(choices=files), gr.update(choices=files)
        
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
        
        journal_type.change(
            fn=update_file_list,
            inputs=[journal_type],
            outputs=[file_dropdown, file_dropdown2]
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
            outputs=[save_status]
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