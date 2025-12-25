# Legacy-Code-Archaeologist
"The Legacy Code Archaeologist" is a high-value tool that solves the massive pain point of technical debt. By visualizing spaghetti code, you turn an abstract headache into a concrete map.
Here is the Architectural Blueprint to build this tool. We will build this as a CLI (Command Line Interface)tool first for simplicity and power, which generates a static HTML report.

1. The Tech Stack
* Language: Python 3.11+ (The ecosystem king for parsing & AI).
* Core Parser: Tree-sitter (Best-in-class polyglot parser. Much better than standard RegEx or simple AST libraries because it handles syntax errors gracefully).
* AI Orchestration: LangChain (To manage context windows and prompt templates).
* LLM Provider: OpenAI API (gpt-4-turbo for complex logic) or Anthropic (claude-3-5-sonnet is currently top-tier for coding tasks).
* Visualization Engine: Mermaid.js (We will generate text syntax, then embed it into a template HTML file).
* CLI Library: Typer or Click (For a clean, modern user experience).

2. The Folder Structure
This structure follows a Service-Repository pattern to keep the parsing logic separate from the AI logic.
Plaintext

legacy_archaeologist/
│
├── core/
│   ├── __init__.py
│   ├── parser_engine.py      # Wrapper for Tree-sitter (handles .py, .java, .php)
│   ├── graph_builder.py      # Logic to structure nodes/edges before AI cleanup
│   └── file_walker.py        # Efficiently recursively lists files (ignoring .git, node_modules)
│
├── ai/
│   ├── __init__.py
│   ├── llm_client.py         # Setup for OpenAI/Anthropic
│   ├── prompts.py            # Storage for "Analyze this code" system prompts
│   └── summarizer.py         # RAG logic: "Explain this specific module"
│
├── templates/
│   └── report_template.html  # HTML skeleton with Mermaid.js CDN link
│
├── utils/
│   ├── __init__.py
│   ├── config.py             # Loads .env variables (API_KEY)
│   └── logger.py             # Nice colored terminal output
│
├── main.py                   # Entry point (CLI commands)
├── requirements.txt          # Dependencies
└── .env                      # API Keys (GitIgnored)

3. Dependency List (requirements.txt)
Create this file immediately to set up your environment.
Plaintext

# Core
python-dotenv==1.0.0      # For managing API keys
typer[all]==0.9.0         # For the CLI interface
rich==13.7.0              # For beautiful terminal loading bars/tables

# Parsing
tree-sitter==0.20.4       # The heavy lifter for code parsing
tree-sitter-languages==1.10.0 # Pre-compiled binaries for standard languages

# AI & Graphing
langchain==0.1.0
langchain-openai==0.0.5
networkx==3.2.1           # Optional: For managing graph complexity locally

4. Implementation Plan (Step-by-Step)
Phase 1: The "Dumb" Parser (No AI yet)
* Goal: Read all files in a folder and map imports without spending money on API tokens.
* Action:
    1. Use file_walker.py to get a list of all .py or .js files.
    2. Use tree-sitter in parser_engine.py to find Class definitions and Import statements.
    3. Build a simple NetworkX graph where File A -> imports -> File B.
Phase 2: The AI Analyst
* Goal: Understand what the code does, not just what it imports.
* Action:
    1. In ai/summarizer.py, read the raw code of a file.
    2. Send it to the LLM with a prompt: "Summarize the responsibility of this class in 1 sentence. Identify if it creates a database connection or handles sensitive data."
    3. Store this metadata in your graph nodes.
Phase 3: The Mermaid Generator
* Goal: Visual Output.
* Action:
    1. Convert your internal NetworkX graph into Mermaid Syntax (graph TD; A[Auth.py] -->|Imports| B[Database.py];).
    2. Inject this string into templates/report_template.html.
    3. Open the HTML file automatically in the browser.

We are using tree-sitter because it constructs a concrete syntax tree (CST). Unlike Regex, it won't break if the code has weird formatting or comments in strange places.
We will use the tree_sitter_languages library, which comes with pre-built binaries for most languages, saving you hours of compilation headaches.
The Code: core/parser_engine.py
This module defines a CodeParser class. It currently targets Python, but I have structured the QUERIESdictionary so you can easily add JavaScript, Java, or Go later.
Python

import os
from tree_sitter_languages import get_language, get_parser

class CodeParser:
    def __init__(self, language_name="python"):
        """
        Initialize the parser for a specific language.
        """
        self.language_name = language_name
        self.language = get_language(language_name)
        self.parser = get_parser(language_name)
        
        # S-Expressions (Tree-sitter Queries)
        # These patterns act like Regex for the Syntax Tree.
        self.QUERIES = {
            "python": """
            (class_definition
                name: (identifier) @class_name)
            
            (function_definition
                name: (identifier) @function_name)
            
            (import_from_statement
                module_name: (dotted_name) @import_src)
            
            (import_statement
                name: (dotted_name) @import_lib)
            """
            # You can add "javascript", "java", etc. here later
        }

    def parse_file(self, file_path):
        """
        Reads a file and returns extracted metadata (Classes, Functions, Imports).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            code_bytes = bytes(f.read(), "utf8")

        # 1. Parse the code into a Tree
        tree = self.parser.parse(code_bytes)
        
        # 2. Run the Query to extract specific nodes
        query_scm = self.QUERIES.get(self.language_name)
        if not query_scm:
            return {"error": f"No queries defined for {self.language_name}"}

        query = self.language.query(query_scm)
        captures = query.captures(tree.root_node)

        # 3. Format the results
        results = {
            "file": file_path,
            "classes": [],
            "functions": [],
            "imports": []
        }

        for node, tag in captures:
            # Extract the actual text from the code using the node's byte range
            text = code_bytes[node.start_byte : node.end_byte].decode("utf8")
            
            if tag == "class_name":
                results["classes"].append(text)
            elif tag == "function_name":
                results["functions"].append(text)
            elif tag == "import_src" or tag == "import_lib":
                results["imports"].append(text)

        return results

# --- Quick Test Block (Runs if you execute this file directly) ---
if __name__ == "__main__":
    # create a dummy python file to test
    dummy_code = """
    import os
    from utils import helper
    
    class DatabaseManager:
        def connect(self):
            pass
    """
    with open("test_dummy.py", "w") as f:
        f.write(dummy_code)

    try:
        parser = CodeParser("python")
        data = parser.parse_file("test_dummy.py")
        
        from rich import print
        print("[bold green]Parsed Data Success:[/bold green]")
        print(data)
    finally:
        # Cleanup
        if os.path.exists("test_dummy.py"):
            os.remove("test_dummy.py")

How to Read This Code
1. get_language & get_parser: These load the pre-compiled binary logic for Python syntax.
2. The QUERIES String: This is the magic. It uses S-Expressions (Lisp-like syntax).
    * (class_definition name: (identifier) @class_name) tells Tree-sitter: "Look for a class definition, find its name identifier, and tag it as @class_name."
3. query.captures: This executes the query against the tree and returns every node that matched our tags.
Why This is Better than Regex
If you have a comment like # class FakeClass:, a Regex might accidentally pick that up as a real class. Tree-sitter ignores it because it understands that comments are not definitions.

The File Walker.
A naive implementation uses glob to find files, but that crashes on large projects because it tries to index node_modules or venv before filtering them out. We need a performant walker that prunes the directory tree—meaning if it sees a "forbidden" folder, it doesn't even enter it.
Here is the implementation for core/file_walker.py.
The Code: core/file_walker.py
Python

import os
from typing import List, Generator

class FileWalker:
    def __init__(self, root_dir: str, extensions: List[str] = None, ignore_dirs: List[str] = None):
        """
        :param root_dir: The starting folder path.
        :param extensions: List of file extensions to target (e.g., ['.py', '.js']).
        :param ignore_dirs: List of folder names to strictly skip (e.g., 'node_modules').
        """
        self.root_dir = os.path.abspath(root_dir)
        # Default to Python if nothing specified
        self.extensions = [e.lower() for e in extensions] if extensions else ['.py']
        
        # Standard junk folders to ignore
        default_ignores = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', '.vscode'}
        if ignore_dirs:
            default_ignores.update(ignore_dirs)
        self.ignore_dirs = default_ignores

    def walk(self) -> Generator[str, None, None]:
        """
        Yields absolute paths of valid files found in the directory tree.
        """
        print(f"🔎 Scanning: {self.root_dir} for {self.extensions}...")
        
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            # 1. Prune the directory list in-place to stop recursion into bad folders
            # We use a slice assignment [:] to modify the list used by os.walk
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                # 2. Check extension
                _, ext = os.path.splitext(file)
                if ext.lower() in self.extensions:
                    yield os.path.join(root, file)

# --- Quick Test Block ---
if __name__ == "__main__":
    # Test on the current directory
    walker = FileWalker(".", extensions=['.py'])
    for filepath in walker.walk():
        print(f"Found: {filepath}")

Integration: Connecting the Walker to the Parser
Now we have the two fundamental non-AI components:
1. The Walker (finds the files).
2. The Parser (extracts data from files).
Let's combine them into a script to verify we can generate a "Knowledge Graph" of your current codebase. Create a temporary file called test_integration.py in your root folder.
The Prototype: test_integration.py
Python

import json
from core.file_walker import FileWalker
from core.parser_engine import CodeParser

def build_codebase_map(target_directory):
    # 1. Initialize logic
    walker = FileWalker(target_directory, extensions=['.py'])
    parser = CodeParser("python")
    
    codebase_map = []

    # 2. Walk and Parse
    for file_path in walker.walk():
        try:
            # Parse the file
            data = parser.parse_file(file_path)
            
            # Add a 'short_name' for cleaner display later
            data['short_name'] = file_path.split("/")[-1] 
            
            codebase_map.append(data)
            print(f"✅ Parsed: {data['short_name']} -> {len(data['functions'])} functions")
            
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")

    return codebase_map

if __name__ == "__main__":
    # Run on the current folder
    result = build_codebase_map(".")
    
    # Dump to JSON to inspect the "Skeleton" of your project
    with open("codebase_skeleton.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print("\n[Job Done] Generated 'codebase_skeleton.json'.")
What just happened?
If you run python test_integration.py now, you will get a codebase_skeleton.json file. This JSON file contains the raw structure of your project.
Example Output JSON:
JSON

[
  {
    "file": "/Users/dev/project/core/parser_engine.py",
    "classes": ["CodeParser"],
    "functions": ["__init__", "parse_file"],
    "imports": ["os", "tree_sitter_languages"]
  }
]

Brain of the operation.
The Parser gave us the "What" (Code structure).
The AI Analyst will give us the "Why" (Business logic and risk).
We will use LangChain here because it handles the "plumbing" of working with LLMs—specifically forcing the model to return valid JSON, which is crucial for our final report.
Step 1: The Prompt Strategy
We don't just want a summary. We want structured data to build our graph.
We will ask the LLM for three specific things:
1. One-Liner: A concise description of the file's purpose.
2. Tags: Keywords (e.g., Database, Auth, UI, Utility) for color-coding the graph.
3. Risk Score (1-10): How dangerous or complex is this code?
Step 2: The Code (ai/summarizer.py)
Create the file ai/summarizer.py.
Note: You will need an OpenAI API Key for this to work. Ensure it is set in your environment as OPENAI_API_KEY.
Python

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class CodeSummarizer:
    def __init__(self, model_name="gpt-4-turbo-preview"):
        """
        Initializes the LLM connection and defines the expected output structure.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        
        # 1. Define the Output Schema (Force JSON Structure)
        summary_schema = ResponseSchema(name="summary", description="A 1-sentence explanation of what this code does.")
        tags_schema = ResponseSchema(name="tags", description="A list of 1-3 keywords (e.g., 'Database', 'API', 'Auth').")
        risk_schema = ResponseSchema(name="complexity_score", description="A number 1-10 indicating complexity or technical debt risk.")
        
        self.output_parser = StructuredOutputParser.from_response_schemas(
            [summary_schema, tags_schema, risk_schema]
        )
        self.format_instructions = self.output_parser.get_format_instructions()

    def analyze_file(self, filename, code_content, parsed_metadata):
        """
        Sends code + metadata to the LLM for analysis.
        """
        # If code is too huge, we truncate it to save tokens/money.
        # A smart archaeologist reads the head (imports/constants) and the structure.
        max_chars = 6000 
        truncated_code = code_content[:max_chars] + "\n...(truncated)..." if len(code_content) > max_chars else code_content

        # 2. Construct the Prompt
        template_string = """
        You are a Senior Software Architect auditing a legacy codebase.
        
        Analyze the following source code file: "{filename}"
        
        Context extracted by parser:
        - Classes: {classes}
        - Functions: {functions}
        - Imports: {imports}
        
        --- SOURCE CODE START ---
        {code}
        --- SOURCE CODE END ---
        
        Based on this, provide a high-level summary, tag the module type, and estimate complexity risk.
        
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template_string)

        # 3. Build the Chain
        # (Prompt -> LLM -> Parser)
        chain = prompt | self.llm | self.output_parser

        try:
            print(f"🤖 AI Analyzing: {filename}...")
            response = chain.invoke({
                "filename": filename,
                "classes": str(parsed_metadata.get('classes', [])),
                "functions": str(parsed_metadata.get('functions', [])),
                "imports": str(parsed_metadata.get('imports', [])),
                "code": truncated_code,
                "format_instructions": self.format_instructions
            })
            return response
            
        except Exception as e:
            print(f"❌ AI Error on {filename}: {e}")
            # Fallback safe response so the tool doesn't crash
            return {
                "summary": "Analysis failed.",
                "tags": ["Unknown"],
                "complexity_score": 0
            }

# --- Quick Test Block ---
if __name__ == "__main__":
    # Test with a fake file content
    fake_code = """
    import sqlite3
    class UserDB:
        def get_users(self):
            conn = sqlite3.connect('users.db')
            return conn.execute('SELECT * FROM users').fetchall()
    """
    
    # Fake metadata from our parser
    fake_meta = {
        "classes": ["UserDB"], 
        "functions": ["get_users"], 
        "imports": ["sqlite3"]
    }
    
    # You need an API KEY for this part to work!
    if os.getenv("OPENAI_API_KEY"):
        summarizer = CodeSummarizer()
        result = summarizer.analyze_file("database.py", fake_code, fake_meta)
        print("\n[AI Result]:", result)
    else:
        print("Please set OPENAI_API_KEY to run the test.")

Step 3: Updating the Integration Script
Now we go back to test_integration.py (our main script logic) and plug in the Brain.
We need to update the loop to:
1. Parse the file structure (Fast/Cheap).
2. Read the file content.
3. Send both to the AI (Slower/Costs Token).
4. Merge the results.
Here is the updated logical flow you should implement in your main controller:
Python

# Inside your main loop (conceptual update):

for file_path in walker.walk():
    # 1. Parse Structure
    structure_data = parser.parse_file(file_path)
    
    # 2. Read Content
    with open(file_path, "r", encoding="utf-8") as f:
        raw_code = f.read()
    
    # 3. AI Analysis
    ai_data = summarizer.analyze_file(file_path, raw_code, structure_data)
    
    # 4. Merge Data
    final_node = {**structure_data, **ai_data}
    
    # Now final_node has: 
    # { "file": "...", "classes": [...], "summary": "Handles DB...", "complexity_score": 8 }

Wait, a Cost Warning ⚠️
Running an LLM on every file in a massive project will burn through API credits quickly.
Architectural Optimization:
In the real main.py, we should add a --smart-scan flag.
* Full Scan: Sends every file to AI.
* Smart Scan: Only sends files that meet certain criteria (e.g., larger than 2KB, or contains "import" keywords) or simply relies on the Tree-sitter parser for the majority and only uses AI for "complex" files (files with > 5 functions).

Data (files & imports) and the Intelligence (AI summary & risk scores). Now we need to Visualize it.
A wall of text is useless. A color-coded map showing exactly which files are dangerous "God Objects" (High Risk) vs. simple utilities (Low Risk) is invaluable.
We will use Mermaid.js because it is text-based (easy for Python to generate) but renders as interactive SVG in the browser.

Step 1: The Graph Builder (core/graph_builder.py)
This module takes our analyzed data and converts it into Mermaid syntax.
Key Feature: We will use the AI's complexity_score (1-10) to apply dynamic styling:
* Red (8-10): High Risk / Technical Debt.
* Orange (5-7): Moderate Risk.
* Green (1-4): Safe / Simple.
Python

import os

class MermaidGenerator:
    def __init__(self, nodes):
        """
        nodes: List of dictionaries containing file data, AI summary, and imports.
        """
        self.nodes = nodes
        self.edges = []
        self.styles = []

    def sanitize(self, name):
        """Mermaid hates spaces and dots in IDs. Clean them up."""
        return name.replace(".", "_").replace("/", "_").replace("\\", "_").replace("-", "_")

    def generate_graph(self):
        graph_lines = ["graph TD"]
        
        # 1. Define Nodes with Styling
        for node in self.nodes:
            file_id = self.sanitize(node['short_name'])
            risk = node.get('complexity_score', 0)
            
            # Determine Color Class
            if risk >= 8:
                style_class = "danger"
            elif risk >= 5:
                style_class = "warning"
            else:
                style_class = "safe"
                
            # Create Node Text: "FileName\n(Risk: 8)"
            label = f"{node['short_name']}<br/>Risk: {risk}"
            
            # Add to graph
            graph_lines.append(f'    {file_id}("{label}")::: {style_class}')
            
            # 2. Build Edges (Imports)
            # We look at the imports list and try to match them to other files we scanned
            for imp in node.get('imports', []):
                # Simple heuristic: if an import matches another file's name
                for target in self.nodes:
                    # Check if import name (e.g. "utils") matches target filename ("utils.py")
                    target_name = target['short_name'].replace(".py", "")
                    if imp == target_name:
                        target_id = self.sanitize(target['short_name'])
                        graph_lines.append(f"    {file_id} --> {target_id}")

        # 3. Define Styles
        graph_lines.append("    classDef danger fill:#ffcccc,stroke:#ff0000,stroke-width:2px;")
        graph_lines.append("    classDef warning fill:#fff4cc,stroke:#ffaa00,stroke-width:2px;")
        graph_lines.append("    classDef safe fill:#ccffcc,stroke:#00aa00,stroke-width:1px;")
        
        # 4. Add Click Events (Optional interaction)
        # We can add click events here to show the AI summary in a tooltip later
        
        return "\n".join(graph_lines)

Step 2: The HTML Template (templates/report_template.html)
Create this HTML file. It contains the logic to render the graph and a side-panel for the detailed AI summaries.
HTML

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Legacy Code Archeology Report</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true });
    </script>
    <style>
        body { font-family: sans-serif; display: flex; height: 100vh; margin: 0; }
        #graph-container { flex: 2; overflow: auto; padding: 20px; background: #f4f4f4; border-right: 2px solid #ddd; }
        #details-panel { flex: 1; padding: 20px; overflow-y: auto; background: white; }
        .card { border: 1px solid #ccc; padding: 15px; margin-bottom: 10px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; color: white; margin-right: 5px;}
        .danger { background: #d9534f; }
        .warning { background: #f0ad4e; }
        .safe { background: #5cb85c; }
        h3 { margin-top: 0; }
    </style>
</head>
<body>

    <div id="graph-container">
        <h2>🏛️ Code Architecture Map</h2>
        <div class="mermaid">
            </div>
    </div>

    <div id="details-panel">
        <h2>📜 AI Analysis Logs</h2>
        </div>

</body>
</html>

Step 3: The Commander (main.py)
This ties everything together using Typer for a slick CLI experience.
Python

import os
import typer
import json
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv

# Import our modules
from core.file_walker import FileWalker
from core.parser_engine import CodeParser
from core.graph_builder import MermaidGenerator
from ai.summarizer import CodeSummarizer

# Load Env (API Key)
load_dotenv()

app = typer.Typer()
console = Console()

@app.command()
def audit(
    path: str = typer.Argument(..., help="Path to the codebase folder to audit"),
    smart_scan: bool = typer.Option(True, help="Skip AI analysis for tiny files to save cost?"),
    output: str = typer.Option("report.html", help="Output filename")
):
    """
    Scans a legacy codebase, uses AI to analyze risk, and generates a visual map.
    """
    console.print(f"[bold blue]🚀 Starting Archeology Audit on:[/bold blue] {path}")

    # 1. Initialize Tools
    walker = FileWalker(path, extensions=['.py'])
    parser = CodeParser("python")
    
    # Only init AI if we have a key
    if os.getenv("OPENAI_API_KEY"):
        summarizer = CodeSummarizer()
        ai_enabled = True
    else:
        console.print("[yellow]⚠️ No OPENAI_API_KEY found. Running in 'Structure Only' mode.[/yellow]")
        ai_enabled = False

    analyzed_nodes = []

    # 2. Walk and Process
    files = list(walker.walk())
    console.print(f"Found {len(files)} target files.")

    for file_path in track(files, description="Analyzing Codebase..."):
        try:
            # Parse Structure
            node_data = parser.parse_file(file_path)
            node_data['short_name'] = os.path.basename(file_path)

            # Read Content for AI
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()

            # AI Analysis (Conditional)
            if ai_enabled:
                # Simple heuristic: Only scan files > 10 lines
                if smart_scan and len(code_content.splitlines()) < 10:
                     node_data.update({
                        "summary": "Too small to analyze", 
                        "complexity_score": 1, 
                        "tags": ["Tiny"]
                    })
                else:
                    ai_result = summarizer.analyze_file(node_data['short_name'], code_content, node_data)
                    node_data.update(ai_result)
            
            analyzed_nodes.append(node_data)

        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")

    # 3. Generate Graph
    console.print("[green]🎨 generating visualization...[/green]")
    graph_gen = MermaidGenerator(analyzed_nodes)
    mermaid_syntax = graph_gen.generate_graph()

    # 4. Generate Side Panel HTML (The Cards)
    cards_html = ""
    for node in analyzed_nodes:
        risk = node.get('complexity_score', 0)
        risk_class = "danger" if risk >= 8 else "warning" if risk >= 5 else "safe"
        
        cards_html += f"""
        <div class="card">
            <h3>{node['short_name']} <span class="badge {risk_class}">Risk: {risk}</span></h3>
            <p><strong>Summary:</strong> {node.get('summary', 'No summary')}</p>
            <p><strong>Tags:</strong> {", ".join(node.get('tags', []))}</p>
            <p><strong>Imports:</strong> {len(node.get('imports', []))}</p>
        </div>
        """

    # 5. Inject into Template
    with open("templates/report_template.html", "r") as f:
        template = f.read()

    final_html = template.replace("", mermaid_syntax)
    final_html = final_html.replace("", cards_html)

    with open(output, "w") as f:
        f.write(final_html)

    console.print(f"[bold green]✅ Audit Complete![/bold green] Open [bold white]{output}[/bold white] in your browser.")

if __name__ == "__main__":
    app()

How to Run It
1. Set up environment: Bash  # Create the virtual environment
2. python -m venv venv
3. source venv/bin/activate  # or venv\Scripts\activate on Windows
4. 
5. # Install dependencies
6. pip install -r requirements.txt
7. 
8. # Create .env file with your key
9. echo "OPENAI_API_KEY=sk-..." > .env
10.    
11. Run the Audit: Point it at any Python project folder you have. Bash  python main.py audit ./path/to/legacy/project
12.    
13. View Results: Open report.html. You will see a graph on the left and detailed AI summaries on the right.
Final CodeForge Check
This project is now Feature Complete as an MVP.
1. Scanner: Finds files.
2. Parser: Extracts imports/classes (using Tree-sitter).
3. Analyst: Uses OpenAI to rate risk.
4. Reporter: visualizing the mess.


Running a full LLM analysis on a massive repository is the fastest way to drain your OpenAI credits. A repository with 500 files can easily cost $20–$50 in a single run if you aren't careful.
To make this tool production-ready, we need to implement Semantic Pruning and Caching.

1. Semantic Pruning (Token Saving)
Instead of sending the whole file, we send a "Skeleton." Since we already have the Tree-sitter parser, we can strip out the "meat" (the logic inside functions) and only send the "bones" (class names, function signatures, and docstrings).
Update to ai/summarizer.py:
Modify the code extraction logic to send a "condensed" version:
Python

def get_code_skeleton(tree, code_bytes):
    """
    Extracts only the class and function signatures, 
    stripping the function bodies to save tokens.
    """
    # Using Tree-sitter to find only the 'header' parts of definitions
    skeleton = []
    # (Conceptual logic: iterate tree and extract only definition lines)
    # This can reduce a 500-line file to 30 lines of pure structural metadata.
    return skeleton
2. Implementing the "Hash-Cache" System
The most expensive mistake is re-analyzing a file that hasn't changed. We will implement a local SQLite cache. If the MD5 Hash of a file matches our database, we skip the AI call and pull the previous summary.
New Module: core/cache_manager.py
Python

import sqlite3
import hashlib

class CacheManager:
    def __init__(self, db_path="archeology_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_cache (
                file_hash TEXT PRIMARY KEY,
                ai_summary TEXT,
                complexity_score INTEGER,
                tags TEXT
            )
        """)

    def get_cached_analysis(self, file_content):
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
        cursor = self.conn.execute(
            "SELECT ai_summary, complexity_score, tags FROM file_cache WHERE file_hash = ?", 
            (file_hash,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "summary": row[0],
                "complexity_score": row[1],
                "tags": row[2].split(",")
            }
        return None

    def save_to_cache(self, file_content, analysis):
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
        self.conn.execute(
            "INSERT OR REPLACE INTO file_cache VALUES (?, ?, ?, ?)",
            (file_hash, analysis['summary'], analysis['complexity_score'], ",".join(analysis['tags']))
        )
        self.conn.commit()

3. Tiered Analysis (The "Filter" Strategy)
We categorize files into "Low Interest" and "High Interest" before the AI sees them.
File Type	Strategy	Cost
Simple Utils (e.g., __init__.py)	Skip AI, use static tags.	$0.00
Medium Files (100-300 lines)	Use GPT-4o-mini (Cheap).	$0.01
God Objects (>500 lines / High Complexity)	Use GPT-4o (High Quality).	$0.05
Final Project Checklist
You now have a fully functional Legacy Code Archaeologist architecture:
1. Scanner (file_walker.py) - Finds the artifacts.
2. Parser (parser_engine.py) - Extracts the DNA.
3. Optimizer (cache_manager.py) - Saves you money.
4. Brain (summarizer.py) - Explains the mystery.
5. Visualizer (graph_builder.py) - Maps the ruins.

Containerizing this tool is a strategic move. Since we are using Tree-sitter, which often requires C compilers to build its language parsers, a Dockerfile ensures that any developer can run the "Archaeologist" on their machine without getting "Missing GCC" or "Linker Error" messages.

1. The Dockerfile
This Dockerfile uses a "Multi-stage" approach (optional, but here simplified for clarity) to ensure all C-dependencies for Tree-sitter are installed, then kept clean.
Dockerfile

# Use a slim Python base
FROM python:3.11-slim

# Install system dependencies for C-compilation (required for Tree-sitter)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a volume for the codebase to be audited
# This allows the container to see your local files
VOLUME ["/codebase"]

# Set the entrypoint to our main script
ENTRYPOINT ["python", "main.py", "audit", "/codebase"]

2. The Orchestrator (docker-compose.yml)
Using Docker Compose makes it much easier to pass your OpenAI API Key and mount your local project folder without typing long terminal commands.
YAML

version: '3.8'

services:
  archeologist:
    build: .
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
      # Change the line below to point to the project you want to audit
      - /path/to/your/target/project:/codebase
    # This ensures the report is saved to your local machine
    command: audit /codebase --output reports/final_audit.html

3. How to Launch the Audit
Once these files are in your project root, running an audit becomes a one-liner:
1. Build the Image: docker-compose build
2. Run the Scan: docker-compose up
Your report will automatically appear in the ./reports/ folder on your host machine.

Final Project Summary & Roadmap
You have built a sophisticated AI-powered developer tool. Here is the final architectural flow:
Component	Responsibility	Technical Secret
Tree-sitter	Structural Parsing	Bypasses the need for code to be "runnable" to analyze it.
LangChain	AI Orchestration	Uses Structured Output Parsers to ensure the UI doesn't break.
SQLite Cache	Cost Control	Prevents burning tokens on unchanged files.
Mermaid.js	Visualization	Renders complex graphs via simple string templates.
Docker	Portability	Package-once, run-anywhere deployment.
The "CodeForge" Hand-off
Your Legacy Code Archaeologist is now ready for its first mission.
Is there any specific feature you'd like to refine further? For example:
* Module 226: Act as a Text Summarization Algorithm Creator to make the AI reports even more concise.
* Module 77: Act as a Secure Code Reviewer to add "Security Vulnerability" detection to the complexity scores.
* Module 201: Act as a Web Scraping Expert to allow the tool to audit public GitHub URLs directly.


