import os, typer
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv

# Import our modules
from core.file_walker import FileWalker
from core.parser_engine import CodeParser
from core.graph_builder import MermaidGenerator
from core.cache_manager import CacheManager
from ai.summarizer import CodeSummarizer

load_dotenv()
app = typer.Typer()
console = Console()

@app.command()
def audit(path: str, output: str = "report.html"):
    """
    Scans a legacy codebase, uses AI to analyze risk, and generates a visual map.
    """
    if not os.path.exists(path):
        console.print(f"[bold red]❌ Error: Path '{path}' does not exist.[/bold red]")
        raise typer.Exit()

    # 1. Initialize Tools
    walker = FileWalker(path)
    parser = CodeParser("python")
    cache = CacheManager()
    
    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️ No OPENAI_API_KEY found. AI analysis will be skipped.[/yellow]")
        summarizer = None
    else:
        summarizer = CodeSummarizer()
    
    analyzed_nodes = []
    files = list(walker.walk())
    
    # 2. Walk and Process
    for file_path in track(files, description=f"Auditing {len(files)} files..."):
        try:
            # Parse Structure (Fast)
            node_data = parser.parse_file(file_path)
            node_data['short_name'] = os.path.basename(file_path)
            
            # Read Content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()

            # AI Analysis (Slow/Cached)
            if summarizer:
                cached_result = cache.get(code_content)
                if cached_result:
                    node_data.update(cached_result)
                else:
                    # Only analyze if not in cache
                    ai_result = summarizer.analyze_file(node_data['short_name'], code_content, node_data)
                    cache.save(code_content, ai_result)
                    node_data.update(ai_result)
            else:
                # Default values if no AI
                node_data.update({"complexity_score": 1, "summary": "AI Disabled", "tags": []})
            
            analyzed_nodes.append(node_data)
            
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")

    # 3. Generate Visualization (Mermaid)
    console.print("[blue]🎨 Building Graph...[/blue]")
    graph_gen = MermaidGenerator(analyzed_nodes)
    mermaid_syntax = graph_gen.generate_graph()

    # 4. Generate Side Panel Cards (HTML Generation)
    console.print("[blue]📝 Compiling Report...[/blue]")
    cards_html = ""
    for node in analyzed_nodes:
        risk = node.get('complexity_score', 0)
        
        # Determine badge color
        badge_class = "risk-high" if risk >= 8 else "risk-med" if risk >= 5 else "risk-low"
        
        # Build tags HTML
        tags_html = "".join([f"<span>{t}</span>" for t in node.get('tags', [])])
        
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="filename">{node['short_name']}</span>
                <span class="badge {badge_class}">Risk: {risk}/10</span>
            </div>
            <div class="meta">Imports: {len(node.get('imports', []))} | Functions: {len(node.get('functions', []))}</div>
            <div class="summary">{node.get('summary', 'No summary available.')}</div>
            <div class="tags">{tags_html}</div>
        </div>
        """

    # 5. Final Injection into Template
    template_path = os.path.join("templates", "report_template.html")
    if not os.path.exists(template_path):
        console.print("[bold red]❌ Template file missing! Check templates/report_template.html[/bold red]")
        raise typer.Exit()

    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Replace placeholders
    final_html = html_content.replace("<!-- GRAPH -->", mermaid_syntax)
    final_html = final_html.replace("<!-- CARDS -->", cards_html)

    with open(output, "w", encoding="utf-8") as f:
        f.write(final_html)

    console.print(f"[bold green]✅ Success![/bold green] Open [bold white]{output}[/bold white] to view your audit.")

if __name__ == "__main__":
    app()
