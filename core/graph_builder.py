class MermaidGenerator:
    def __init__(self, nodes):
        self.nodes = nodes

    def sanitize(self, text):
        # Mermaid IDs cannot contain slashes, dots, or spaces
        return text.replace(".", "_").replace("/", "_").replace("\\", "_").replace("-", "_").replace(" ", "_")

    def generate_graph(self):
        lines = ["graph TD"]
        # Define Styles
        lines.append("classDef danger fill:#ffcccc,stroke:#ff0000,stroke-width:2px;")
        lines.append("classDef warning fill:#fff4cc,stroke:#ffaa00,stroke-width:2px;")
        lines.append("classDef safe fill:#ccffcc,stroke:#00aa00,stroke-width:1px;")

        # Lookup table to map filenames to full IDs (for linking imports)
        # We map "utils" -> ["api_utils_py", "db_utils_py"] to handle ambiguity later if needed
        file_map = {n['short_name'].replace(".py", ""): self.sanitize(n['rel_path']) for n in self.nodes}

        for node in self.nodes:
            # Use RELATIVE PATH for unique ID, but SHORT NAME for the Label
            node_id = self.sanitize(node['rel_path'])
            label = node['short_name']
            
            risk = node.get('complexity_score', 0)
            style = "danger" if risk >= 8 else "warning" if risk >= 5 else "safe"
            
            # Node Syntax: id("Label<br/>Risk"):::style
            lines.append(f'    {node_id}("{label}<br/>Risk: {risk}"):::{style}')
            
            # Edges (Imports)
            for imp in node.get('imports', []):
                # If the imported name matches a file we scanned
                if imp in file_map:
                    target_id = file_map[imp]
                    # Prevent self-referencing loops
                    if target_id != node_id:
                        lines.append(f"    {node_id} --> {target_id}")
                        
        return "\n".join(lines)
