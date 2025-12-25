import os
from tree_sitter_languages import get_language, get_parser

class CodeParser:
    def __init__(self, language_name="python"):
        self.language_name = language_name
        self.language = get_language(language_name)
        self.parser = get_parser(language_name)
        
        # S-Expressions for extraction
        self.QUERIES = {
            "python": """
            (class_definition name: (identifier) @class_name)
            (function_definition name: (identifier) @function_name)
            (import_from_statement module_name: (dotted_name) @import_src)
            (import_statement name: (dotted_name) @import_lib)
            """
        }

    def parse_file(self, file_path):
        if not os.path.exists(file_path): return {}
        
        with open(file_path, "rb") as f:
            code_bytes = f.read()

        tree = self.parser.parse(code_bytes)
        query = self.language.query(self.QUERIES.get(self.language_name))
        captures = query.captures(tree.root_node)

        results = {"classes": [], "functions": [], "imports": []}
        
        for node, tag in captures:
            text = code_bytes[node.start_byte : node.end_byte].decode("utf8")
            if tag == "class_name": results["classes"].append(text)
            elif tag == "function_name": results["functions"].append(text)
            elif tag in ["import_src", "import_lib"]: results["imports"].append(text)
            
        return results
