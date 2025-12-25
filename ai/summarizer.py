from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class CodeSummarizer:
    def __init__(self, model_name="gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        
        # Define expected JSON structure
        schemas = [
            ResponseSchema(name="summary", description="1-sentence explanation of responsibility."),
            ResponseSchema(name="tags", description="List of 1-3 keywords (e.g., 'Auth', 'DB')."),
            ResponseSchema(name="complexity_score", description="Int 1-10. 10 is technical debt.")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

    def analyze_file(self, filename, code_content, metadata):
        # Truncate strictly for cost control
        truncated_code = code_content[:6000] 
        
        template = """
        Analyze source file: "{filename}"
        Metadata: {metadata}
        --- CODE ---
        {code}
        --- END ---
        Provide summary, tags, and risk score (1-10).
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.output_parser
        
        try:
            return chain.invoke({
                "filename": filename,
                "metadata": str(metadata),
                "code": truncated_code,
                "format_instructions": self.format_instructions
            })
        except Exception:
            return {"summary": "Analysis Failed", "complexity_score": 0, "tags": []}
