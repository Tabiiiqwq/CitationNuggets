from agents import Agent, Runner
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from typing import List


PURE_LLM_PROMPT = """
You are an academic citation expert specialized in filling in missing citations in research papers.

# YOUR TASK:
You'll be given some parts from a research paper where citations have been masked as [CITATION].
For each masked citation, you need to identify the most likely paper **titles** that could be referenced at that position.

# HOW TO ANALYZE:
1. Carefully read the context around each masked citation.
2. Identify key clues:
   - Author names mentioned
   - Years mentioned
   - Specific techniques/methods described
   - Contributions attributed to the masked citation
   - How the citation relates to the current paper

3. For each masked citation:
   - Think about what papers would be relevant based on the context
   - Consider seminal works in the field that match the description
   - Use your knowledge of academic literature to suggest plausible titles
   - Provide multiple candidates when appropriate

# IMPORTANT NOTES:
- Each [CITATION] may represent MULTIPLE papers cited together
- Use exact paper titles in your output when possible
- Some citations might be seminal/classic papers in the field
- Pay attention to chronology (papers can't cite future work)
- Focus only on academic papers (not websites, blogs, etc.)
- If you are uncertain about exact titles, generate plausible ones based on the context

# OUTPUT FORMAT:
Return a list of possible paper titles for this part.
Structure your response according to the output schema.
"""


class CitationFillingResult(BaseModel):
    titles_of_cited_papers: List[str]


class PureLLMCitationAgent:
    def __init__(self):
        self.llm_agent = Agent(
            name="pure llm citation agent",
            instructions=PURE_LLM_PROMPT,
            output_type=CitationFillingResult,
            model="gpt-4o",
            # No tools here - pure LLM reasoning only
        )
        
    def format_output_as_list(self, result: CitationFillingResult) -> list[str]:
        """Format the output as a list of strings."""
        formatted_result = result.titles_of_cited_papers
        
        return formatted_result
        
    def run(self, query: str):
        result = Runner.run_sync(
            self.llm_agent,
            query,
            max_turns=1  # Only needs one turn since there are no tools
        )
        
        result = result.final_output_as(CitationFillingResult)
        
        result = self.format_output_as_list(result)
        
        return result


def predict_pure_llm(marked_text: str) -> List[str]:
    """Predict citations using only the LLM without external search."""
    agent = PureLLMCitationAgent()
    result = agent.run(marked_text)
    return result


def test():
    agent = PureLLMCitationAgent()
    test_query = """
    2.1. Human Motion Generation

    Human motion generation has progressed significantly in
    recent years [CITATION_0] leveraging a variety of conditioning sig-
    nals such as text [CITATION_1], actions [CITATION_2], speech [CITATION_3],
    music [CITATION_4], and scenes/objects [CITATION_5].
    """
    result = agent.run(test_query)
    print(result)
    
    
if __name__ == "__main__":
    test()