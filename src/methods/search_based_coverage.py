from agents import Agent, Runner, WebSearchTool, function_tool, RunContextWrapper
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Union, Any
from src.utils.arxiv_wrapper import ArxivWrapper  



ARXIV_AGENT_PROMPT = """
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

3. Use Tools when you need.

4. For each masked citation:
   - Formulate precise search queries based on the context
   - Use the web search tool to find relevant paper titles
   - Select the most likely papers that fit the citation context
   - Provide multiple candidates when appropriate

# IMPORTANT NOTES:
- Each [CITATION] may represent MULTIPLE papers cited together
- You can focus more on quantity in your search results, since there could be multiple relevant papers!
- Use exact paper titles in your output
- Some citations might be seminal/classic papers in the field
- Pay attention to chronology (papers can't cite future work)
- Focus only on academic papers (not websites, blogs, etc.)

# OUTPUT FORMAT:
Return a list of possible paper titles for this part.
Structure your response according to the output schema.
"""


@function_tool
def arxiv_search(query: str, max_results: int) -> str:
    """
    Search the arXiv database for academic papers based on the provided query.
    
    This function supports arXiv's advanced search syntax, allowing precise queries
    using field specifiers and boolean operators to efficiently locate relevant papers.
    
    Args:
        query (str): Search query string, supporting advanced syntax
        max_results (int): Maximum number of results to return
        
    Returns:
        str: Formatted text containing search results with titles, authors, abstracts, and URLs
        
    Examples:
        # Basic search
        arxiv_search("transformer neural networks", 100)
        
        # Title exact match
        arxiv_search('ti:Attention is all you need', 3)
        
        # Author search
        arxiv_search('au:Hinton AND au:Bengio', 10)
        
        # Combined field search
        arxiv_search('ti:transformer AND au:Vaswani', 5)
        
        # Using boolean operators
        arxiv_search('ti:(attention NOT language) AND cat:cs.CL', 50)
        
        # Search by arXiv ID
        arxiv_search('id:1706.03762', 1)
        
        # Subject category search (cs.AI=Artificial Intelligence, cs.CL=Computational Linguistics)
        arxiv_search('cat:cs.AI AND ti:reinforcement', 50)
        
        # Date range search
        arxiv_search('submittedDate:[20200101 TO 20201231] AND ti:LLM', 100)
    
    Field Specifiers:
        ti: Title
        au: Author
        abs: Abstract
        co: Comments
        jr: Journal reference
        cat: Category/subject
        rn: Report number
        id: arXiv identifier
        all: All fields
    """
    arxiv = ArxivWrapper(query=query, max_results=max_results)
    results = arxiv.search_papers()
    
    return results



class CitationFillingResult(BaseModel):
    titles_of_cited_papers: List[str]


class ArxivSearchAgent():
    def __init__(self):
        self.arxiv_agent = Agent(
            name="arxiv agent",
            instructions=ARXIV_AGENT_PROMPT,
            model_settings=ModelSettings(tool_choice="required"),
            output_type=CitationFillingResult,
            model="gpt-4o",
            tools=[WebSearchTool(), arxiv_search],

        )
        
    def format_output_as_list(self, result: CitationFillingResult) -> list[str]:
        """Format the output as a list of lists."""
        formatted_result = result.titles_of_cited_papers
        
        return formatted_result
        
    def run(self, query: str):
        result = Runner.run_sync(
            self.arxiv_agent,
            query,
            max_turns=20
        )
        
        result = result.final_output_as(CitationFillingResult)
        
        result = self.format_output_as_list(result)
        
    
        return result

def predict_search_based(marked_text: str) -> List[List[str]]:
    """Predict citations using search-based methods."""
    agent = ArxivSearchAgent()
    result = agent.run(marked_text)
    return result

# def predict_search_based_sync(marked_text: str) -> List[List[str]]:
#     """Synchronous wrapper."""
#     import asyncio
#     return asyncio.run(predict_search_based(marked_text))



def test():
    agent = ArxivSearchAgent()
    test_query = """
    2.1. Human Motion Generation\n\nHuman motion generation has progressed significantly in\nrecent years [CITATION_0] leveraging a variety of conditioning sig-\nnals such as text [CITATION_1], actions [CITATION_2], speech [CITATION_3],\nmusic [CITATION_4], and scenes\/objects [CITATION_5]. Recently, multimodal motion generation has\nalso gained attention [CITATION_6] enabling multiple input\nmodalities. However, most existing methods focus solely\non generative tasks without supporting estimation. For in-\nstance, the method [CITATION_7] supports video input but treats it as\na generative task, resulting in motions that loosely imitate\nvideo content rather than precisely matching it. In contrast,\nour method jointly handles generation and estimation tasks,\nyielding more precise video-conditioned results.\n\nFor long-sequence motion generation, existing works\nmostly rely on ad-hoc post-processing techniques to stitch\nseparately generated fixed-length motions [CITATION_8]. In\ncontrast, our method introduces a novel diffusion-based ar-\nchitecture enabling seamless generation of arbitrary-length\nmotions conditioned on multiple modalities without com-\nplex post-processing.\n\nExisting datasets, such as AMASS [CITATION_9], are limited in\nsize and diversity. To address the scarcity of 3D data,\nMotion-X [CITATION_10] and MotionBank [CITATION_11] augment datasets us-\ning 2D videos and 3D pose estimation models [CITATION_12], but\nthe resulting motions often contain artifacts.\nIn contrast,\nour method directly leverages in-the-wild videos with 2D\nannotations without explicit 3D reconstruction, reducing re-\nliance on noisy data and enhancing robustness and diversity.\n\n2.2. Human Motion Estimation\n\nHuman pose estimation from images [CITATION_13], videos [CITATION_14], or even sparse marker data [CITATION_15] has been\nstudied extensively in the literature. Recent works focus pri-\nmarily on estimating global human motion in world-space\ncoordinates [CITATION_16]. This is an inherently\nill-posed problem, hence these methods leverage generative\npriors and SLAM methods to constrain human and camera\nmotions, respectively. However, these methods typically in-\nvolve computationally expensive optimization or separate\npost-processing steps.\n\nMore recent approaches aim to estimate global human\nmotion in a feed-forward manner [CITATION_17], offer-\ning faster solutions. Our method extends this direction by\njointly modeling generation and estimation within a uni-\nfied diffusion framework. This integration leverages shared\nrepresentations and generative priors during training to pro-\nduce more plausible estimations.
    """
    result = agent.run(test_query)
    print(result)
    
if __name__ == "__main__":
    test()