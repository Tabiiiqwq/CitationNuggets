from agents import Agent, Runner, WebSearchTool, function_tool, RunContextWrapper
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Union, Any


ARXIV_AGENT_PROMPT = """
You are an academic citation expert specialized in filling in missing citations in research papers.

# YOUR TASK:
You'll be given a "Related Work" section from a research paper where citations have been masked as [CITATION_0], [CITATION_1], etc.
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
- Each [CITATION_X] may represent MULTIPLE papers cited together
- Focus on quality over quantity in your search results
- Use exact paper titles in your output
- Some citations might be seminal/classic papers in the field
- Pay attention to chronology (papers can't cite future work)
- Focus only on academic papers (not websites, blogs, etc.)

# OUTPUT FORMAT:
Return a list of possible paper titles for each citation slot.
Structure your response according to the output schema。
"""



class CitationFillingResult(BaseModel):
    titles_for_one_citation_slot: list[str]
    
    
class RelatedWorksCitationResult(BaseModel):
    citation_for_all_slots: list[CitationFillingResult]

class ArxivSearchAgent():
    def __init__(self):
        self.arxiv_agent = Agent(
            name="arxiv agent",
            instructions=ARXIV_AGENT_PROMPT,
            model_settings=ModelSettings(tool_choice="required"),
            output_type=RelatedWorksCitationResult,
            model="gpt-4o",
            tools=[WebSearchTool()],
        )
        
    def format_outpur_as_list(self, result: RelatedWorksCitationResult) -> list[list[str]]:
        """Format the output as a list of lists."""
        formatted_result = []
        for citation in result.citation_for_all_slots:
            formatted_result.append(citation.titles_for_one_citation_slot)
        return formatted_result
        
    async def run(self, query: str):
        result = await Runner.run(
            self.arxiv_agent,
            query,
        )
        
        result = result.final_output_as(RelatedWorksCitationResult)
        
        result = self.format_outpur_as_list(result)
        
    
        return result

async def predict_search_based_marked_citations(marked_text: str) -> List[List[str]]:
    """Predict citations using search-based methods."""
    agent = ArxivSearchAgent()
    result = await agent.run(marked_text)
    return result

def predict_search_based_marked_citations_sync(marked_text: str) -> List[List[str]]:
    """Synchronous wrapper for predict_search_based_marked_citations."""
    import asyncio
    return asyncio.run(predict_search_based_marked_citations(marked_text))



async def test():
    agent = ArxivSearchAgent()
    test_query = """
    2.1. Human Motion Generation\n\nHuman motion generation has progressed significantly in\nrecent years [CITATION_0] leveraging a variety of conditioning sig-\nnals such as text [CITATION_1], actions [CITATION_2], speech [CITATION_3],\nmusic [CITATION_4], and scenes\/objects [CITATION_5]. Recently, multimodal motion generation has\nalso gained attention [CITATION_6] enabling multiple input\nmodalities. However, most existing methods focus solely\non generative tasks without supporting estimation. For in-\nstance, the method [CITATION_7] supports video input but treats it as\na generative task, resulting in motions that loosely imitate\nvideo content rather than precisely matching it. In contrast,\nour method jointly handles generation and estimation tasks,\nyielding more precise video-conditioned results.\n\nFor long-sequence motion generation, existing works\nmostly rely on ad-hoc post-processing techniques to stitch\nseparately generated fixed-length motions [CITATION_8]. In\ncontrast, our method introduces a novel diffusion-based ar-\nchitecture enabling seamless generation of arbitrary-length\nmotions conditioned on multiple modalities without com-\nplex post-processing.\n\nExisting datasets, such as AMASS [CITATION_9], are limited in\nsize and diversity. To address the scarcity of 3D data,\nMotion-X [CITATION_10] and MotionBank [CITATION_11] augment datasets us-\ning 2D videos and 3D pose estimation models [CITATION_12], but\nthe resulting motions often contain artifacts.\nIn contrast,\nour method directly leverages in-the-wild videos with 2D\nannotations without explicit 3D reconstruction, reducing re-\nliance on noisy data and enhancing robustness and diversity.\n\n2.2. Human Motion Estimation\n\nHuman pose estimation from images [CITATION_13], videos [CITATION_14], or even sparse marker data [CITATION_15] has been\nstudied extensively in the literature. Recent works focus pri-\nmarily on estimating global human motion in world-space\ncoordinates [CITATION_16]. This is an inherently\nill-posed problem, hence these methods leverage generative\npriors and SLAM methods to constrain human and camera\nmotions, respectively. However, these methods typically in-\nvolve computationally expensive optimization or separate\npost-processing steps.\n\nMore recent approaches aim to estimate global human\nmotion in a feed-forward manner [CITATION_17], offer-\ning faster solutions. Our method extends this direction by\njointly modeling generation and estimation within a uni-\nfied diffusion framework. This integration leverages shared\nrepresentations and generative priors during training to pro-\nduce more plausible estimations.
    """
    result = await agent.run(test_query)
    print(result)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test())