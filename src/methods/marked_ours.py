"""
Module providing baseline citation prediction methods for marked citation text.
"""

import os
import re
import json
import random
import logging
import serpapi
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

prompt_get_query = """
Here is a related work content, but all citations are replaced with [citation_num] masks.
Your job is to generate a string as the following function's input.
You should generate query term for **each** missing [citation_num] mask.

Function defined as follows:
def get_related_works(query: str) -> str:
    Retrieve related works for each citation topic listed in the query.

    This function takes a structured multi-line query string, where each citation section starts with a line ending 
    in a colon (":") followed by one or more subtopic keywords or query terms. For each query term, the function 
    returns a set of relevant academic papers or research articles.

    Important Note:
        This function only retrieves potentially related papers based on the provided topics. It is the responsibility 
        of the caller to review the results and determine which papers are truly appropriate for inclusion as related works.

    Args:
        query (str): A multi-line string formatted as:
            citation1:
            query_term_1
            query_term_2
            citation2:
            query_term_3
            ...

    Returns:
        str: A formatted string summarizing retrieved papers for each citation and query term.

    Example:
        >>> query = \"\"\"
        ... citation1:
        ... deep learning
        ... neural networks
        ... citation2:
        ... computer vision
        ... image classification
        ... \"\"\"
        >>> print(get_related_works(query))
        citation1:
          - [Title 1] Author A et al. (2020)
          - [Title 2] Author B et al. (2021)
        citation2:
          - [Title 3] Author C et al. (2019)
          - [Title 4] Author D et al. (2022)

Here is your masked version of related work:
<<<masked_related_work>>>

You should strictly follow the function args input format.
"""

prompt_get_most_relevant_single = """
Task:
You are given a related works section where all citation placeholders are masked. 
Your goal is to select the single most appropriate paper to fill the placeholder <<<citation_mask_num>>>.

Input:
- Masked related works context:
  <<<masked_related_work>>>
- Candidate papers for <<<citation_mask_num>>>:
  <<<papers_for_masked_related_work>>>

Instruction:
Choose exactly one paper from the candidates that best fits the masked citation <<<citation_mask_num>>> in the context above.
Only output the title of the selected paper. Do not explain your choice.
"""


prompt_get_most_relevant_group = """
Task:
You are given a related works section where all citation placeholders are masked.
Your goal is to identify candidate papers that are relevant to filling the placeholder <<<citation_mask_num>>>.

Input:
- Masked related works context:
  <<<masked_related_work>>>
- Candidate papers for <<<citation_mask_num>>>:
  <<<papers_for_masked_related_work>>>

Instruction:
Select papers from the candidate list that you consider relevant for the masked citation <<<citation_mask_num>>> in the context above.
You are free to select any number of papers that you think fits this position.
Output a list of titles, one paper title per line. Do not provide any additional commentary or explanation.
"""


def temp_query_openai(user_message: str, model="gpt-4o-mini") -> str | None:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    temp_messages = [{"role": "user", "content": user_message}]
    data = {
        "model": model,
        "messages": temp_messages
    }
    
    if model == "gpt-4o-mini":
        data["temperature"] = 0.7

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)

    if response.status_code != 200:
        return None
    
    bot_reply = response.json()["choices"][0]["message"]["content"]

    return bot_reply


def parse_and_process_query_related_works(query: str) -> Dict[str, List[str]]:
    
    def query_single(query_term: str) -> List[str]:
        """
        Query SerpAPI for a given term and return a list of related papers.

        Args:
            query_term (str): A string like "transformer robustness"

        Returns:
            List[str]: A list of strings like "Title: Snippet", sorted by position.
        """
        result = serpapi.search(
            q=f"{query_term}",
            engine="google_scholar",
            api_key=SERPAPI_KEY,
            start=0,
            num=10
        )

        sorted_papers = sorted(result["organic_results"], key=lambda paper: paper.get("position", float("inf")))
        return [f"{paper['title']}: {paper['snippet']}" for paper in sorted_papers]

    result = {}
    current_citation = None

    for line in tqdm(query.strip().splitlines(), desc="querying related works..."):
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_citation = line[:-1]
            result[current_citation] = []
        else:
            if current_citation is None:
                raise ValueError(f"Query term '{line}' encountered before any citation header.")
            papers = query_single(line)
            result[current_citation].extend(papers)

    return result


def fill_related_works(masked_related_work: str, 
                       only_one_paper_per_slot: bool = False) -> List[List[str]]:
    temp_prompt = prompt_get_query.replace("<<<masked_related_work>>>", masked_related_work)
    query_text = temp_query_openai(temp_prompt, model="o3-mini")

    print(query_text)

    related_all_papers = parse_and_process_query_related_works(query_text)

    output = []

    for mask_num, related_paper in tqdm(related_all_papers.items(), desc="selecting papers..."):
        logger.info(mask_num)
        if only_one_paper_per_slot:
            temp_prompt = prompt_get_most_relevant_single
        else:
            temp_prompt = prompt_get_most_relevant_group
        temp_prompt = temp_prompt.replace("<<<masked_related_work>>>", masked_related_work)
        temp_prompt = temp_prompt.replace("<<<papers_for_masked_related_work>>>", "\n".join(related_paper))
        temp_prompt = temp_prompt.replace("<<<citation_mask_num>>>", mask_num)
        relevant_paper = temp_query_openai(temp_prompt, model="o3-mini")
        relevant_paper_list = relevant_paper.splitlines()
        logger.info(relevant_paper_list)
        output.append(relevant_paper_list)
    
    return output


def predict_graph_citations(marked_text: str) -> List[List[str]]:
    return fill_related_works(marked_text)


if __name__ == "__main__":
    print(SERPAPI_KEY)