import arxiv
import pandas as pd
import os
import re
import requests
from pdfminer.high_level import extract_text
import io

class ArxivWrapper:
    """
    A wrapper for the arXiv API.

    Attributes:
        query (str): The query string for the search.
        max_results (int): The maximum number of results to return.

    """

    def __init__(self, query, max_results):
        """
        The constructor for ArxivWrapper class.

        Parameters:
            query (str): The query string for the search.
            max_results (int): The maximum number of results to return.
        """
        self.query = query
        self.max_results = max_results

    def search_papers(self):
        """
        Search for papers using the arXiv API and return a DataFrame.

        Returns:
            df (DataFrame): A DataFrame containing the search results.
        """
        output_str = ''
       
        client = arxiv.Client()

        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            # sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

    
        for result in client.results(search):
            entry_id = result.entry_id
            uid = entry_id.split(".")[-1]
            title = result.title
            date_published = result.published
            abstract = result.summary
            pdf_url = result.pdf_url
            # response = requests.get(result.pdf_url)
            # pdf_bytes = io.BytesIO(response.content)

            # text = extract_text(pdf_bytes)
    
            output_str += f"Title: {title}\n"
            output_str += f"Date Published: {date_published}\n"
            output_str += f"Abstract: {abstract}\n"
            output_str += f"PDF URL: {pdf_url}\n"
            # output_str += f"Text: {text}\n\n"
            output_str += "-" * 80 + "\n\n"
        
        return output_str
    
if __name__ == "__main__":
    query = 'submittedDate:[20200101 TO 20201231] AND ti:LLM'
    max_results = 2
    arxiv_wrapper = ArxivWrapper(query, max_results)
    result = arxiv_wrapper.search_papers()
    print(result)