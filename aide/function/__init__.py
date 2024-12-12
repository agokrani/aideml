from pydantic import BaseModel, Field
from aide.function.search import search_arxiv, search_papers_with_code

func_name_to_func = {
    "SearchArxiv": search_arxiv,
    "SearchPapersWithCode": search_papers_with_code,
}


class SearchArxiv(BaseModel):
    """
    Search for papers on arXiv and return the top results based on keywords
    (task, model, dataset, etc.) Use this function when there is a need to search
    for research papers.
    """

    query: str = Field(description="The search query to perform")
    max_results: int = Field(description="The maximum number of results to return")


class SearchPapersWithCode(BaseModel):
    """
    Search for papers on Papers with Code and return the top results based on keywords
    (task, model, dataset, etc.) Use this function when there is a need to search
    for research papers and the source code.
    """

    query: str = Field(description="The search query to perform")
    max_results: int = Field(description="The maximum number of results to return")


def get_function(func_name: str) -> BaseModel:
    if func_name in func_name_to_func.keys():
        return func_name_to_func[func_name]
    else:
        return None
