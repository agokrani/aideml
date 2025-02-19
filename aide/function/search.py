from xml.etree import ElementTree
import requests


def search_arxiv(query, max_results=8):
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": max_results}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: Unable to fetch data from arXiv (Status code: {response.status_code})"

    root = ElementTree.fromstring(response.content)
    output = ""
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        link = entry.find("{http://www.w3.org/2005/Atom}id").text
        published = entry.find("{http://www.w3.org/2005/Atom}published").text
        authors = [
            author.find("{http://www.w3.org/2005/Atom}name").text
            for author in entry.findall("{http://www.w3.org/2005/Atom}author")
        ]

        output += f"""
        Title: {title.strip()}
        Summary: {summary.strip()}
        Link: {link.strip()}
        Published: {published.strip()}
        Authors: {authors}
        """

    return output


def search_papers_with_code(query: str, max_results: int = 8) -> str:
    url = "https://paperswithcode.com/api/v1/search/"
    response = requests.get(url, params={"page": 1, "q": query})
    if response.status_code != 200:
        return "Failed to retrieve data from Papers With Code."

    data = response.json()
    if "results" not in data:
        return "No results found for the given query."

    results = data["results"][: int(max_results)]  # Get top-k results
    result_strings = []

    for result in results:
        paper = result["paper"]
        paper_title = paper.get("title", "No title available")
        abstract = paper.get("abstract", "No abstract available")
        paper_pdf_url = paper.get("url_pdf", "No PDF available")
        repository = result.get("repository", [])
        if repository:
            code_url = repository.get("url", "No official code link available")
        else:
            code_url = "No official code link available"

        result_string = f"Title: {paper_title}\nAbstract:{abstract}\nPaper URL: {paper_pdf_url}\nCode URL: {code_url}\n"
        result_strings.append(result_string)

    return "\n".join(result_strings)
