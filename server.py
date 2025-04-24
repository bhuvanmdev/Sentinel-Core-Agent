from mcp.server.fastmcp import server, FastMCP
import re
import requests
from typing import Optional, Union
import asyncio
import sys
from contextlib import asynccontextmanager
from crawl4ai import *
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter
from webagent import WebResearchAgent

import argparse

parser = argparse.ArgumentParser(description="MCP Server")
parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on")
parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face model",default=False)


args = parser.parse_args()
crawler = None

@asynccontextmanager
async def lifespan(app: FastMCP):
    global crawler
    # Startup: create and initialize the global crawler
    crawler = AsyncWebCrawler(verbose=False, always_by_pass_cache=True)
    await crawler.__aenter__()
    yield
    # Shutdown: clean up the crawler
    if crawler:
        await crawler.__aexit__(None, None, None)

sys.stdout.reconfigure(encoding='utf-8')

import logging
log = logging.getLogger("mcp")

log.info("Getting the embedding model...")
if args.use_hf:
    embed_model = HuggingFaceEmbeddings(

        model_name=  "intfloat/e5-large-v2",#"NovaSearch/stella_en_1.5B_v5"  , #"nvidia/NV-Embed-v2",  #
        model_kwargs={"trust_remote_code":True,'device': 'cuda',#"model_kwargs":{"device_map": "auto" if torch.cuda.is_available() else 'cpu',}
    },
        encode_kwargs={'normalize_embeddings': True}
    )

mcp = FastMCP(name="Python Tools server",lifespan=lifespan)


@mcp.tool()
def is_file_folder_present(file_folder_name: str, path: Optional[str] = None) -> str:
    """parse the entire file system and check if a file or folder exists"""
    import os
    # if path is None or empty, set it to the list of partitions
    if path is None:
        partitions = [f"{chr(i)}:\\" for i in range(67, 91) if os.path.exists(f"{chr(i)}:\\")]
        for partition in partitions:
            path = ''.join(partition.split("\\")[:-1])
            # return path if the file or folder is found
            for root, dirs, files in os.walk(partition):
                path = os.path.join(path, root.split("\\")[-1])
                if file_folder_name.lower() in map(str.lower,dirs) or file_folder_name.lower() in map(str.lower,files):
                    return f"{file_folder_name} found in {path}"
    else:
        fin_path = path
        for root, dirs, files in os.walk(path):
            fin_path = os.path.join(fin_path, root.split("\\")[-1])
            if file_folder_name.lower() in map(str.lower,dirs) or file_folder_name.lower() in map(str.lower,files):
                return f"{file_folder_name} found in {fin_path}"
    return f"{file_folder_name} not found in {path}"
                

        
@mcp.tool()
def cur_datetimetime() -> str:
    """Get the current time"""
    from datetime import datetime
    now = datetime.now()
    # print(f"Current Time and date: {now}")
    return now

def braveai(query,temp=False,hash_=None,id_=None,ind=1):
    url_encoding = {
    "$": "%24",
    "&": "%26",
    "+": "%2B",
    ",": "%2C",
    ":": "%3A",
    ";": "%3B",
    "=": "%3D",
    "?": r"%3F",
    "@": "%40",
    " ": "+",
    "#": "%23",
    "<": "%3C",
    ">": r"%3E",
    "%": "%25",
    "{": "%7B",
    "}": "%7D",
    "|": "%7C",
    "\\": "%5C",
    "^": r"%5E",
    "~": r"%7E",
    "[": "%5B",
    "]": "%5D",
    "`": "%60",
    "'": "%27",
    '"': "%22",
    "/": r"%2F",
    "(": "%28",
    ")": "%29",

}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }
    if temp:
        url = 'https://search.brave.com/search?q=what+is+langchain.&source=web&summary=1'
        ini = "."#input("Enter the query: ")
        ini = ''.join([url_encoding.get(x, x) for x in ini])
        response = requests.get(url, headers=headers)
        hash_ = re.findall(r"results_hash\\\":\\\"([a-f0-9]+)\\\"",response.text)[0]
        id_  = re.findall(r"conversation\".*:.*\"([a-f0-9]+)\"",response.text)[0]
        return hash_,id_
    else:
        ini = query
        inp = "From this point onwards answer all the questions asked, only in markup format. This includes bold, newline and links. Ignore all other stylings and no json formatting allowed.\n\n   " + query 
        ini = ''.join([url_encoding.get(x, x) for x in ini])
        url = r"https://search.brave.com/api/chatllm/conversation?key=%7B%22query%22%3A%22"+ini+r"%22%2C%22country%22%3A%22us%22%2C%22language%22%3A%22en%22%2C%22safesearch%22%3A%22moderate%22%2C%22results_hash%22%3A%22"+hash_+r"%22%7D&conversation="+id_+r"&index="+str(ind)+r"&followup="+inp
        response = requests.get(url, headers=headers)
        output = ""
        for x in response.text.split('"\n"'):
            output += x.strip('"')
        return output.replace('\\n','\n').replace('\\','')

# @mcp.tool()
# def general_url_info_provider(query: str) -> str:
#     """Providex you the general fotmat the url is supposed to be but not the specific url. If any specific url is asked, they will start hallucinating."""
#     hash_,id_ = braveai(query,temp=True)
#     return braveai(query,temp=False,hash_=hash_,id_=id_)


@mcp.tool()
def read_file(path: str) -> str:
    """Read the content of a file of a windows file."""
    print(f"Executing resource 'read_file' with path={path}")
    try:
        with open(path, 'r') as file:
            content = file.read()
        print(f"Result: {content}")
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return str(e)
    
@mcp.tool()
def write_file(path: str, content: str, binary_data: bool) -> str:
    """Write both text and binary files."""
    try:
        if binary_data:
            with open(path, 'wb') as file:
                file.write(content.encode())
            return f"File written to {path}"
        else:
            with open(path, 'w') as file:
                file.write(content)
            return f"File written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# TODO: Integrate this with plane scraper to locally index commonly used documentation and web pages.
# @mcp.tool()
async def web_page_scrapper(url: str) -> str:
    """Scrapes a webpage and return the content in markdown. FYI, bool here refers to python boolean(i.e True or False)."""
    print(f"Executing resource 'web_page_scrapper' with url={url}")
    try:
        # config = CrawlerRunConfig(wait_until="js:() => window.loaded === true")
        res = await crawler.arun(url=url)#),config=config)
        # if Index:
        #     if Index_name is None:
        #         return "Index name is required."
        #     try:
        #         # Create a text splitter to split the content into chunks
        #         # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #         text_splitter = MarkdownHeaderTextSplitter([
        #                                                     ("#", "Header 1"),
        #                                                     ("##", "Header 2"),
        #                                                     ("###", "Header 3"),
        #                                                 ], strip_headers=False)
                
        #         # Split the content into chunks
        #         docs = text_splitter.create_documents([res_mark])
                
        #         # Create a Chroma vector store and persist it to disk
        #         vectordb = Chroma.from_documents(docs, embed_model, collection_name=Index_name, persist_directory="Indexes")
        #         vectordb.persist()
                
        #         return f"Indexed content saved to {Index_name}"
        #     except Exception as e:
        #         print(f"Error indexing content: {e}")
        #         return str(e)
        return str(res.cleaned_html) #+ "\n\n\n\n" + "All available links in the website:\n" + str(res.links)
    except Exception as e:
        print(f"Error scraping page: {e}")
        return str(e)

@mcp.tool()
async def deep_research(query: str, max_results: int = 5, depth: int = 1) -> str:
    """Perform a deep research online, on a query and return the results."""
    print(f"Executing resource 'deep_research' with query={query}, max_results={max_results}, depth={depth}")
    try:
        agent = WebResearchAgent(crawler,max_iterations=max_results, max_scrape_urls_per_iteration=depth, logger=log)
        results = await agent.run_research(query)
        return str(results)
    except Exception as e:
        print(f"Error performing deep research: {e}")
        return str(e)


@mcp.tool()
def get_all_vector_indexes() -> str:
    """Get all the vector embedding indexes in the current directory."""
    import os
    indexes = []
    for root, dirs, files in os.walk(os.getcwd()):
        for dir in dirs:
            if dir.startswith("Indexs"):
                indexes.append(dir)
    return "\n".join(indexes) if indexes else "No indexes found."

@mcp.tool()
def search_via_index(query: str, index_name: str) -> str:
    """Search a query via a vector embedding index."""

    # Load the vector store from disk
    vectordb = Chroma(persist_directory="Indexes", collection_name=index_name, embedding_function=embed_model)
    
    # Perform the search
    results = vectordb.similarity_search(query, k=5)
    
    # Format the results as a string
    return "\n".join([str(result.page_content) for result in results]) if results else "No results found."



if __name__ == "__main__":

    print("Starting MCP Server via stdio...")
    mcp.run(transport="stdio") # This starts the MCP server loop    
    print("MCP Demo Server stopped.")