from mcp.server.fastmcp import FastMCP
import re
import requests
from typing import Optional, Union
import asyncio
from crawl4ai import *
import sys
sys.stdout.reconfigure(encoding='utf-8')

mcp = FastMCP("Python Tools server") # Good practice to give it a descriptive name


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
        inp = "From this point onwards answer all the questions asked only in markup format. This includes bold, newline and links. Ignore all other stylings and no json formatting allowed.\n\n   " + query 
        ini = ''.join([url_encoding.get(x, x) for x in ini])
        url = r"https://search.brave.com/api/chatllm/conversation?key=%7B%22query%22%3A%22"+ini+r"%22%2C%22country%22%3A%22us%22%2C%22language%22%3A%22en%22%2C%22safesearch%22%3A%22moderate%22%2C%22results_hash%22%3A%22"+hash_+r"%22%7D&conversation="+id_+r"&index="+str(ind)+r"&followup="+inp
        response = requests.get(url, headers=headers)
        output = ""
        for x in response.text.split('"\n"'):
            output += x.strip('"')
        return output.replace('\\n','\n').replace('\\','')

@mcp.tool()
def browser_ai_search(query: str) -> str:
    """Search any query and obtain a response from another AI agent connected to the internet."""
    hash_,id_ = braveai(".",temp=True)
    return braveai(query,temp=False,hash_=hash_,id_=id_)


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


async def web_page_scrapper(url: str) -> str:
    """Scrapes a webpage and return the content in markdown."""
    print(f"Executing resource 'web_page_scrapper' with url={url}")
    try:
        async with AsyncWebCrawler(
            launch_options={
                "headless": True,
                "args": ["--no-sandbox", "--disable-setuid-sandbox"]
            },
            thread_safe=True
        ) as crawler:
            result = await crawler.arun(
                url=url,
            )

        return f"The scrapped content is given below \n{str(result.markdown,encoding='utf-8')}"
    except Exception as e:
        print(f"Error scraping page: {e}")
        return str(e)


if __name__ == "__main__":
    print("Starting MCP Server via stdio...")
    mcp.run(transport="stdio") # This starts the MCP server loop
    print("MCP Demo Server stopped.")