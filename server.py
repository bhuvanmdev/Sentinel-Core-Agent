from mcp.server.fastmcp import FastMCP
import re
import requests


mcp = FastMCP("DemoServer") # Good practice to give it a descriptive name

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"Executing tool 'add' with a={a}, b={b}") # Add logging/print for debugging
    result = a + b
    print(f"Result: {result}")
    return result


#get all files as a tree of c or d partition and tree as a string
@mcp.resource("tree://{partition}/{depth}", name="partition_tree", description="Get all files as a tree of any given partition(c,d etc...) and tree as a string based on the depths", mime_type="text/plain")
def get_tree(partition: str,depth: int) -> str:
    """Get all files as a tree of c or d partition and tree as a string"""
    print(f"Executing resource 'get_tree' with path={partition}")
    try:
        import os
        tree = []
        if not os.path.exists(partition+":"):
            return f"Partition {partition} does not exist."
        for root, dirs, files in os.walk(partition+":\\"):
            level = root.replace(partition, '').count(os.sep)
            if level > depth:
                continue
            indent = ' ' * 4 * (level)
            tree.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                tree.append(f"{subindent}{f}")
        result = "\n".join(tree)
        print(f"Result: {result}")
        return result
    except Exception as e:
        print(f"Error getting tree: {e}")
        return str(e)

@mcp.tool()
def cur_time() -> str:
    """Get the current time"""
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Current Time: {current_time}")
    return current_time

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
def aisearch(query: str) -> str:
    """Search any query and obtain a response from another AI agent knowledgeable about the query."""
    hash_,id_ = braveai(query,temp=True)
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


if __name__ == "__main__":
    print("Starting MCP Demo Server via stdio...")
    mcp.run(transport="stdio") # This starts the MCP server loop
    print("MCP Demo Server stopped.")