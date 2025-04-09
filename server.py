from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("DemoServer") # Good practice to give it a descriptive name

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"Executing tool 'add' with a={a}, b={b}") # Add logging/print for debugging
    result = a + b
    print(f"Result: {result}")
    return result

# Add a dynamic greeting resource
# @mcp.resource("greeting://{name}", name="greeter", description="Any type of greeting to be done to the user, use this resource, by replacing the {name} placeholder for users name in the URI and returning the Json. (i.e greeting://{name})", mime_type="text/plain")
# def get_greeting(name: str) -> str:
#     """Get a personalized greeting"""
#     print(f"Executing resource 'get_greeting' with name={name}") # Add logging/print
#     greeting = f"Hello, ma dude {name}!"
#     print(f"Result: {greeting}")
#     return greeting

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
        for root, dirs, files in os.walk(partition+":"):
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


# write a mcp resource template that takes the path of the file and returns the content of the file as a string
@mcp.resource("file://{path}", name="file_reader", description="Read the content of a file", mime_type="text/plain")
def read_file(path: str) -> str:
    """Read the content of a file"""
    print(f"Executing resource 'read_file' with path={path}")
    try:
        with open(path, 'r') as file:
            content = file.read()
        print(f"Result: {content}")
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return str(e)

# --- Missing Piece ---
# You need to tell the server to start running and listen for connections (via stdio in this case)
if __name__ == "__main__":
    print("Starting MCP Demo Server via stdio...")
    mcp.run(transport="stdio") # This starts the MCP server loop
    print("MCP Demo Server stopped.")