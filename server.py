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
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print(f"Executing resource 'get_greeting' with name={name}") # Add logging/print
    greeting = f"Hello, ma dude {name}!"
    print(f"Result: {greeting}")
    return greeting

# --- Missing Piece ---
# You need to tell the server to start running and listen for connections (via stdio in this case)
if __name__ == "__main__":
    print("Starting MCP Demo Server via stdio...")
    mcp.run(transport="stdio") # This starts the MCP server loop
    print("MCP Demo Server stopped.")