[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/bhuvanmdev-sentinel-core-agent-badge.png)](https://mseep.ai/app/bhuvanmdev-sentinel-core-agent)

# This section was written completely by using the SentinelCore agent and its tools(prompt->get the details from internet->write it to a file) via gemini 2.0 flash.

```
USER-PROMPT:-Now first search for a github account named bhuvanmdev and scrape his fontpage and search for a repo that has something to do with MCP application. Then go to that repository and and scrape the client.py and server.py file contents and create a neat summary of it and write it to readme.md file in the current dir.
``` 

# Sentinel Core Agent Summary

This document provides a summary of the `client.py` and `server.py` files from this repo.

## Client.py

The `client.py` file manages the interaction between a user, an LLM (Language Model), and various tools. It initializes and orchestrates a chat session where the LLM can use tools to answer user queries. The client handles server connections, tool execution with retries, and communication with the LLM provider.

**Key Components:**

*   **Configuration:** Loads environment variables (including the LLM API key) and server configurations from a JSON file.
*   **Server:** Manages connections to MCP (Microservice Communication Protocol) servers, lists available tools, executes tools with a retry mechanism, and cleans up resources.
*   **Tool:** Represents a tool with its properties (name, description, input schema) and provides a method to format the tool information for the LLM.
*   **LLMClient:** Manages communication with the LLM, using either Azure OpenAI or Google Gemini models.
*   **ChatSession:** Orchestrates the interaction between the user, LLM, and tools. It processes LLM responses, executes tools if needed, and handles the main chat session loop.

The client sets up a system message for the LLM, providing it with available tools and instructions on how to use them. It then enters a loop where it takes user input, sends it to the LLM, and processes the LLM's response. If the response contains a tool call, the client executes the tool and sends the result back to the LLM. This process continues until the user exits the chat session.

## Server.py

The `server.py` file implements an MCP server with various tools, including file system operations, web scraping, and AI-powered search. It uses the `fastmcp` library to create and run the server.

**Key Tools:**

*   **is_file_folder_present:** Checks if a file or folder exists in the file system.
*   **cur_datetimetime:** Returns the current date and time.
*   **browser_ai_search:** Searches the web using an AI agent (Brave Search) and returns the response.
*   **read_file:** Reads the content of a file.
*   **write_file:** Writes content to a file (both text and binary).
*   **web_page_scrapper:** Scrapes a webpage and returns the content in markdown format. It can also index the content for vector-based search.
*   **get_all_vector_indexes:** Retrieves all vector embedding indexes in the current directory.
*   **search_via_index:** Searches a query via a vector embedding index.

The server also initializes an asynchronous web crawler (`crawl4ai`) and sets up an embedding model for vectorizing content. The server's lifespan is managed using an asynchronous context manager to ensure proper startup and shutdown of the crawler.

The server starts the MCP server loop using `mcp.run(transport="stdio")`, allowing it to receive and process tool calls from the client.
