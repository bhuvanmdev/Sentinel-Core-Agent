import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import re
import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("GPT")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def GPT(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("GPT not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))###

        return tools
    
    async def list_resource_templates(self) -> list[Any]:
        """List available resources from the server.

        Returns:
            A list of available resources.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        resources_response = await self.session.list_resource_templates()
        # logging.info(resources_response)
        resources = []

        for item in resources_response:
            # logging.info(item)
            if isinstance(item, tuple) and item[0] == "resourceTemplates":
                for resource in item[1]:
                    resources.append(Resource_template(resource.name, resource.description, resource.uriTemplate, resource.mimeType))

        return resources

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise
    async def get_resource(
        self, resource_name: str, retries: int = 2, delay: float = 1.0
    ) -> Any:
        """Get a resource with retry mechanism.

        Args:
            resource_name: Name of the resource to get.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Resource result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If resource retrieval fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Getting {resource_name}...")
                result = await self.session.read_resource(resource_name)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error getting resource: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose() # exiting the async context manager
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
            # args.desc.append("- id: "
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

class Resource_template:
    """Represents a resource with its properties and formatting."""

    def __init__(self, name: str, description: str, uri: str, mimetype: str) -> None:
        self.name: str = name
        self.description: str = description
        self.uri: str = uri
        self.mimetype: str = mimetype
    
    def format_for_llm(self) -> str:
        """Format resource information for LLM.

        Returns:
            A formatted string describing the resource.
        """
        logging.info(f"Resource: {self.name}, Description: {self.description}, URI: {self.uri}, MIME Type: {self.mimetype}")
        return f"""
Resource: {self.name}
Description: {self.description}
URI: {self.uri}
MIME Type: {self.mimetype}
"""

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini") -> None:
        self.api_key: str = api_key
        self.model = model
        if model == "gpt":
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version=os.environ["OPENAI_API_VERSION"],
                temperature=1,
                # max_tokens=2**9,
                timeout=None,
                max_retries=2,
                api_key=api_key,
                # top_p=0.85,
            )
        elif model == "gemini":
                api_key = os.environ.get("GKEY",api_key)
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=api_key,
                    temperature=0.5,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    # top_p=0.85,
            )

    async def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        """
        res = await self.llm.ainvoke(messages)
        return res.content





class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            # pattern to match the JSON object given by the LLM in ```json\n<json-object>\n``` format
            # pattern = r"```json_tool\n(.*?)\n```"
            # match = re.search(pattern, llm_response, re.DOTALL)
            # if match:
            #     llm_response = match.group(1).strip()
            # logging.info(f"json {llm_response}")

                # logging.info(f"Executing tool: {tool_call['tool']}")
                # logging.info(f"With arguments: {tool_call['arguments']}")

            for server in self.servers:
                tools = await server.list_tools()
                resource = await server.list_resource_templates()
                tool_call = json.loads(llm_response.strip())
                if ("tool" in tool_call and "arguments" in tool_call) and any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        result = await server.execute_tool(
                            tool_call["tool"], tool_call["arguments"]
                        )

                        if isinstance(result, dict) and "progress" in result:
                            progress = result["progress"]
                            total = result["total"]
                            percentage = (progress / total) * 100
                            logging.info(
                                f"Progress: {progress}/{total} "
                                f"({percentage:.1f}%)"
                            )

                        return f"Tool execution result: {result}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logging.error(error_msg)
                        return error_msg
                elif ("Resource" in tool_call and "uri" in tool_call) and any(resource.name == tool_call["Resource"] for resource in resource):
                    try:
                        result = await server.get_resource(
                            tool_call["uri"]
                        )
                        return f"Resource retrieval result: {result}"
                    except Exception as e:
                        error_msg = f"Error retrieving resource: {str(e)}"
                        logging.error(error_msg)
                        return error_msg
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return f"Error decoding JSON response: {llm_response}"  

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            all_resources = []

            for server in self.servers:
                tools = await server.list_tools()
                resources = await server.list_resource_templates()
                # logging.info(all_resources)
                all_resources.extend(resources)
                all_tools.extend(tools)
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            resources_description = "\n".join([resource.format_for_llm() for resource in all_resources])


            # system_message = (
            #     "You are a helpful assistant with access to these tools and resources:\n\n"
            #     "tools:\n"
            #     f"{tools_description}\n\n"
            #     "resources:\n"
            #     f"{resources_description}\n\n"
            #     "Choose the appropriate tool\\resource based on the user's question. "
            #     "If no tool\\resource is needed, reply directly.\n\n"
            #     "IMPORTANT: When you need to use a tool\\resourcse, you must ONLY respond with "
            #     "the exact JSON object format below, nothing else:\n"
            #     "for tools:\n"
            #     "```json_tool\n"
            #     "{\n"
            #     '    "tool": "tool-name",\n'
            #     '    "arguments": {\n'
            #     '        "argument-name": "value"\n'
            #     "    }\n"
            #     "}```\n\n"
            #     "\n"
            #     # "for resources-template:\n"
            #     # "{\n"
            #     # '    "Resource": "resource-name"\n'
            #     # '    "uri": "<protocol>://<data-to-be-sent>"\n'
            #     # "    }\n"
            #     # "}"
            #     # "```\n\n"
            #     # "Also remember, protocol can be anything, from greeting:// to http://\n\n"
            #     # "After receiving a tool's\\resour response:\n"
            #     "1. Transform the raw data into a natural, conversational response\n"
            #     "2. Keep responses concise but informative\n"
            #     "3. Focus on the most relevant information\n"
            #     "4. Use appropriate context from the user's question\n"
            #     "5. Avoid simply repeating the raw data\n\n"
            #     "6. When Using tools ALWAYS ENCLOSE THE JSON OBJECT IN ```json_tool\n<json-object>\n``` format\n\n."
            #     "Please use only the tools that are explicitly defined above."
            # )
            # logging.info(resources_description)
# """Workflow:

# 1.  Analyze User Request: Carefully understand the user's request and identify any actions required to fulfill it.
# 2.  Resource/Tool Availability Check: BEFORE attempting to answer directly, meticulously examine the `tools` and `resources` descriptions provided.
# 3.  Resource Prioritization: If a `resource` is available that can directly address the user's need, YOU MUST USE IT FIRST. Resources are preferred over tools.
# 4.  Tool Usage (If No Resource Available): If no suitable `resource` exists, then check if a `tool` can perform the necessary action.
# 5.  Direct Response (Only as Last Resort):  Only respond directly if ABSOLUTELY NO SUITABLE `TOOL` OR `RESOURCE` is available.


# """

            system_message = f"""You are a helpful AI assistant. Your primary goal is to assist users by completing their requests accurately and efficiently.

**Core Principle: Tool-First Approach**
You MUST prioritize using the available tools to fulfill user requests. Only attempt to answer directly if a request cannot be addressed by any of the provided tools.

**Available Tools:**
{tools_description}"""+r"""

**Tool Usage Format:**
When you need to use a tool, you MUST output a JSON object in the following exact format. Do NOT add any text before or after this JSON block:

```json_tool
{
    "tool": "tool-name",
    "arguments": {
        "argument-name": "value"
    }
}
```

-The JSON object MUST be perfectly valid.
-"tool" MUST be the exact name of a tool listed in Available Tools.
-"arguments" MUST contain all required arguments for that tool and their corresponding values.

Execution Process:

1.Understand and Plan:
-Analyze the user's request.
-If tools are applicable, create a step-by-step plan outlining which tool(s) to use and in what sequence. Be as deatiled on what you will do in this planning process step by step.
-If no tools are suitable, prepare a direct answer.

2.Execute Tool(s) Sequentially (If Applicable):
-Begin executing your plan, starting with the first tool call.
-Output the json_tool block for the first tool.
-WAIT for the user to provide the tool execution result (which will start with Tool execution result:).
-DO NOT provide any conversational text, summaries, or confirmations between tool calls.
-If your plan requires more tools, use the result from the previous step (if necessary) to formulate the arguments for the next tool call. Output the json_tool block for the next tool.
-Repeat this process (output json_tool -> wait for user result) for every tool in your plan.

3.Handle Tool Responses:
-Only after all planned tool calls have been successfully executed and you have received the final Tool execution result: from the user:
-Synthesize the information gathered from all tool responses.
-Formulate a single, comprehensive, and user-friendly response that directly addresses the user's original request.
-Do NOT just repeat the raw data from the tool results. Process and present the information clearly and concisely.

4.Error Handling:
-If a tool execution results in an error (isError=true in the result):
-Silently RETRY the exact same tool call up to 2 more times.
-If the error persists after all retries, STOP the execution plan.
-Inform the user that you encountered a persistent error with a specific tool (tool-name) and could not complete the request. Provide any relevant error details if available in the tool result. Do not attempt to provide a partial answer based on successful preceding steps.

Examples:

1) Single Tool Conversation Example:

User: What's the current weather in London?
Assistant: Sure! to get the current weather in London, I will use the following tool:
1. will use the `get_weather` tool to fetch the current weather data.
2. I will then process the weather data and provide you with a summary.
Is this fine or is there any changes to make here?
User: LGTM
Assistant:
```json_tool
{
    "tool": "get_weather",
    "arguments": {
        "location": "London"
    }
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

User: Tool execution result: meta=... content=[TextContent(type='text/plain', text='The current weather in London is 15°C and cloudy.')] isError=false
Assistant: The current weather in London is 15°C and cloudy.

**2) Multi-Tool Conversation Example:**

User: Please find the email address for 'John Doe' in my contacts, and then draft an email to him asking about the project status.
Assistant: Sure! to find the email address for 'John Doe' and draft an email, I will use the following tools:
1. I will first use the `search_contacts` tool to find John Doe's email address.
2. Once I have the email address, I will use the `draft_email` tool to create an email asking about the project status. 
3. I will then provide you with the drafted email for your review before sending it.

Is this fine or is there any changes to make here?
User: LGTM
Assistant:
```jaon_tool
{
    "tool": "search_contacts",
    "arguments": {
        "name": "John Doe"
    }
}
```
User: Tool execution result: meta=... content=[TextContent(type='application/json', text='{"email": "john.doe@example.com", "phone": "123-456-7890"}')] isError=false
Assistant:
```json_tool
{
    "tool": "draft_email",
    "arguments": {
        "recipient": "john.doe@example.com",
        "subject": "Project Status Query",
        "body": "Hi John,\n\nCould you please provide an update on the current project status?\n\nThanks,\n[Your Name]"
    }
}
```
User: Tool execution result: meta=... content=[TextContent(type='text/plain', text='Email draft created successfully.')] isError=false
Assistant: I found John Doe's email address (john.doe@example.com) and have drafted an email to him asking about the project status. Please review it before sending.

Follow these instructions carefully to ensure smooth and effective interaction."""

            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
            # messages = [AIMessage(content=system_message)]
            messages = [SystemMessage(content=system_message)]
            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append(HumanMessage(content=user_input))
                    
                    # Initial LLM response
                    llm_response = await self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)
                    
                    # Check if the response contains a tool call
                    tool_pattern = r".*```json_tool\n(.*?)\n```.*"
                    tool_match = re.match(tool_pattern, llm_response, re.DOTALL)

                    if tool_match:
                        # print(tool_match.group(1))
                        # input()
                        # Process tool execution
                        tool_result = await self.process_llm_response(tool_match.group(1).strip('\n'))
                        logging.info("\nTool execution result: %s", tool_result)
                        
                        # Add the tool result to messages
                        messages.append(HumanMessage(content=tool_result))
                        
                        # Continue tool chain if needed
                        while True:
                            next_response = await self.llm_client.get_response(messages)
                            logging.info("\nAssistant follow-up: %s", next_response)
                            
                            # Check if there's another tool call
                            next_match = re.match(tool_pattern, next_response, re.DOTALL)
                            if next_match:
                                # Process another tool execution
                                next_tool_result = await self.process_llm_response(next_match.group(1).strip('\n'))
                                logging.info("\nTool execution result: %s", next_tool_result)
                                messages.append(HumanMessage(content=next_tool_result)) 
                            else:
                                # No more tool calls, add final response to conversation
                                messages.append(AIMessage(content=next_response))
                                break
                    else:
                        # No tool call, just add response to conversation
                        messages.append(AIMessage(content=llm_response))

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.GPT)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
