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

    def __init__(self, api_key: str, model: str = "gpt") -> None:
        self.api_key: str = api_key
        self.model = model
        if model == "gpt":
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version=os.environ["OPENAI_API_VERSION"],
                temperature=1,
                max_tokens=2**9,
                timeout=None,
                max_retries=2,
                api_key=api_key,
                # top_p=0.85,
            )
        elif model == "gemini":
                api_key = os.environ.get("GKEY",api_key)
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="Gemini-2.0-Flash",
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
            pattern = r"```json_tool\n(.*?)\n```"
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                llm_response = match.group(1).strip()
            logging.info(f"json {llm_response}")
            tool_call = json.loads(llm_response)
                # logging.info(f"Executing tool: {tool_call['tool']}")
                # logging.info(f"With arguments: {tool_call['arguments']}")

            for server in self.servers:
                tools = await server.list_tools()
                resource = await server.list_resource_templates()
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
            raise llm_response

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
            #     "{\n"
            #     '    "tool": "tool-name",\n'
            #     '    "arguments": {\n'
            #     '        "argument-name": "value"\n'
            #     "    }\n"
            #     "}\n\n"
            #     "\n"
            #     "for resources-template:\n"
            #     "{\n"
            #     '    "Resource": "resource-name"\n'
            #     '    "uri": "<protocol>://<data-to-be-sent>"\n'
            #     "    }\n"
            #     "}\n\n"
            #     "Also remember, protocol can be anything, from greeting:// to http://\n\n"
            #     "After receiving a tool's\\resour response:\n"
            #     "1. Transform the raw data into a natural, conversational response\n"
            #     "2. Keep responses concise but informative\n"
            #     "3. Focus on the most relevant information\n"
            #     "4. Use appropriate context from the user's question\n"
            #     "5. Avoid simply repeating the raw data\n\n"
            #     "Please use only the tools that are explicitly defined above."
            # )
            # logging.info(resources_description)
            system_message = f"""
You are a helpful assistant with access to tools and resources. Your primary goal is to assist the user by answering their questions or completing their requests.  You MUST prioritize using available tools and resources before attempting to answer directly.

Tools & Resources:

Tools:
{tools_description}

Resources:
{resources_description}
"""+r"""
Workflow:

1.  Analyze User Request: Carefully understand the user's request and identify any actions required to fulfill it.
2.  Resource/Tool Availability Check: BEFORE attempting to answer directly, meticulously examine the `tools` and `resources` descriptions provided.
3.  Resource Prioritization: If a `resource` is available that can directly address the user's need, YOU MUST USE IT FIRST. Resources are preferred over tools.
4.  Tool Usage (If No Resource Available): If no suitable `resource` exists, then check if a `tool` can perform the necessary action.
5.  Direct Response (Only as Last Resort):  Only respond directly if ABSOLUTELY NO SUITABLE `TOOL` OR `RESOURCE` is available.

Response Format (when using a Tool or Resource):

You MUST respond with the exact JSON object format below, and NOTHING ELSE, when utilizing a `tool` or `resource`. This is critical for proper system functionality.

For Tools:

```json_tool
{
    "tool": "tool-name",
    "arguments": {
        "argument-name": "value"
    }
}
```

For Resources:

```json_tool
{
    "Resource": "resource-name",
    "uri": "<protocol>://<data-to-be-sent>"
}
```

Important Notes about the JSON Format:

*   The JSON object MUST be well-formed and valid.
*   The `"tool"` or `"Resource"` field must exactly match the name listed in the `tools_description` or `resources_description`.
*   The `"arguments"` field in the `tool` object MUST include all required arguments as defined in the `tools_description` and their corresponding values.
*   The `"uri"` field in the `Resource` object is a Uniform Resource Identifier. The `<protocol>` can be anything appropriate (e.g., `http`, `https`, `greeting`, `data`, etc.), and `<data-to-be-sent>` represents the data you need to send to the resource.

After Receiving a Tool/Resource Response (and ONLY AFTER receiving a response from the tool or resource) from the user:
1.  Process the raw data returned by the tool or resource from a user and convert it into a natural, conversational response that is easy for the user to understand.
2.  Keep your responses concise but informative.
3.  Use appropriate context from the user's question(i.e question that led to tool calling) to frame your response.
4.  DO NOT SIMPLY REPEAT THE RAW JSON DATA ONCE YOU GET THE TOOL RESPONSE IN THE FORM OF "Tool execution result: meta=... content=[TextContent(type='...', text='...', annotations=...)] isError=...". Instead, focus on the most relevant information and present it in a user-friendly manner.

Conversation example

user: "Can you give me a list of all files in the C: drive?"
assistant:```json_tool
{
    "Resource": "resource-name",
    "uri": "<protocol>://<data-to-be-sent>"
}
```
user: Resource retrieval result: ...
AI: "Here is the list of files in the C: drive: ..."
"""

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

                    llm_response = await self.llm_client.get_response(messages)
                    # logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)
                    logging.info("\nAssistant_with_tools: %s", result)
                    if result != llm_response:
                        # messages[-1] = AIMessage(content=llm_response)
                        messages.append(HumanMessage(content=result))
                        # messages.append(ToolMessage(content=result,))
                        final_response = await self.llm_client.get_response(messages)
                        pattern = r"```.*\n(.*?)\n```"
                        match = re.search(pattern, final_response, re.DOTALL)
                        # logging.info("\nFinal response after: %s", final_response)
                        # if match:
                        #     final_response = match.group(1).strip()
                        logging.info("\nFinal response: %s", final_response)
                        messages.append(
                            AIMessage(content=final_response)
                        )

                    else:
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
