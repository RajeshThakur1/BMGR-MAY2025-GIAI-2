import asyncio
import json
import litellm
import warnings

from contextlib import AsyncExitStack
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

session: ClientSession = None

exit_stack = AsyncExitStack()
model = "gpt-4o"


async def connect_to_server(server_script_path: str = "server.py"):
    """Connect to MCP server and list available tools."""

    global session, exit_stack
    server_parameters = StdioServerParameters(command="python", args=[server_script_path])

    stdio_trannsport = await exit_stack.enter_async_context(stdio_client(server_parameters))

    stdio, write = stdio_trannsport
    session = ClientSession(stdio, write)

    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()


### Next, tool discovery 

async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Retrieve the MCP tools in Litellm/OpenAI function format."""
        
    global session
    
    tools_result = await session.list_tools()

    print("Connected to server with tools:")
    for tool in tools_result.tools:
        print(f"  • {tool.name}: {tool.description}")

    formatted = []
    for tool in tools_result.tools:
        formatted.append({"type": "function",
                          "function": {"name": tool.name,
                                       "description": tool.description,
                                       "parameters": tool.input_schema}
                        })
    return formatted


### Process queries

async def process_query(query: str) -> str:

    global session, model

    tools = await get_mcp_tools()

    first_response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools = tools,
        tool_choice = "auto",

    )

    # Extract the assistant message

    assistant_message = first_response.choices[0].message
    messages: List[Dict[str, Any]] = [{"role": "user", "content": query}, assistant_message]

    ### Check if litellm want to call any tool or not

    calls = assistant_message.tool_calls or assistant_message.function_call

    if calls:
        calls_list = calls if isinstance(calls, list) else [calls]
        for call in calls_list:
            if call.function: ## newer LiteLLm structure

                tool_name = call.function.name
                raw_args = call.function.arguments

            else: 
                tool_name = call.name
                raw_args = call.arguments
                

            try:
                parsed_args = json.loads(raw_args)

            except Exception as e:
                parsed_args = raw_args
            
            ### Checjk if the LLM requires any tool, if yes then parse the tool name and arguments

            print(f"\n Assistant requires tool: {tool_name}({parsed_args})")
            permission = input("Allow tool call? (y/n): ")
            if permission.lower() == "y":
                result = await session.call_tool(tool_name, arguments=parsed_args)
                tool_output = result.content[0].text
                print(f"→ {tool_name} returned: {tool_output}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id if hasattr(call, "id") else None,
                    "content": tool_output,
                })

            else:
                denied_msg = f"[Tool ]'{tool_name}' denied"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id if hasattr(call, "id") else None,
                    "content": denied_msg,
                })

                print(f" -> Skipped {tool_name}")

                ### Take explicit user permissions

                ### Second pass: ask LiteLLM for a reply now that we have the tool output
                second_response = await litellm.acompletion(
                    model=model,
                    message=messages,
                    tools=tools,
                    tool_choice="none"
                )
                return second_response.choices[0].message.content.strip()
    else:
        return assistant_message.content.strip()  ## LiteLLM answered without needing any tools



### Now that we have our central logic in place let's give a final touch by ensuring that the session closed properrly when script exist
async def cleanup():
    """close the MCP client session cleanly"""
    global exit_stack
    await exit_stack.aclose()
    print("\n MCP client session closed")


async def main():
    await connect_to_server("/Users/rajesh/Desktop/rajesh/Archive/teaching/BMGR_JUNE/mcp_tutorials/mcp_day_1/src/mcp_day_1/server.py")
    query = """what is 90 multiplied by 68.6.
            Also tell the weather in Lucknow.
            use MCP tools for both.
            """
    print(f"\nQuery: {query}")
    response = await process_query(query)
    print(f"\n Final Response: {response}")
    await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
        

            
