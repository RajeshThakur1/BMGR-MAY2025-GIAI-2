import asyncio
from mcp import Client, ClientSession
from mcp.client.sse import sse_client

async def main():

    # connect with the server using SSE

    async with sse_client("http://127.0.0.1:8000/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            # Initilizing the Connection

            await session.initialize()

            # List available tools

            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"- {tool.name} ({tool.description})")

                print("\n\n\n\n-----------------------------------\n\n\n\n")

            # Call a tool

            result = await session.call_tool("calculate", arguments={"expression": "2+3"})
            print(f"Result: {result}")
            
if __name__ == "__main__":
    asyncio.run(main())

# you can run the client either of the way 
# uv run mcp_client_sse.py

# or 
# python mcp_client_sse.py