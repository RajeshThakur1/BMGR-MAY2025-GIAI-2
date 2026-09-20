import asyncio
from pdb import run
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    ### Defining my server paramters


    # When you server is running in stdion mode it is not necessary to server up and running in advance you client will automatically run the server.py file and connect to it.
    server_parameters = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    ## connect to the server

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()     # initilize the connections

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
# uv run mcp_clint.py

# or 
# python mcp_clint.py