import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Use the stdio server script (NOT the root server.py, which is configured for SSE)
SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


async def main():
    # Stdio client *starts* the server as a child process and talks over stdin/stdout.
    # You should not already have the server running in another terminal.
    # Use sys.executable so the child uses the same conda/venv as this client.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            result = await session.call_tool(
                "calculate", arguments={"expression": "2 + 3"}
            )
            print(f"2 + 3 = {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
