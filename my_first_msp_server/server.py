from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Sample Server")

# Define a tool to get weather information
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    # mockup tool
    return f"Weather in {location}: Sunny, 72°F"

# Define a calculator tool
@mcp.tool()
def calculate(expression: str) -> float:
    """Calculate the result of a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

# Define a currency converter tool
@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert amount from one currency to another."""
    # This is a simplified example - in reality, you'd want to use real exchange rates
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0,
        "INR": 83.0
    }
    if from_currency not in rates or to_currency not in rates:
        return f"Unsupported currency pair: {from_currency} to {to_currency}"

    converted = amount * (rates[to_currency] / rates[from_currency])
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


# Run the server
if __name__ == "__main__":
    mcp.run()
    # mcp.run(transport="sse", host="127.0.0.1", port=8001)