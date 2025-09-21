from mcp.server.fastmcp import FastMCP
from typing import List

# Mock leave database; in production, replace with actual DB/query logic
employee_leaves = {
    "E001": {"balance": 18, "history": ["2024-12-25", "2025-01-01"]},
    "E002": {"balance": 20, "history": []}
}

## Create MCP Server 
mcp = FastMCP("LeaveManager")

# Tool: Get leave balance for employee
@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Check remaining leave days for an employee"""
    data = employee_leaves.get(employee_id)
    if data:
        return f"{employee_id} has {data['balance']} leave days remaining."
    return "Employee ID not found."


# Tool: Apply for leave
@mcp.tool()
def apply_leave(employee_id: str, leave_dates: List[str]) -> str:
    """Apply leave for specific dates"""
    if employee_id not in employee_leaves:
        return "Employee ID not found."
    requested_days = len(leave_dates)
    available_balance = employee_leaves[employee_id]["balance"]
    if available_balance < requested_days:
        return f"Insufficient leave balance. You requested {requested_days} day(s) but have only {available_balance}."
    # Deduct balance and update history
    employee_leaves[employee_id]["balance"] -= requested_days
    employee_leaves[employee_id]["history"].extend(leave_dates)
    return f"Leave applied for {requested_days} day(s). Remaining balance: {employee_leaves[employee_id]['balance']}."


# Tool: Get leave history for employee
@mcp.tool()
def get_leave_history(employee_id: str) -> str:
    """Get list of leave dates for employee"""
    data = employee_leaves.get(employee_id)
    if data:
        history = ', '.join(data['history']) if data['history'] else "No leaves taken."
        return f"Leave history for {employee_id}: {history}"
    return "Employee ID not found."

@mcp.resource("greeting://{name}")
def get_greeting(name:str) -> str:
    "Get a personalized greeting"
    return f"Hello, {name}! How can I assist with your Leave management today"


# Prompt: Custom greeting prompt generator
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run()