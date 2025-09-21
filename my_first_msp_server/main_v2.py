from mcp.server.fastmcp import FastMCP
from typing import List, Optional
import sqlite3
import os
from datetime import datetime, date
import json

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hr_leave_management.db")


def init_database():
    """Initialize the SQLite database with required tables"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                department TEXT,
                position TEXT,
                hire_date DATE,
                leave_balance INTEGER DEFAULT 20,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create leave_records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leave_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                leave_date DATE NOT NULL,
                leave_type TEXT DEFAULT 'Annual',
                status TEXT DEFAULT 'Approved',
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Create leave_balance_history table for tracking balance changes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leave_balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                old_balance INTEGER,
                new_balance INTEGER,
                change_reason TEXT,
                changed_by TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
    
        conn.commit()
        conn.close()
        print(f"Database initialized successfully at: {DATABASE_PATH}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        print(f"Database path: {DATABASE_PATH}")
        raise

init_database()

# Create MCP server
mcp = FastMCP("HRLeaveManager")


# Helper functions
def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DATABASE_PATH)


def employee_exists(employee_id: str) -> bool:
    """Check if employee exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# HR Management Tools

@mcp.tool()
def add_employee(employee_id: str, name: str, email: str, department: str, position: str, initial_leave_balance: int = 20) -> str:
    """Add a new employee to the system"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if employee already exists
        if employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} already exists."
        
        # Insert new employee
        cursor.execute('''
            INSERT INTO employees (employee_id, name, email, department, position, leave_balance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (employee_id, name, email, department, position, initial_leave_balance))
        
        # Log balance change
        cursor.execute('''
            INSERT INTO leave_balance_history (employee_id, old_balance, new_balance, change_reason, changed_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (employee_id, 0, initial_leave_balance, "Initial balance", "HR System"))
        
        conn.commit()
        conn.close()
        
        return f"Employee {name} (ID: {employee_id}) added successfully with {initial_leave_balance} leave days."
    
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            return f"Email {email} is already registered to another employee."
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error adding employee: {str(e)}"
    

@mcp.tool()
def remove_employee(employee_id: str) -> str:
    """Remove an employee from the system"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        # Get employee name before deletion
        cursor.execute("SELECT name FROM employees WHERE employee_id = ?", (employee_id,))
        result = cursor.fetchone()
        employee_name = result[0] if result else "Unknown"
        
        # Delete related records first (due to foreign key constraints)
        cursor.execute("DELETE FROM leave_records WHERE employee_id = ?", (employee_id,))
        cursor.execute("DELETE FROM leave_balance_history WHERE employee_id = ?", (employee_id,))
        cursor.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        
        conn.commit()
        conn.close()
        
        return f"Employee {employee_name} (ID: {employee_id}) and all related records removed successfully."
    
    except Exception as e:
        return f"Error removing employee: {str(e)}"



@mcp.tool()
def update_leave_balance(employee_id: str, new_balance: int, reason: str = "HR adjustment") -> str:
    """Update an employee's leave balance"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        # Get current balance
        cursor.execute("SELECT leave_balance, name FROM employees WHERE employee_id = ?", (employee_id,))
        result = cursor.fetchone()
        old_balance = result[0]
        employee_name = result[1]
        
        # Update balance
        cursor.execute("UPDATE employees SET leave_balance = ? WHERE employee_id = ?", (new_balance, employee_id))
        
        # Log balance change
        cursor.execute('''
            INSERT INTO leave_balance_history (employee_id, old_balance, new_balance, change_reason, changed_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (employee_id, old_balance, new_balance, reason, "HR Manager"))
        
        conn.commit()
        conn.close()
        
        return f"Leave balance updated for {employee_name} (ID: {employee_id}): {old_balance} → {new_balance} days. Reason: {reason}"
    
    except Exception as e:
        return f"Error updating leave balance: {str(e)}"



@mcp.tool()
def list_employees(department: Optional[str] = None) -> str:
    """List all employees or filter by department"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if department:
            cursor.execute('''
                SELECT employee_id, name, email, department, position, leave_balance, hire_date
                FROM employees WHERE department = ? ORDER BY name
            ''', (department,))
        else:
            cursor.execute('''
                SELECT employee_id, name, email, department, position, leave_balance, hire_date
                FROM employees ORDER BY name
            ''')
        
        employees = cursor.fetchall()
        conn.close()
        
        if not employees:
            dept_msg = f" in {department}" if department else ""
            return f"No employees found{dept_msg}."
        
        result = "Employee List:\n" + "="*80 + "\n"
        result += f"{'ID':<8} {'Name':<20} {'Email':<25} {'Dept':<12} {'Position':<15} {'Balance':<8}\n"
        result += "-"*80 + "\n"
        
        for emp in employees:
            result += f"{emp[0]:<8} {emp[1]:<20} {emp[2]:<25} {emp[3]:<12} {emp[4]:<15} {emp[5]:<8}\n"
        
        return result
    
    except Exception as e:
        return f"Error listing employees: {str(e)}"
    

@mcp.tool()
def get_employee_details(employee_id: str) -> str:
    """Get detailed information about an employee"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        # Get employee details
        cursor.execute('''
            SELECT employee_id, name, email, department, position, leave_balance, hire_date, created_at
            FROM employees WHERE employee_id = ?
        ''', (employee_id,))
        
        emp = cursor.fetchone()
        
        # Get leave history count
        cursor.execute('''
            SELECT COUNT(*) FROM leave_records WHERE employee_id = ?
        ''', (employee_id,))
        leave_count = cursor.fetchone()[0]
        
        # Get recent balance changes
        cursor.execute('''
            SELECT change_reason, old_balance, new_balance, changed_at
            FROM leave_balance_history WHERE employee_id = ?
            ORDER BY changed_at DESC LIMIT 5
        ''', (employee_id,))
        balance_history = cursor.fetchall()
        
        conn.close()
        
        result = f"Employee Details for {emp[1]} (ID: {emp[0]})\n"
        result += "="*50 + "\n"
        result += f"Email: {emp[2]}\n"
        result += f"Department: {emp[3]}\n"
        result += f"Position: {emp[4]}\n"
        result += f"Current Leave Balance: {emp[5]} days\n"
        result += f"Hire Date: {emp[6]}\n"
        result += f"Total Leaves Taken: {leave_count} days\n\n"
        
        if balance_history:
            result += "Recent Balance Changes:\n"
            result += "-"*30 + "\n"
            for change in balance_history:
                result += f"{change[0]}: {change[1]} → {change[2]} ({change[3]})\n"
        
        return result
    
    except Exception as e:
        return f"Error getting employee details: {str(e)}"



@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Check remaining leave days for an employee"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        cursor.execute('''
            SELECT name, leave_balance FROM employees WHERE employee_id = ?
        ''', (employee_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return f"{result[0]} (ID: {employee_id}) has {result[1]} leave days remaining."
        return "Employee ID not found."
    
    except Exception as e:
        return f"Error checking leave balance: {str(e)}"




@mcp.tool()
def apply_leave(employee_id: str, leave_dates: List[str], leave_type: str = "Annual") -> str:
    """Apply leave for specific dates"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        # Get current balance and name
        cursor.execute('''
            SELECT leave_balance, name FROM employees WHERE employee_id = ?
        ''', (employee_id,))
        
        result = cursor.fetchone()
        available_balance = result[0]
        employee_name = result[1]
        requested_days = len(leave_dates)
        
        if available_balance < requested_days:
            return f"Insufficient leave balance for {employee_name}. Requested: {requested_days} days, Available: {available_balance} days."
        
        # Validate date format and check for duplicates
        valid_dates = []
        for date_str in leave_dates:
            try:
                # Validate date format (YYYY-MM-DD)
                datetime.strptime(date_str, '%Y-%m-%d')
                
                # Check if already applied for this date
                cursor.execute('''
                    SELECT 1 FROM leave_records WHERE employee_id = ? AND leave_date = ?
                ''', (employee_id, date_str))
                
                if cursor.fetchone():
                    continue  # Skip duplicate date
                
                valid_dates.append(date_str)
            except ValueError:
                continue  # Skip invalid date format
        
        if not valid_dates:
            return "No valid dates provided or all dates are already taken."
        
        # Apply leave for valid dates
        for leave_date in valid_dates:
            cursor.execute('''
                INSERT INTO leave_records (employee_id, leave_date, leave_type)
                VALUES (?, ?, ?)
            ''', (employee_id, leave_date, leave_type))
        
        # Update balance
        new_balance = available_balance - len(valid_dates)
        cursor.execute('''
            UPDATE employees SET leave_balance = ? WHERE employee_id = ?
        ''', (new_balance, employee_id))
        
        # Log balance change
        cursor.execute('''
            INSERT INTO leave_balance_history (employee_id, old_balance, new_balance, change_reason, changed_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (employee_id, available_balance, new_balance, f"Applied {len(valid_dates)} days leave", employee_name))
        
        conn.commit()
        conn.close()
        
        applied_days = len(valid_dates)
        return f"Leave applied successfully for {employee_name}. Applied: {applied_days} day(s), Remaining balance: {new_balance} days."
    
    except Exception as e:
        return f"Error applying leave: {str(e)}"


@mcp.tool()
def get_leave_history(employee_id: str) -> str:
    """Get list of leave dates for employee"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not employee_exists(employee_id):
            conn.close()
            return f"Employee with ID {employee_id} not found."
        
        # Get employee name
        cursor.execute("SELECT name FROM employees WHERE employee_id = ?", (employee_id,))
        employee_name = cursor.fetchone()[0]
        
        # Get leave history
        cursor.execute('''
            SELECT leave_date, leave_type, status, applied_at
            FROM leave_records WHERE employee_id = ?
            ORDER BY leave_date DESC
        ''', (employee_id,))
        
        leave_records = cursor.fetchall()
        conn.close()
        
        if not leave_records:
            return f"No leave records found for {employee_name} (ID: {employee_id})."
        
        result = f"Leave History for {employee_name} (ID: {employee_id})\n"
        result += "="*60 + "\n"
        result += f"{'Date':<12} {'Type':<10} {'Status':<10} {'Applied At':<20}\n"
        result += "-"*60 + "\n"
        
        for record in leave_records:
            result += f"{record[0]:<12} {record[1]:<10} {record[2]:<10} {record[3]:<20}\n"
        
        return result
    
    except Exception as e:
        return f"Error getting leave history: {str(e)}"


# Reporting and Analytics Tools

@mcp.tool()
def get_department_summary(department: Optional[str] = None) -> str:
    """Get leave summary by department or overall"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if department:
            cursor.execute('''
                SELECT COUNT(*) as emp_count, 
                       SUM(leave_balance) as total_balance,
                       AVG(leave_balance) as avg_balance
                FROM employees WHERE department = ?
            ''', (department,))
            
            result = cursor.fetchone()
            if not result or result[0] == 0:
                return f"No employees found in {department} department."
            
            emp_count, total_balance, avg_balance = result
            
            summary = f"Department Summary for {department}\n"
            summary += "="*40 + "\n"
            summary += f"Total Employees: {emp_count}\n"
            summary += f"Total Leave Balance: {total_balance} days\n"
            summary += f"Average Leave Balance: {avg_balance:.1f} days\n"
            
        else:
            # Overall summary
            cursor.execute('''
                SELECT department, COUNT(*) as emp_count, 
                       SUM(leave_balance) as total_balance,
                       AVG(leave_balance) as avg_balance
                FROM employees 
                GROUP BY department
                ORDER BY department
            ''')
            
            dept_summaries = cursor.fetchall()
            
            if not dept_summaries:
                return "No employees found in the system."
            
            summary = "Overall Leave Summary by Department\n"
            summary += "="*60 + "\n"
            summary += f"{'Department':<15} {'Employees':<10} {'Total Balance':<15} {'Avg Balance':<12}\n"
            summary += "-"*60 + "\n"
            
            total_employees = 0
            total_balance = 0
            
            for dept_data in dept_summaries:
                dept, emp_count, dept_total, dept_avg = dept_data
                summary += f"{dept:<15} {emp_count:<10} {dept_total:<15} {dept_avg:<12.1f}\n"
                total_employees += emp_count
                total_balance += dept_total
            
            summary += "-"*60 + "\n"
            summary += f"{'TOTAL':<15} {total_employees:<10} {total_balance:<15} {total_balance/total_employees:<12.1f}\n"
        
        conn.close()
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@mcp.tool()
def get_recent_leave_activity(days: int = 30) -> str:
    """Get recent leave applications"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.name, e.employee_id, lr.leave_date, lr.leave_type, lr.applied_at
            FROM leave_records lr
            JOIN employees e ON lr.employee_id = e.employee_id
            WHERE lr.applied_at >= datetime('now', '-{} days')
            ORDER BY lr.applied_at DESC
        '''.format(days))
        
        recent_leaves = cursor.fetchall()
        conn.close()
        
        if not recent_leaves:
            return f"No leave applications found in the last {days} days."
        
        result = f"Recent Leave Activity (Last {days} days)\n"
        result += "="*70 + "\n"
        result += f"{'Employee':<20} {'ID':<8} {'Leave Date':<12} {'Type':<10} {'Applied At':<20}\n"
        result += "-"*70 + "\n"
        
        for leave in recent_leaves:
            result += f"{leave[0]:<20} {leave[1]:<8} {leave[2]:<12} {leave[3]:<10} {leave[4]:<20}\n"
        
        return result
    
    except Exception as e:
        return f"Error getting recent activity: {str(e)}"

# Resources and Prompts


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}! Welcome to the HR Leave Management System. How can I assist you today?"


@mcp.prompt()
def hr_dashboard_prompt() -> str:
    """Generate a comprehensive HR dashboard prompt"""
    return """Create a comprehensive HR dashboard showing:
1. Employee overview with key metrics
2. Leave balance distribution
3. Recent leave activity
4. Department-wise summaries
5. Any alerts or recommendations

Format it as a professional dashboard with clear sections and visual indicators where appropriate."""


@mcp.prompt()
def leave_report_prompt(employee_id: str) -> str:
    """Generate a detailed leave report prompt for an employee"""
    return f"""Generate a detailed leave report for employee {employee_id} including:
1. Current leave balance and status
2. Leave history with dates and types
3. Balance change history
4. Recommendations for leave planning
5. Any patterns or insights from their leave usage

Make it professional and actionable for HR review."""

# Initialize with sample data
def init_sample_data():
    """Initialize database with sample data for demo purposes"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return  # Data already exists
    
    # Sample employees
    sample_employees = [
        ("E001", "John Smith", "john.smith@company.com", "Engineering", "Senior Developer", 18),
        ("E002", "Sarah Johnson", "sarah.johnson@company.com", "HR", "HR Manager", 22),
        ("E003", "Mike Wilson", "mike.wilson@company.com", "Marketing", "Marketing Specialist", 20),
        ("E004", "Emily Brown", "emily.brown@company.com", "Engineering", "Junior Developer", 20),
        ("E005", "David Lee", "david.lee@company.com", "Sales", "Sales Manager", 15),
    ]
    
    for emp_data in sample_employees:
        cursor.execute('''
            INSERT INTO employees (employee_id, name, email, department, position, leave_balance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', emp_data)
        
        # Log initial balance
        cursor.execute('''
            INSERT INTO leave_balance_history (employee_id, old_balance, new_balance, change_reason, changed_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (emp_data[0], 0, emp_data[5], "Initial balance", "System"))
    
    # Sample leave records
    sample_leaves = [
        ("E001", "2024-12-25", "Holiday"),
        ("E001", "2024-12-26", "Holiday"),
        ("E002", "2024-12-30", "Personal"),
        ("E005", "2024-12-31", "Personal"),
    ]
    
    for leave_data in sample_leaves:
        cursor.execute('''
            INSERT INTO leave_records (employee_id, leave_date, leave_type)
            VALUES (?, ?, ?)
        ''', leave_data)
    
    conn.commit()
    conn.close()

# Initialize sample data
init_sample_data()




# To start the server:
if __name__ == "__main__":
    mcp.run()