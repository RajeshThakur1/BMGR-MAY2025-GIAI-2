from crewai.tools import tool

class CalculatorTools():
    
    @tool
    def calculate(expression: str) -> float:
        """This tool allows you to perform mathematical operations like 
            addition, subtraction, multiplication, and division. 
            It takes a mathematical expression as input, such as 150+25 or 300/5*2.
            """
        try:
            return eval(expression)
        except Exception as e:
            return f"Error: Invalid syntax in mathematical expression:  {e}"


        
    

