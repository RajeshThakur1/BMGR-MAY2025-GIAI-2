from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()
class UserCreateRequest(BaseModel):
    username: str
    email: str

# Define a simple user request body
# Endpoint to create a user
@app.post("/users")
def create_user():
    # Here you would add your logic to actually create the user
    # For demonstration, we're skipping storage logic
    # print(user.username)
    # print(user.email)
    # Return a generic success message
    return {"message": "User created successfully."}

# If running directly, use: uvicorn filename:app --reload

if __name__ == "__main__":
    uvicorn.run(app, port=8000)