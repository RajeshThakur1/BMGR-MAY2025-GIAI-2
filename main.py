from fastapi import FastAPI, HTTPException

from starlette.responses import  Response
import uvicorn
from app.api.api import read_user, read_questions

app = FastAPI()

@app.get("/")
def root():
    return {"massage": "Fast API in Python"}

@app.get("/users")
def read_users():
    return read_user()

@app.post('/question')
def get_quest(que_id:int):
    return read_questions(que_id)




if __name__ == "__main__":
    uvicorn.run(app, port=5002)
