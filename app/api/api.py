from fastapi import FastAPI, HTTPException
from starlette.responses import Response
import uvicorn
app = FastAPI()
@app.get("/")
def root():
    return {"message": "Fast API in python"}

@app.post("/add")
def add(a, b):
    a = float(a)
    b = float(b)
    c = a + b
    return c

@app.post("/sub")
def sub(a,b):
    c = a-b
    return c


if __name__ == "__main__":
    # print(add(2,4.3))
    # print(sub(b=6, a=7))
    #
    # print("This is the initialization of my AI Journey")

    uvicorn.run(app,port=5001)

