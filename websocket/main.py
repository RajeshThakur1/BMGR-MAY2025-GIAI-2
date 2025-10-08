import uvicorn
from fastapi import FastAPI, HTTPException
import publisher_module, subscriber_module
import logging

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backend")

app = FastAPI()
app.include_router(publisher_module.router)
app.include_router(subscriber_module.router)

@app.on_event("startup")
async def startup_event():
    logger.info("WebSocket server started")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
