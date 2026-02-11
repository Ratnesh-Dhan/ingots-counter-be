from fastapi import FastAPI
import uvicorn
from app.api.routes import router

app = FastAPI()
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000,reload=True, log_level="info")
