#  uvicorn main:app --reload

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
from app.api.routes import router

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Count"]
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0", port=8000, reload=True, log_level="info")
