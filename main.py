from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import token, chat

app = FastAPI(title="LiveKit Avatar Agent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(token.router, prefix="/api", tags=["token"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
def root():
    return {"message": "LiveKit Avatar Agent API", "status": "running"}
