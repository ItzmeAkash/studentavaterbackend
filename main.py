from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import token

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


@app.get("/")
def root():
    return {"message": "LiveKit Avatar Agent API", "status": "running"}
