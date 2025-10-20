from fastapi import FastAPI
from app.api import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def read_root():
    return {
            "name": "SagePick AI Service",
            "version": "0.1.0",
            "description": "Welcome to the SagePick AI Service API!"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)