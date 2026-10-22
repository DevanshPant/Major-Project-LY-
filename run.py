"""Single entry point.  Run with:  python run.py

Launches uvicorn against app.main:app on port 8000.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
