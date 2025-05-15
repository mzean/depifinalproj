"""FastAPI application for English to French translation."""

import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from pathlib import Path

from src.config import API_HOST, API_PORT, DEBUG_MODE
from src.request import router as translation_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="English-French Translator API",
    description="API for translating English text to French using Seq2Seq and T5 models",
    version="0.1.0",
)

# Add API routers
app.include_router(translation_router, tags=["translation"])

# Create templates directory and HTML files for the web interface
templates_path = Path("templates")
templates_path.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Create simple HTML template for the web interface
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>English-French Translator</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        label {
            font-weight: bold;
        }
        textarea {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 150px;
            font-family: inherit;
        }
        select, button {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 1rem;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .metadata {
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <h1>English-French Translator</h1>
    
    <div class="container">
        <div class="form-group">
            <label for="englishText">English Text:</label>
            <textarea id="englishText" placeholder="Enter English text to translate..."></textarea>
        </div>
        
        <div class="form-group">
            <label for="modelType">Translation Model:</label>
            <select id="modelType">
                <option value="t5">T5 Transformer</option>
                <option value="seq2seq">Seq2Seq LSTM</option>
            </select>
        </div>
        
        <button id="translateButton">Translate</button>
        
        <div class="loading" id="loading">
            Translating...
        </div>
        
        <div class="result" id="result" style="display: none;">
            <div class="form-group">
                <label for="frenchText">French Translation:</label>
                <textarea id="frenchText" readonly></textarea>
            </div>
            <div class="metadata" id="metadata"></div>
        </div>
    </div>

    <script>
        document.getElementById('translateButton').addEventListener('click', async () => {
            const englishText = document.getElementById('englishText').value.trim();
            const modelType = document.getElementById('modelType').value;
            
            if (!englishText) {
                alert('Please enter some text to translate.');
                return;
            }
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const frenchText = document.getElementById('frenchText');
            const metadata = document.getElementById('metadata');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: englishText,
                        model_type: modelType
                    }),
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    frenchText.value = data.translated_text;
                    metadata.textContent = `Model: ${data.model_used}, Processing time: ${data.processing_time.toFixed(3)}s`;
                    result.style.display = 'block';
                } else {
                    alert(`Translation failed: ${data.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

# Write the HTML template to the templates directory
with open(templates_path / "index.html", "w") as f:
    f.write(index_html)

@app.get("/", response_class=HTMLResponse)
async def get_web_interface(request: Request):
    """Serve the web interface.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Web interface HTML
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their processing time.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware in the chain
        
    Returns:
        Response: FastAPI response
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.debug(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Process time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    logger.info(f"Starting server at http://{API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=DEBUG_MODE)
