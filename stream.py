from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio

app = FastAPI()

async def event_generator():
    """
    An async generator that yields Server-Sent Events (SSE) messages every second.
    Each message is prefixed with 'data:' and ended with a blank line as per SSE spec.
    """
    count = 0
    while True:
        # Prepare an SSE-formatted message
        yield f"data: Message {count}\n\n"
        count += 1
        await asyncio.sleep(1)

@app.get("/stream")
async def stream():
    """
    Stream endpoint using SSE. Clients can connect and will receive a continuous
    stream of messages in real time.
    """
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def get_home():
    """
    Simple HTML page demonstrating consumption of the SSE stream.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>FastAPI SSE Demo</title>
    </head>
    <body>
        <h1>Server-Sent Events Streaming</h1>
        <ul id="messages"></ul>
        <script>
            const evtSource = new EventSource('/stream');
            const list = document.getElementById('messages');
            evtSource.onmessage = function(event) {
                const item = document.createElement('li');
                item.textContent = event.data;
                list.appendChild(item);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# To run this app, install dependencies and start uvicorn:
#   pip install fastapi uvicorn
#   uvicorn stream:app --reload --host 0.0.0.0 --port 8000
