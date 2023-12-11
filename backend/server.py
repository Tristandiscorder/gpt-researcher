from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
from gpt_researcher.utils.websocket_manager import WebSocketManager
from .utils import write_md_to_pdf
from typing import List


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str
    pdf1: List[bytes]


app = FastAPI()

app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")

templates = Jinja2Templates(directory="./frontend")

manager = WebSocketManager()


# Dynamic directory for outputs once first research is run
@app.on_event("startup")
def startup_event():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "report": None})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json(mode="binary") 
            if data.startswith("start"):
                json_data = json.loads(data[6:])
                
                research_request = ResearchRequest(**json_data)
                task = research_request.task
                pdf1 = research_request.pdf1  # pdf1 as binary data
                report_type = research_request.report_type
                ################before declaring pdf1
                #task = json_data.get("task")
                #pdf1 = data.get("pdf1")  # Get pdf1 as binary data
                #report_type = json_data.get("report_type")
                if task and report_type:
                    report = await manager.start_streaming(task, pdf1, report_type, websocket)
                    path = await write_md_to_pdf(report)
                    await websocket.send_json({"type": "path", "output": path})
                else:
                    print("Error: not enough parameters provided.")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)

