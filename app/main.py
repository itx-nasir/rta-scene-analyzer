from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid
import aiofiles
from app.models import yolo_model, scene_model, scene_classes, predict_objects, predict_scene
from app.utils import generate_scene_description

app = FastAPI(title="RTA Scene Analyzer", version="1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image contents
        contents = await file.read()
        
        # Save uploaded file
        upload_filename = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
        upload_path = f"static/uploads/{upload_filename}"
        
        async with aiofiles.open(upload_path, 'wb') as f:
            await f.write(contents)
        
        # Run object detection with bounding boxes
        objects, processed_filename = predict_objects(contents, yolo_model)
        
        # Run scene classification
        scene = predict_scene(contents, scene_model, scene_classes)
        
        # Generate scene description
        description = generate_scene_description(scene, objects)
        
        return {
            "scene_class": scene,
            "objects_detected": objects,
            "scene_description": description,
            "processed_image": processed_filename,
            "upload_image": upload_filename
        }
        
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RTA Scene Analyzer API is running"}
