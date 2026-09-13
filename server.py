import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
from pipeline import HighPrecisionExtractionEngine

app = FastAPI(title="Industrial Matting Service", version="2.0.0")

engine = None

@app.on_event("startup")
def load_engine():
    global engine
    engine = HighPrecisionExtractionEngine()

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

@app.post("/v2/extract")
async def extract_foreground(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="MIME type must be an image.")

    content = await file.read()
    try:
        input_image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image payload.")

    result_rgba = engine.execute(input_image)

    output_stream = io.BytesIO()
    result_rgba.save(output_stream, format="PNG", optimize=True)
    output_stream.seek(0)

    return StreamingResponse(output_stream, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)