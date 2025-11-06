from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fracture_pipeline import FracturePipeline
import os
import shutil

# Inicializa la app
app = FastAPI(
    title="🩻 Fracture Detection API",
    description="API que clasifica imágenes de rayos X y localiza la fractura usando Grad-CAM.",
    version="1.0.0"
)

# Configurar CORS (por si la usas en un dashboard o frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa el modelo
MODEL_PATH = "modelo_fractura_resnet50_2.pth"
pipeline = FracturePipeline(MODEL_PATH)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", tags=["General"])
def root():
    """Verifica que la API está en funcionamiento."""
    return {"message": "✅ API activa. Usa /docs para probarla en Swagger UI."}

@app.post("/predict/", tags=["Predicción"])
async def predict(file: UploadFile = File(...)):
    """
    Clasifica si una radiografía está fracturada o no.
    Si se detecta fractura, devuelve una imagen con el mapa de calor (Grad-CAM).
    """
    # Guardar imagen temporalmente
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ejecutar pipeline
    result = pipeline.predict(file_path)

    if result["prediction"] == "fractured":
        return FileResponse(result["heatmap_path"], media_type="image/jpeg")
    else:
        return JSONResponse(content={
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "message": "✅ No se detecta fractura"
        })
