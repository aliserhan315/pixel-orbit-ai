#if u dont have openai api use this 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBPE_Ihw6_M52lG204_ZoanMB_QbhvYL5E")
imagen = genai.ImageGenerationModel("imagen-3.0-generate-001")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    number_of_images: int = 1  # Default to one image
    aspect_ratio: str = "1:1"  # Default aspect ratio

# Generate image endpoint
@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    """
    Generate images using the Gemini API based on the input prompt and settings.
    """
    try:
        # Generate images via Gemini API
        result = imagen.generate_images(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            number_of_images=request.number_of_images,
            aspect_ratio=request.aspect_ratio,
            safety_filter_level="block_only_high",
            person_generation="allow_adult",
        )

        # Convert images to Base64
        images_base64 = [
            base64.b64encode(BytesIO(image._pil_image.tobytes()).getvalue()).decode("utf-8")
            for image in result.images
        ]

        return {"images": images_base64}

    except Exception as e:
        # Handle exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint to check API availability.
    """
    return {"message": "Welcome to the Gemini Image Generator API!"}
