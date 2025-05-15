"""Request handling for the translation API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
import time

from src.config import ModelType
from src.inference import get_translation_model


# Request and response models
class TranslationRequest(BaseModel):
    """Request model for translation API."""
    
    text: str = Field(..., min_length=1, max_length=1000, 
                     description="English text to translate to French")
    model_type: Optional[ModelType] = Field(None, 
                                          description="Model type to use for translation (seq2seq or t5)")
    
    @validator('text')
    def text_not_empty(cls, v):
        """Validate that text is not empty."""
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty')
        return v


class TranslationResponse(BaseModel):
    """Response model for translation API."""
    
    original_text: str = Field(..., description="Original English text")
    translated_text: str = Field(..., description="Translated French text")
    model_used: ModelType = Field(..., description="Model used for translation")
    processing_time: float = Field(..., description="Processing time in seconds")


# Create router
router = APIRouter()


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest) -> TranslationResponse:
    """Translate English text to French.
    
    Args:
        request: Translation request with text and optional model type
        
    Returns:
        TranslationResponse: Translation result
    """
    try:
        start_time = time.time()
        
        # Get appropriate translation model
        model = get_translation_model(request.model_type)
        
        # Translate text
        translated_text = model.translate(request.text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            model_used=model.model_type,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.
    
    Returns:
        Dict: Status information
    """
    return {"status": "healthy"}
