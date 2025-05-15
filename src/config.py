"""Project configuration handling."""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Literal

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Model configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "t5")  # Default to t5 if not specified
MODEL_PATH = os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "t5" / "t5_english_french_model"))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", str(ARTIFACTS_DIR / "t5" / "t5_english_french_model"))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t")

# Inference parameters
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "4"))
NUM_RETURN_SEQUENCES = int(os.getenv("NUM_RETURN_SEQUENCES", "1"))

# Model types
ModelType = Literal["seq2seq", "t5"]

def get_model_config(model_type: ModelType = None):
    """Get model configuration based on model type.
    
    Args:
        model_type: The type of model to use. If None, uses the default from environment.
        
    Returns:
        dict: Model configuration dictionary
    """
    if model_type is None:
        model_type = MODEL_TYPE
        
    if model_type == "seq2seq":
        # Seq2Seq specific configurations
        return {
            "model_path": os.getenv("SEQ2SEQ_MODEL_PATH", str(ARTIFACTS_DIR / "seq2seq" / "model.keras")),
            "tokenizer_eng_path": os.getenv("SEQ2SEQ_TOKENIZER_ENG_PATH", 
                                           str(ARTIFACTS_DIR / "seq2seq" / "tokenizer_eng.json")),
            "tokenizer_fr_path": os.getenv("SEQ2SEQ_TOKENIZER_FR_PATH", 
                                          str(ARTIFACTS_DIR / "seq2seq" / "tokenizer_fr.json")),
            "max_length": MAX_LENGTH,
        }
    elif model_type == "t5":
        # T5 specific configurations
        return {
            "model_path": MODEL_PATH,
            "tokenizer_path": TOKENIZER_PATH,
            "max_length": MAX_LENGTH,
            "beam_size": BEAM_SIZE,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
