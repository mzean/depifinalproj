# English-French Translator

A machine translation system that translates English text to French using both Seq2Seq LSTM and T5 Transformer models.

## Project Overview

This project provides a machine translation service for converting English text to French using two different model architectures:

1. **Sequence-to-Sequence LSTM Model**: A traditional encoder-decoder architecture with LSTM units.
2. **T5 Transformer Model**: Google's T5 (Text-to-Text Transfer Transformer) fine-tuned on English-French translation.

The project includes both model training notebooks and a FastAPI service for deploying the models.

## Features

- Translation of English text to French
- Choice between Seq2Seq and T5 translation models
- RESTful API for translation services
- Comprehensive model training notebooks
- Simple web interface for trying out the translation

## Directory Structure

```
english-french-translator/
├── .env                    # Environment variables (actual values, not committed)
├── .env.example            # Example environment variables (for reference)
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── main.py                 # FastAPI application entry point
├── notebooks/              # Jupyter notebooks for model development
│   ├── eng_to_french.ipynb # Seq2Seq model notebook
│   └── transformer-t5.ipynb# T5 Transformer model notebook
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── inference.py        # Model inference code
│   └── request.py          # API request handling
└── artifacts/              # Model artifacts storage
    ├── seq2seq/            # Seq2Seq model files
    └── t5/                 # T5 model files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/english-french-translator.git
cd english-french-translator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your specific configuration
```

## Model Training

The project includes Jupyter notebooks for training both models:

- `notebooks/eng_to_french.ipynb`: Training the Seq2Seq LSTM model
- `notebooks/transformer-t5.ipynb`: Fine-tuning the T5 Transformer model

To train the models:

1. Open the respective notebook in Jupyter:
```bash
jupyter notebook notebooks/
```

2. Run all cells in the notebook to train the model
3. The trained models will be saved to the specified paths

## Running the API Server

Start the FastAPI server:

```bash
python main.py
```

By default, the server will run at http://localhost:8000.

## API Usage

### Translate Text

**Endpoint**: `/translate`

**Method**: POST

**Request Body**:
```json
{
  "text": "Hello, how are you?",
  "model_type": "t5"  // or "seq2seq"
}
```

**Response**:
```json
{
  "original_text": "Hello, how are you?",
  "translated_text": "Bonjour, comment allez-vous ?",
  "model_used": "t5"
}
```

## Web Interface

A simple web interface is available at http://localhost:8000 when the server is running.

## Model Performance

### Seq2Seq LSTM Model
- Accuracy: ~92%
- Training time: ~3 hours
- Model size: ~50MB

### T5 Transformer Model
- Accuracy: ~97%
- Training time: ~2 hours
- Model size: ~230MB

## Project Execution Plan

1. **Data Preparation**: Process the English-French dataset
2. **Model Training**: Train both Seq2Seq and T5 models
3. **Model Evaluation**: Evaluate models on test data
4. **Model Export**: Save models for inference
5. **API Development**: Create FastAPI endpoints
6. **Deployment**: Deploy the API service
7. **Testing & Optimization**: Test performance and optimize as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [English-French Translation Dataset](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)
- T5 Transformer: [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/t5)
