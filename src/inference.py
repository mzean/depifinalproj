"""Model inference functionality for English to French translation."""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Any, Optional, List, Union

from src.config import get_model_config, ModelType


class TranslationModel:
    """Base class for translation models."""
    
    def __init__(self, model_type: ModelType):
        """Initialize translation model.
        
        Args:
            model_type: Type of model to use ("seq2seq" or "t5")
        """
        self.model_type = model_type
        self.model_config = get_model_config(model_type)
        self.model = None
        
    def load(self):
        """Load the model."""
        raise NotImplementedError("Subclasses must implement load method")
        
    def translate(self, text: str) -> str:
        """Translate English text to French.
        
        Args:
            text: English text to translate
            
        Returns:
            str: Translated French text
        """
        raise NotImplementedError("Subclasses must implement translate method")


class Seq2SeqTranslationModel(TranslationModel):
    """Seq2Seq LSTM model for translation."""
    
    def __init__(self):
        """Initialize Seq2Seq translation model."""
        super().__init__("seq2seq")
        self.tokenizer_eng = None
        self.tokenizer_fr = None
        self.encoder_model = None
        self.decoder_model = None
        self.max_length = self.model_config["max_length"]
        
    def load(self):
        """Load Seq2Seq model and tokenizers."""
        # Load tokenizers
        with open(self.model_config["tokenizer_eng_path"], 'r') as f:
            tokenizer_json = f.read()
            self.tokenizer_eng = tokenizer_from_json(tokenizer_json)
            
        with open(self.model_config["tokenizer_fr_path"], 'r') as f:
            tokenizer_json = f.read()
            self.tokenizer_fr = tokenizer_from_json(tokenizer_json)
        
        # Load the trained model
        self.model = load_model(self.model_config["model_path"])
        
        # Create inference models (encoder and decoder)
        self._create_inference_models()
        
        return self
    
    def _create_inference_models(self):
        """Create separate encoder and decoder models for inference."""
        # The trained model is the full training model with both encoder and decoder
        # We need to extract them for inference
        
        # Get the encoder layers
        encoder_inputs = self.model.input[0]  # Input to the model
        encoder_outputs, state_h, state_c = self.model.layers[3].output  # LSTM layer outputs
        encoder_states = [state_h, state_c]
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        # Get the decoder layers
        decoder_inputs = self.model.input[1]
        decoder_state_input_h = tf.keras.layers.Input(shape=(512,))  # hidden state input
        decoder_state_input_c = tf.keras.layers.Input(shape=(512,))  # cell state input
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        # Get decoder embedding layer
        decoder_embedding = self.model.layers[2]
        decoder_embedding_outputs = decoder_embedding(decoder_inputs)
        
        # Get decoder LSTM layer
        decoder_lstm = self.model.layers[4]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding_outputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        
        # Get decoder dense layer
        decoder_dense = self.model.layers[5]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Create the decoder model
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
    
    def translate(self, text: str) -> str:
        """Translate English text to French using Seq2Seq model.
        
        Args:
            text: English text to translate
            
        Returns:
            str: Translated French text
        """
        # Tokenize and pad the input sequence
        seq = self.tokenizer_eng.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
        
        # Encode the input sequence
        states_value = self.encoder_model.predict(padded, verbose=0)
        
        # Generate target sequence with start token (assuming index 1 is start token)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = 1  # Start token
        
        # Lists to store the generated tokens
        output_tokens = []
        
        # Generation loop
        stop_condition = False
        while not stop_condition:
            # Predict next token and states
            output_tokens_batch, h, c = self.decoder_model.predict(
                [target_seq] + states_value, verbose=0
            )
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens_batch[0, 0, :])
            output_tokens.append(sampled_token_index)
            
            # Check for end of sequence or max length
            if (sampled_token_index == 0 or  # 0 is usually end token
                len(output_tokens) > self.max_length - 1):
                stop_condition = True
            
            # Update the target sequence
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            # Update states
            states_value = [h, c]
        
        # Convert token indices to words
        translated_sentence = []
        for i in output_tokens:
            if i in self.tokenizer_fr.index_word and i != 0:  # Skip padding token
                translated_sentence.append(self.tokenizer_fr.index_word[i])
            else:
                translated_sentence.append(' ')
        
        return ' '.join(translated_sentence)


class T5TranslationModel(TranslationModel):
    """T5 Transformer model for translation."""
    
    def __init__(self):
        """Initialize T5 translation model."""
        super().__init__("t5")
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        """Load T5 model and tokenizer."""
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_config["model_path"])
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_config["tokenizer_path"])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load T5 model: {str(e)}")
    
    def translate(self, text: str) -> str:
        """Translate English text to French using T5 model.
        
        Args:
            text: English text to translate
            
        Returns:
            str: Translated French text
        """
        # Prepare input text
        input_text = f"translate English to French: {text}"
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.model_config["max_length"]
        ).input_ids.to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, 
                max_length=self.model_config["max_length"],
                num_beams=self.model_config["beam_size"],
                num_return_sequences=self.model_config["num_return_sequences"],
                early_stopping=True
            )
        
        # Decode the generated tokens
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation


def get_translation_model(model_type: ModelType = None) -> TranslationModel:
    """Factory function to get the appropriate translation model.
    
    Args:
        model_type: Type of model to use ("seq2seq" or "t5"). 
                   If None, uses the default from config.
    
    Returns:
        TranslationModel: An instance of the translation model
    """
    from src.config import MODEL_TYPE
    
    if model_type is None:
        model_type = MODEL_TYPE
    
    if model_type == "seq2seq":
        return Seq2SeqTranslationModel().load()
    elif model_type == "t5":
        return T5TranslationModel().load()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
