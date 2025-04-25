import os
import torch
import uvicorn
import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import ms-swift related modules
from swift.llm import PtEngine, RequestConfig, InferRequest, get_template

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable to specify using GPU 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Create FastAPI application
app = FastAPI(title="Consistency Model API Server (Swift)")

# Define request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 10

# Define response models
class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = "consistency-model-response"
    object: str = "chat.completion"
    choices: List[ChatCompletionChoice]

# Global variable to store engine
engine = None

# Model loading function
def load_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    global engine
    
    logger.info(f"Loading consistency check model: {model_name} to GPU 0 (physical device GPU 4)")
    
    try:
        # Set system prompt
        system_prompt = "You are a professional question-answering assistant."
        
        # Load model using PtEngine
        engine = PtEngine(
            model_name,
            max_batch_size=1,
            torch_dtype=torch.float16,
            default_system=system_prompt,
            trust_remote_code=True
        )
        
        # Test if model is available
        test_request = InferRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        test_config = RequestConfig(max_tokens=10)
        engine.infer([test_request], test_config)
        
        logger.info("Model loaded and tested successfully")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        
        # Try to use backup model
        try:
            logger.info("Trying to use backup model Qwen/Qwen2.5-3B-Instruct")
            engine = PtEngine(
                "Qwen/Qwen2.5-3B-Instruct",
                max_batch_size=1,
                torch_dtype=torch.float16,
                default_system=system_prompt,
                trust_remote_code=True
            )
            
            # Test backup model
            test_request = InferRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            )
            test_config = RequestConfig(max_tokens=10)
            engine.infer([test_request], test_config)
            
            logger.info("Backup model loaded and tested successfully")
            return True
        except Exception as backup_e:
            logger.error(f"Backup model loading also failed: {str(backup_e)}")
            return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        logger.error("Unable to load any model, the service may not work properly")

# Define chat completion endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    global engine
    
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request messages to format accepted by the model
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Create inference request
        infer_request = InferRequest(messages=messages)
        
        # Create request config
        request_config = RequestConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Run inference
        response = engine.infer([infer_request], request_config)[0]
        
        # Extract answer
        output_text = response.choices[0].message.content
        
        # Build response
        choice = ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=output_text.strip())
        )
        
        return ChatCompletionResponse(choices=[choice])
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Main function
if __name__ == "__main__":
    # Start service on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
