from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 密钥配置
API_KEY = "your_secret_api_key_12345"  # 修改为你的密钥
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# 模型配置
MODEL_PATH = "deepseek_sql_model"  # 确保这是你的模型路径
model = None
tokenizer = None

class SQLQuery(BaseModel):
    query: str

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "SQL Explanation API is running"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")

def load_model():
    global model, tokenizer
    try:
        if model is None or tokenizer is None:
            logger.info("Loading model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                MODEL_PATH,
                max_seq_length=2048,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(model)
            if torch.cuda.is_available():
                model = model.to("cuda")
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_explanation(query: str) -> Optional[str]:
    try:
        prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a SQL expert with advance understanding of SQL queries. You can understand database schema from the query. Think like sql expert and generate a summary of the query which explains the use case of the query. As in
what the query is trying to read from the database in a usecase sense.

### Query:
{}

### Response:
<think>{}"""

        inputs = tokenizer([prompt_style.format(query, "")], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )

        response = tokenizer.batch_decode(outputs)[0]
        return response.split("### Response:")[1].strip()
    except Exception as e:
        logger.error(f"Error in generate_explanation: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    load_model()
    logger.info("Application startup complete")

@app.post("/explain-sql/")
async def explain_sql(
    query: SQLQuery,
    api_key: str = Depends(get_api_key)
):
    logger.info("Received SQL explanation request")
    try:
        explanation = generate_explanation(query.query)
        if explanation:
            logger.info("Successfully generated explanation")
            return {
                "status": "success",
                "explanation": explanation
            }
        else:
            logger.error("Failed to generate explanation")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate explanation"
            )
    except Exception as e:
        logger.error(f"Error in explain_sql: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )