from dotenv import find_dotenv, load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

LLM_TYPE = ["openai", "ollama"][1]
EMBEDDING_TYPE = ["openai", "ollama"][1]

ORIGINAL_DOCS_PATH = "./data/warehouse_docs/"
VECTOR_STORE_PATH = "./data/chroma_db_warehouse"

# Load environment variables
load_dotenv(find_dotenv())

# Performance settings
ENABLE_CACHE = True
REDIS_URL = "redis://localhost:6379"
MAX_WORKERS = 5
BATCH_SIZE = 5
EARLY_STOPPING_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.7

# Model settings for performance
if LLM_TYPE == "ollama":
    llm = ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=2048,  # Reduced context for faster inference
        num_gpu=1,
        num_thread=8,
    )
elif LLM_TYPE == "openai":
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        # "gpt-3.5-turbo",
        temperature=0,
        # max_retries=3,
        # request_timeout=60,
        # headers=HEADERS
    )

EMBEDDING = (
    OllamaEmbeddings(
        model="llama3.2",
        base_url="http://localhost:11434",
        # verbose=True
    )
    if EMBEDDING_TYPE == "ollama"
    else OpenAIEmbeddings()
)
