from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from .routes import router as api_router, auth_router, finetune_router, webhook_router, qa_router
from .database import init_db
from core.exceptions import GistifyError, UnsupportedFileTypeError, EmptyContentError, FileProcessingError, UrlConnectionError

# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    init_db()
    yield
    # Shutdown event (if any)
    pass

app = FastAPI(title="Gistify API", lifespan=lifespan)

@app.exception_handler(GistifyError)
async def gistify_exception_handler(request: Request, exc: GistifyError):
    status_code = 500  # Default to Internal Server Error
    if isinstance(exc, (UnsupportedFileTypeError, EmptyContentError, UrlConnectionError)):
        status_code = 400 # Bad Request for these specific user errors
    elif isinstance(exc, FileProcessingError):
        status_code = 422 # Unprocessable Entity for file processing issues

    return JSONResponse(
        status_code=status_code,
        content={"error_type": exc.__class__.__name__, "detail": exc.detail},
    )

app.include_router(api_router)
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(finetune_router, prefix="/finetune", tags=["finetune"])
app.include_router(webhook_router, prefix="/webhooks", tags=["webhooks"])
app.include_router(qa_router, prefix="/qa", tags=["qa"])