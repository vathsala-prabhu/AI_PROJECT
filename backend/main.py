"""FastAPI app: exposes the sentiment pipeline over HTTP."""
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.auth.service import register_user
from backend.db.database import get_db
from backend.schemas import AnalyzeRequest, RegisterRequest
from backend.services.file_parser import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_BYTES,
    decode_bytes,
    get_extension,
    parse_csv,
)
from backend.services.report import build_csv_report, build_report
from backend.services.validation import ValidationError

app = FastAPI(
    title="MoodMentor API",
    description="Milestone 1: text ingestion and baseline sentiment analysis.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "MoodMentor API is running. See /docs for the API."}


@app.get("/health")
def health():
    """Simple check that the server is running."""
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Analyze pasted text and return the sentiment report."""
    try:
        return build_report(request.text)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    column: str | None = Form(None),
):
    """Analyze an uploaded .txt (one report) or .csv (per-row + overall)."""
    extension = get_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .txt or .csv file.",
        )

    data = await file.read(MAX_FILE_BYTES + 1)  # read one byte past the limit
    try:
        content = decode_bytes(data)
        if extension == ".txt":
            return {
                "filename": file.filename,
                "file_type": "txt",
                "report": build_report(content),
            }
        rows, column_used = parse_csv(content, column)
        return {
            "filename": file.filename,
            "file_type": "csv",
            **build_csv_report(rows, column_used),
        }
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/register", status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    try:
        user = register_user(db, request.username, request.email, request.password)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": user.id, "username": user.username, "email": user.email}