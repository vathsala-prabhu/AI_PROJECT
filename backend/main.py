"""FastAPI app: exposes the sentiment pipeline over HTTP."""
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth.profile import update_profile
from backend.auth.security import create_access_token
from backend.auth.service import authenticate_user, register_user
from backend.db.crud import (
    list_analyses,
    overview_stats,
    save_analysis,
    save_csv_rows,
)
from backend.db.database import get_db
from backend.db.models import User
from backend.schemas import (
    AnalyzeRequest,
    LoginRequest,
    ProfileUpdate,
    RegisterRequest,
)
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


def user_dict(user: User) -> dict:
    """The user data that is safe to send to the browser (never the password)."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "bio": user.bio,
        "journaling_goal": user.journaling_goal,
        "journaling_frequency": user.journaling_frequency,
        "preferred_time": user.preferred_time,
    }


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


@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Check credentials and return an access token."""
    try:
        user = authenticate_user(db, request.username, request.password)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if user is None:
        # Same message for unknown user and wrong password, on purpose.
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    """Return the logged-in user's account and profile. Requires a valid token."""
    return user_dict(current_user)


@app.put("/me/profile")
def edit_profile(
    request: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the logged-in user's profile fields."""
    try:
        user = update_profile(db, current_user, request.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return user_dict(user)


@app.post("/my/analyze")
def analyze_and_save(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze text and save the result to the logged-in user's history."""
    try:
        report = build_report(request.text)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    row = save_analysis(db, current_user.id, report)
    report["saved_id"] = row.id
    return report


@app.post("/my/analyze-file")
async def analyze_file_and_save(
    file: UploadFile = File(...),
    column: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze an uploaded .txt/.csv and save the result(s) to the user's history."""
    result = await analyze_file(file=file, column=column)  # raises 400 on bad input
    if result["file_type"] == "txt":
        save_analysis(db, current_user.id, result["report"], source="txt")
        result["saved_entries"] = 1
    else:
        result["saved_entries"] = save_csv_rows(db, current_user.id, result["results"])
    return result


@app.get("/my/history")
def history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the logged-in user's saved analyses, newest first."""
    return [
        {
            "id": r.id,
            "source": r.source,
            "text": r.input_text[:200],
            "label": r.label,
            "compound": r.compound,
            "positive": r.positive,
            "negative": r.negative,
            "neutral": r.neutral,
            "created_at": r.created_at.isoformat(),
        }
        for r in list_analyses(db, current_user.id)
    ]


@app.get("/my/overview")
def overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dashboard numbers for the logged-in user."""
    return overview_stats(db, current_user.id)