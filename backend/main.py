
"""FastAPI app: exposes the MoodMentor sentiment and emotion pipeline over HTTP."""

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
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
    description=(
        "AI-powered sentiment analysis and six-class emotion "
        "detection using fine-tuned DistilBERT."
    ),
    version="0.2.0",
)


def user_dict(user: User) -> dict:
    """The user data that is safe to send to the browser."""

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
    return {
        "message": "MoodMentor API is running. See /docs for the API."
    }


@app.get("/health")
def health():
    """Simple check that the server is running."""

    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Analyze pasted text and return the sentiment and emotion report."""

    try:
        return build_report(request.text)

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    column: str | None = Form(None),
):
    """Analyze an uploaded .txt or .csv file."""

    extension = get_extension(file.filename)

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .txt or .csv file.",
        )

    data = await file.read(MAX_FILE_BYTES + 1)

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
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/register", status_code=201)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Create a new user account."""

    try:
        user = register_user(
            db,
            request.username,
            request.email,
            request.password,
        )

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
    }


@app.post("/login")
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """Check credentials and return an access token."""

    try:
        user = authenticate_user(
            db,
            request.username,
            request.password,
        )

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
        },
    }


@app.get("/me")
def me(
    current_user: User = Depends(get_current_user),
):
    """Return the logged-in user's account and profile."""

    return user_dict(current_user)


@app.put("/me/profile")
def edit_profile(
    request: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the logged-in user's profile fields."""

    try:
        user = update_profile(
            db,
            current_user,
            request.model_dump(),
        )

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return user_dict(user)


@app.post("/my/analyze")
def analyze_and_save(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze text and save the result to the user's history."""

    try:
        report = build_report(request.text)

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    row = save_analysis(
        db,
        current_user.id,
        report,
    )

    report["saved_id"] = row.id

    return report


@app.post("/my/analyze-file")
async def analyze_file_and_save(
    file: UploadFile = File(...),
    column: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze an uploaded file and save the result to user history."""

    result = await analyze_file(
        file=file,
        column=column,
    )

    if result["file_type"] == "txt":

        save_analysis(
            db,
            current_user.id,
            result["report"],
            source="txt",
        )

        result["saved_entries"] = 1

    else:

        result["saved_entries"] = save_csv_rows(
            db,
            current_user.id,
            result["results"],
        )

    return result


@app.get("/my/history")
def history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the logged-in user's saved analyses, newest first."""

    return [
        {
            "id": row.id,
            "source": row.source,
            "text": row.input_text[:200],
            "label": row.label,
            "emotion": row.emotion,
            "compound": row.compound,
            "positive": row.positive,
            "negative": row.negative,
            "neutral": row.neutral,
            "created_at": row.created_at.isoformat(),
        }
        for row in list_analyses(db, current_user.id)
    ]


@app.get("/my/overview")
def overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return dashboard statistics for the logged-in user."""

    return overview_stats(db, current_user.id)
