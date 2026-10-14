"""Reads uploaded .txt and .csv files safely."""
import io

import pandas as pd

from backend.services.validation import ValidationError

MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_CSV_ROWS = 500
ALLOWED_EXTENSIONS = {".txt", ".csv"}
COMMON_TEXT_COLUMNS = [
    "text", "comment", "review", "message", "content", "sentence", "feedback",
]


def get_extension(filename) -> str:
    """Return the lowercase file extension, e.g. '.csv' (or '' if none)."""
    filename = (filename or "").lower()
    dot = filename.rfind(".")
    return filename[dot:] if dot != -1 else ""


def decode_bytes(data: bytes) -> str:
    """Turn uploaded bytes into text, or raise ValidationError."""
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValidationError(
            f"File is too large. The limit is {MAX_FILE_BYTES // (1024 * 1024)} MB."
        )
    if b"\x00" in data:
        raise ValidationError("This does not look like a plain text file.")
    for encoding in ("utf-8-sig", "latin-1"):  # latin-1 never fails
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationError("Could not read the file. Please save it as UTF-8.")


def _choose_column(df: pd.DataFrame, requested) -> str:
    """Pick which CSV column holds the text."""
    columns = list(df.columns)
    lookup = {str(c).strip().lower(): c for c in columns}

    if requested:
        key = requested.strip().lower()
        if key in lookup:
            return lookup[key]
        raise ValidationError(
            f"Column '{requested}' was not found. "
            f"Columns in your file: {', '.join(map(str, columns))}."
        )

    for name in COMMON_TEXT_COLUMNS:
        if name in lookup:
            return lookup[name]
    if len(columns) == 1:
        return columns[0]

    raise ValidationError(
        "Could not tell which column contains the text. "
        f"Columns found: {', '.join(map(str, columns))}. "
        "Name a column 'text' or choose one."
    )


def parse_csv(content: str, column=None):
    """Return ([(row_number, raw_text), ...], column_used).

    Row 1 is the header, so the first data row is row 2.
    """
    try:
        df = pd.read_csv(
            io.StringIO(content),
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        )
    except pd.errors.EmptyDataError:
        raise ValidationError("The CSV file has no data.")
    except (pd.errors.ParserError, UnicodeError):
        raise ValidationError(
            "The CSV file is malformed. Make sure text containing "
            "commas is wrapped in double quotes."
        )

    if df.empty:
        raise ValidationError("The CSV file has a header but no rows.")
    if len(df) > MAX_CSV_ROWS:
        raise ValidationError(
            f"Too many rows ({len(df):,}). The limit is {MAX_CSV_ROWS}."
        )

    col = _choose_column(df, column)
    rows = [(i, value) for i, value in enumerate(df[col].tolist(), start=2)]
    return rows, str(col)