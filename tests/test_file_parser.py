import pytest

from backend.services.file_parser import (
    MAX_CSV_ROWS,
    MAX_FILE_BYTES,
    decode_bytes,
    get_extension,
    parse_csv,
)
from backend.services.validation import ValidationError


def test_get_extension():
    assert get_extension("notes.TXT") == ".txt"
    assert get_extension("data.csv") == ".csv"
    assert get_extension("noextension") == ""
    assert get_extension(None) == ""


def test_decode_empty_rejected():
    with pytest.raises(ValidationError):
        decode_bytes(b"")


def test_decode_too_large_rejected():
    with pytest.raises(ValidationError):
        decode_bytes(b"a" * (MAX_FILE_BYTES + 1))


def test_decode_binary_rejected():
    with pytest.raises(ValidationError):
        decode_bytes(b"\x00\x01\x02")


def test_decode_utf8_with_bom():
    assert decode_bytes("\ufeffhello".encode("utf-8")) == "hello"


def test_decode_latin1_fallback():
    assert decode_bytes(b"caf\xe9") == "café"


def test_csv_detects_text_column():
    rows, col = parse_csv("id,text\n1,I love it\n2,I hate it\n")
    assert col == "text"
    assert [r[1] for r in rows] == ["I love it", "I hate it"]


def test_csv_single_column_any_name():
    rows, col = parse_csv("feelings\nI am happy\nI am sad\n")
    assert col == "feelings"
    assert len(rows) == 2


def test_csv_ambiguous_columns_rejected():
    with pytest.raises(ValidationError) as exc:
        parse_csv("a,b\n1,2\n")
    assert "column" in str(exc.value).lower()


def test_csv_explicit_column():
    rows, col = parse_csv("a,b\nx,hello there\n", column="b")
    assert col == "b"
    assert rows[0][1] == "hello there"


def test_csv_unknown_column_rejected():
    with pytest.raises(ValidationError):
        parse_csv("a,b\n1,2\n", column="zzz")


def test_csv_header_only_rejected():
    with pytest.raises(ValidationError):
        parse_csv("text\n")


def test_csv_empty_rejected():
    with pytest.raises(ValidationError):
        parse_csv("")


def test_csv_too_many_rows_rejected():
    content = "text\n" + "hello\n" * (MAX_CSV_ROWS + 1)
    with pytest.raises(ValidationError):
        parse_csv(content)


def test_csv_row_numbers_start_at_two():
    rows, _ = parse_csv("text\nfirst\nsecond\n")
    assert [r[0] for r in rows] == [2, 3]