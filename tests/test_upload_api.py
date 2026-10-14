from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def upload(name, data, content_type="text/plain", **form):
    return client.post(
        "/analyze-file",
        files={"file": (name, data, content_type)},
        data=form,
    )


def test_txt_upload_ok():
    r = upload("note.txt", b"I am not happy with the results.")
    assert r.status_code == 200
    body = r.json()
    assert body["file_type"] == "txt"
    assert body["report"]["sentiment"]["label"] == "Negative"


def test_csv_upload_ok():
    r = upload("data.csv", b"text\nI love it\nI hate it\n", "text/csv")
    assert r.status_code == 200
    body = r.json()
    assert body["file_type"] == "csv"
    assert body["rows_analyzed"] == 2
    assert body["overall"]["counts"]["Positive"] == 1
    assert body["overall"]["counts"]["Negative"] == 1


def test_csv_skips_blank_rows():
    r = upload("data.csv", b'text\nI love it\n"   "\nI hate it\n', "text/csv")
    assert r.status_code == 200
    body = r.json()
    assert body["rows_analyzed"] == 2
    assert body["rows_skipped"] == 1
    assert body["skipped"][0]["row"] == 3


def test_unsupported_extension():
    r = upload("doc.pdf", b"hello")
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"].lower()


def test_empty_file():
    r = upload("empty.txt", b"")
    assert r.status_code == 400


def test_binary_file():
    r = upload("bad.txt", b"\x00\x01\x02")
    assert r.status_code == 400


def test_csv_ambiguous_columns():
    r = upload("data.csv", b"a,b\n1,2\n", "text/csv")
    assert r.status_code == 400
    assert "column" in r.json()["detail"].lower()


def test_csv_with_column_choice():
    r = upload("data.csv", b"a,b\nx,I love this\n", "text/csv", column="b")
    assert r.status_code == 200
    assert r.json()["column_used"] == "b"


def test_missing_file_returns_422():
    r = client.post("/analyze-file")
    assert r.status_code == 422