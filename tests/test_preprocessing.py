from backend.services.preprocessing import clean_text, preprocess


def test_clean_removes_url_and_punctuation():
    result = clean_text("I LOVE this!!! Visit https://example.com now.")
    assert "http" not in result
    assert "!" not in result
    assert result == result.lower()


def test_clean_removes_html_and_numbers():
    assert clean_text("<b>Day 5</b> was fine") == "day was fine"


def test_clean_collapses_whitespace():
    assert clean_text("too     many    spaces") == "too many spaces"


def test_stopwords_removed():
    result = preprocess("This is a really great day")
    assert "is" not in result["tokens_no_stopwords"]
    assert "great" in result["tokens_no_stopwords"]
    assert "day" in result["tokens_no_stopwords"]


def test_lemmatization():
    result = preprocess("The children were playing")
    assert "child" in result["lemmas"]


def test_empty_text_returns_empty_lists():
    result = preprocess("")
    assert result["tokens"] == []
    assert result["lemmas"] == []


def test_only_symbols_returns_empty_lists():
    result = preprocess("!!! ??? 123")
    assert result["tokens"] == []