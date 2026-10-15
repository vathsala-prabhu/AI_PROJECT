"""MoodMentor Streamlit UI (Milestone 1). Talks to the FastAPI backend."""
import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("MOODMENTOR_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 30
ICONS = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

st.set_page_config(page_title="MoodMentor", page_icon="🧠", layout="wide")


# ---------- API helpers ----------
def api_analyze_text(text):
    return requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=TIMEOUT)


def api_analyze_file(uploaded, column):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    data = {"column": column} if column else {}
    return requests.post(
        f"{API_URL}/analyze-file", files=files, data=data, timeout=TIMEOUT
    )


def call_api(func, *args):
    """Call the backend; show a friendly error and return None on failure."""
    try:
        response = func(*args)
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the MoodMentor API. Start the backend with: "
            "python -m uvicorn backend.main:app --reload"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again.")
        return None

    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", "Something went wrong.")
    except ValueError:
        detail = "Something went wrong."
    if not isinstance(detail, str):
        detail = "Invalid request. Please check your input."
    st.error(detail)
    return None


# ---------- Display helpers ----------
def show_scores(sentiment):
    st.subheader(f"{ICONS[sentiment['label']]} {sentiment['label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compound", f"{sentiment['compound']:.3f}")
    c2.metric("Positive", f"{sentiment['positive']:.3f}")
    c3.metric("Negative", f"{sentiment['negative']:.3f}")
    c4.metric("Neutral", f"{sentiment['neutral']:.3f}")
    chart = pd.DataFrame(
        {"score": [sentiment["positive"], sentiment["neutral"], sentiment["negative"]]},
        index=["Positive", "Neutral", "Negative"],
    )
    st.bar_chart(chart)


def show_text_report(report):
    st.success(report["summary"])
    st.caption(
        f"{report['word_count']} words · {report['char_count']} characters · "
        f"generated {report['generated_at']}"
    )
    show_scores(report["sentiment"])

    with st.expander("Preprocessing steps (demonstration)"):
        st.caption(
            "VADER analyzed the ORIGINAL text, so negation like 'not happy' "
            "is kept. These steps are shown separately."
        )
        pre = report["preprocessing"]
        st.markdown("**Cleaned text**")
        st.code(pre["cleaned_text"] or "(empty)")
        st.markdown("**Tokens**")
        st.write(pre["tokens"])
        st.markdown("**Tokens without stop-words**")
        st.write(pre["tokens_no_stopwords"])
        st.markdown("**Lemmas**")
        st.write(pre["lemmas"])

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="moodmentor_report.json",
        mime="application/json",
    )


def show_csv_report(data):
    overall = data["overall"]
    st.success(
        f"Analyzed {data['rows_analyzed']} rows from column "
        f"'{data['column_used']}'. Overall: {overall['label']} "
        f"(average compound {overall['average_compound']:.3f})."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Positive", overall["counts"]["Positive"])
    c2.metric("⚪ Neutral", overall["counts"]["Neutral"])
    c3.metric("🔴 Negative", overall["counts"]["Negative"])
    c4.metric("Skipped rows", data["rows_skipped"])

    st.bar_chart(pd.DataFrame({"rows": overall["counts"]}))

    st.markdown("**Per-row results**")
    table = pd.DataFrame(data["results"])[
        ["row", "text", "label", "compound", "positive", "negative", "neutral"]
    ]
    st.dataframe(table, hide_index=True)

    if data["skipped"]:
        with st.expander(f"Skipped rows ({len(data['skipped'])})"):
            st.dataframe(pd.DataFrame(data["skipped"]), hide_index=True)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(data, indent=2),
        file_name="moodmentor_csv_report.json",
        mime="application/json",
    )


# ---------- Pages ----------
def page_home():
    st.title("🧠 MoodMentor")
    st.subheader("AI-powered emotional tone analysis")
    st.write(
        "MoodMentor reads your writing and tells you whether its tone is "
        "positive, negative or neutral, using the VADER sentiment model."
    )
    st.markdown(
        """
**What you can do in Milestone 1**
- Paste text and get a sentiment report
- Upload a `.txt` file (one report) or a `.csv` file (per-row results and overall summary)
- See the preprocessing steps: cleaning, tokenization, stop-word removal, lemmatization

**How to read the scores**
- **Compound** runs from -1 (very negative) to +1 (very positive)
- Positive if compound ≥ 0.05, Negative if ≤ -0.05, otherwise Neutral
- Positive, Negative and Neutral show the share of the text in each category
"""
    )
    st.info("Choose **Analyze Text** or **Upload File** in the sidebar to begin.")


def page_text():
    st.title("Analyze Text")
    text = st.text_area(
        "Enter or paste your text",
        height=200,
        max_chars=50_000,
        placeholder="e.g. I am not happy with the results.",
    )
    if st.button("Analyze", type="primary"):
        report = call_api(api_analyze_text, text)
        if report:
            show_text_report(report)


def page_upload():
    st.title("Upload File")
    st.caption("Supported: .txt (analyzed as one text) and .csv (one row at a time). Max 2 MB.")
    uploaded = st.file_uploader("Choose a file", type=["txt", "csv"])

    column = None
    if uploaded is not None and uploaded.name.lower().endswith(".csv"):
        column = st.text_input(
            "CSV column with the text (optional)",
            help="Leave blank to auto-detect a column named text, comment, review, etc.",
        ).strip() or None

    if st.button("Analyze file", type="primary"):
        if uploaded is None:
            st.warning("Please choose a file first.")
            return
        data = call_api(api_analyze_file, uploaded, column)
        if data:
            if data["file_type"] == "txt":
                show_text_report(data["report"])
            else:
                show_csv_report(data)


# ---------- Navigation ----------
PAGES = {
    "Home": page_home,
    "Analyze Text": page_text,
    "Upload File": page_upload,
}
st.sidebar.title("MoodMentor")
choice = st.sidebar.radio("Go to", list(PAGES))
PAGES[choice]()