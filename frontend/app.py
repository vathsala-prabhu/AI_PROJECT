"""MoodMentor Streamlit UI (Milestone 1). Talks to the FastAPI backend."""
import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("MOODMENTOR_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 30
ICONS = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

st.set_page_config(
    page_title="MoodMentor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Login state ----------
def is_logged_in():
    return bool(st.session_state.get("token"))


def log_out():
    st.session_state.pop("token", None)
    st.session_state.pop("user", None)


def go_to(page):
    """Ask the app to switch page on the next run."""
    st.session_state["goto"] = page


# ---------- API helpers ----------
def auth_headers():
    return {"Authorization": f"Bearer {st.session_state['token']}"}


def api_register(username, email, password):
    return requests.post(
        f"{API_URL}/register",
        json={"username": username, "email": email, "password": password},
        timeout=TIMEOUT,
    )


def api_login(username, password):
    return requests.post(
        f"{API_URL}/login",
        json={"username": username, "password": password},
        timeout=TIMEOUT,
    )


def api_analyze_text(text):
    return requests.post(
        f"{API_URL}/my/analyze",
        json={"text": text},
        headers=auth_headers(),
        timeout=TIMEOUT,
    )


def api_history():
    return requests.get(
        f"{API_URL}/my/history", headers=auth_headers(), timeout=TIMEOUT
    )


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

    if response.status_code in (200, 201):
        return response.json()

    try:
        detail = response.json().get("detail", "Something went wrong.")
    except ValueError:
        detail = "Something went wrong."
    if not isinstance(detail, str):
        detail = "Invalid request. Please check your input."

    # A 401 on a logged-in call means the token expired: log out cleanly.
    if response.status_code == 401 and is_logged_in():
        log_out()
        go_to("Login")
        st.session_state["flash"] = "Your session expired. Please log in again."
        st.rerun()

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


def require_login():
    """Show a message and return False if the user is not logged in."""
    if is_logged_in():
        return True
    st.warning("Please log in to use this page.")
    st.button("Go to Login", on_click=go_to, args=("Login",))
    return False


# ---------- Pages ----------
def page_home():
    st.title("🧠 MoodMentor")
    st.subheader("AI-powered emotional tone analysis")
    st.write(
        "MoodMentor reads your writing and tells you whether its tone is "
        "positive, negative or neutral, using the VADER sentiment model."
    )

    if is_logged_in():
        st.success(f"Welcome back, {st.session_state['user']['username']}!")
        st.button("Analyze some text", type="primary", on_click=go_to, args=("Analyze Text",))
    else:
        c1, c2, _ = st.columns([1, 1, 4])
        c1.button("Login", key="home_login", type="primary", on_click=go_to, args=("Login",))
        c2.button("Register", key="home_register", on_click=go_to, args=("Register",))

    st.markdown(
        """
**What you can do**
- Create an account and log in
- Paste text and get a sentiment report, saved to your history
- Upload a `.txt` file (one report) or a `.csv` file (per-row results and overall summary)
- See the preprocessing steps: cleaning, tokenization, stop-word removal, lemmatization

**How to read the scores**
- **Compound** runs from -1 (very negative) to +1 (very positive)
- Positive if compound ≥ 0.05, Negative if ≤ -0.05, otherwise Neutral
- Positive, Negative and Neutral show the share of the text in each category
"""
    )


def page_register():
    st.title("Create an account")
    with st.form("register_form"):
        username = st.text_input("Username", help="3-30 letters, numbers or underscore")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", help="At least 8 characters")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register", type="primary")

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
        elif call_api(api_register, username, email, password):
            st.session_state["flash"] = "Account created. Please log in."
            go_to("Login")
            st.rerun()

    st.button(
        "Already have an account? Log in",
        key="register_to_login",
        on_click=go_to,
        args=("Login",),
    )


def page_login():
    st.title("Log in")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.info(flash)

    with st.form("login_form"):
        username = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        data = call_api(api_login, username, password)
        if data:
            st.session_state["token"] = data["access_token"]
            st.session_state["user"] = data["user"]
            go_to("Analyze Text")
            st.rerun()

    st.button(
        "No account yet? Register",
        key="login_to_register",
        on_click=go_to,
        args=("Register",),
    )


def page_text():
    st.title("Analyze Text")
    if not require_login():
        return
    text = st.text_area(
        "Enter or paste your text",
        height=200,
        max_chars=50_000,
        placeholder="e.g. I am not happy with the results.",
    )
    if st.button("Analyze", type="primary"):
        report = call_api(api_analyze_text, text)
        if report:
            st.caption("✅ Saved to your history.")
            show_text_report(report)


def page_upload():
    st.title("Upload File")
    if not require_login():
        return
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


def page_history():
    st.title("My History")
    if not require_login():
        return
    rows = call_api(api_history)
    if rows is None:
        return
    if not rows:
        st.info("No saved analyses yet. Analyze some text first.")
        return
    table = pd.DataFrame(rows)[
        ["created_at", "text", "label", "compound", "positive", "negative", "neutral"]
    ]
    st.dataframe(table, hide_index=True)


# ---------- Navigation (top button bar) ----------
PUBLIC_PAGES = {"Home": page_home, "Login": page_login, "Register": page_register}
USER_PAGES = {
    "Home": page_home,
    "Analyze Text": page_text,
    "Upload File": page_upload,
    "My History": page_history,
}
pages = USER_PAGES if is_logged_in() else PUBLIC_PAGES

# Apply any page switch requested by a button or by code (go_to).
if "goto" in st.session_state:
    target = st.session_state.pop("goto")
    if target in pages:
        st.session_state["page"] = target
if st.session_state.get("page") not in pages:
    st.session_state["page"] = "Home"


def nav_click(name):
    st.session_state["page"] = name


def logout_click():
    log_out()
    st.session_state["page"] = "Home"


current = st.session_state["page"]
extra = 2 if is_logged_in() else 0  # room for the username and Log out
cols = st.columns(len(pages) + extra)

for col, name in zip(cols, pages):
    col.button(
        name,
        key=f"nav_{name}",
        on_click=nav_click,
        args=(name,),
        type="primary" if name == current else "secondary",
        use_container_width=True,
    )

if is_logged_in():
    cols[-2].caption(f"👤 {st.session_state['user']['username']}")
    cols[-1].button("Log out", key="nav_logout", on_click=logout_click, use_container_width=True)

st.divider()
pages[current]()