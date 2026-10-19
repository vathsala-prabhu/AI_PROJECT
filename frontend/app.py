"""MoodMentor Streamlit UI (Milestone 1). Talks to the FastAPI backend."""
import json
import os

import pandas as pd
import requests
import streamlit as st

from styles import (
    auth_panel,
    eyebrow,
    inject_css,
    landing_grids,
    logo_html,
    page_header,
    score_bar,
)

API_URL = os.getenv("MOODMENTOR_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 30
MIX_COLORS = {"Positive": "#2E9E5B", "Neutral": "#9AA7C7", "Negative": "#E5675B"}
FREQUENCY_OPTIONS = ["Not set", "Daily", "A few times a week", "Weekly", "Occasionally"]
TIME_OPTIONS = ["Not set", "Morning", "Afternoon", "Evening", "Night", "No preference"]

st.set_page_config(
    page_title="MoodMentor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
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


def logout_click():
    log_out()
    go_to("Home")


inject_css(hide_sidebar=not is_logged_in())


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


def api_analyze_file(uploaded, column):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    data = {"column": column} if column else {}
    return requests.post(
        f"{API_URL}/my/analyze-file",
        files=files,
        data=data,
        headers=auth_headers(),
        timeout=TIMEOUT,
    )


def api_history():
    return requests.get(f"{API_URL}/my/history", headers=auth_headers(), timeout=TIMEOUT)


def api_overview():
    return requests.get(f"{API_URL}/my/overview", headers=auth_headers(), timeout=TIMEOUT)


def api_me():
    return requests.get(f"{API_URL}/me", headers=auth_headers(), timeout=TIMEOUT)


def api_update_profile(payload):
    return requests.put(
        f"{API_URL}/me/profile", json=payload, headers=auth_headers(), timeout=TIMEOUT
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
    with st.container(border=True):
        eyebrow("Your sentiment snapshot")
        st.markdown(
            f'<div class="big-label {sentiment["label"]}">{sentiment["label"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Compound score {sentiment['compound']:+.3f}")
        score_bar(sentiment["compound"])
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{sentiment['positive']:.3f}")
        c2.metric("Neutral", f"{sentiment['neutral']:.3f}")
        c3.metric("Negative", f"{sentiment['negative']:.3f}")


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
        f"(average compound {overall['average_compound']:+.3f})."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive rows", overall["counts"]["Positive"])
    c2.metric("Neutral rows", overall["counts"]["Neutral"])
    c3.metric("Negative rows", overall["counts"]["Negative"])
    c4.metric("Skipped rows", data["rows_skipped"])

    st.markdown("**Score for every row**")
    table = pd.DataFrame(data["results"])[
        ["row", "text", "label", "compound", "positive", "negative", "neutral"]
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)

    if data["skipped"]:
        with st.expander(f"Skipped rows ({len(data['skipped'])})"):
            st.dataframe(pd.DataFrame(data["skipped"]), hide_index=True)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(data, indent=2),
        file_name="moodmentor_csv_report.json",
        mime="application/json",
    )


# ---------- Public pages ----------
def topbar():
    c1, c2, c3 = st.columns([6, 1.2, 1.4])
    c1.markdown(logo_html(), unsafe_allow_html=True)
    c2.button("Log in", key="top_login", on_click=go_to, args=("Login",), use_container_width=True)
    c3.button(
        "Register", key="top_register", type="primary",
        on_click=go_to, args=("Register",), use_container_width=True,
    )


def page_home():
    st.markdown(
        '<div class="headline"><h1>How are you really feeling?</h1>'
        "<p>Write it down. MoodMentor shows you the tone of your words.</p></div>",
        unsafe_allow_html=True,
    )
    landing_grids()
    st.markdown(
        '<div class="footnote">For reflection and general wellness only. '
        "Not a clinical or diagnostic tool.</div>",
        unsafe_allow_html=True,
    )


def page_login():
    left, right = st.columns(2, gap="large")
    with left:
        st.write("")
        eyebrow("Welcome back")
        st.markdown('<div class="page-title">Sign in to your space</div>', unsafe_allow_html=True)
        st.caption("Your entries and scores are waiting.")
        flash = st.session_state.pop("flash", None)
        if flash:
            st.info(flash)

        with st.form("login_form"):
            username = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in →", type="primary", use_container_width=True)

        if submitted:
            data = call_api(api_login, username, password)
            if data:
                st.session_state["token"] = data["access_token"]
                st.session_state["user"] = data["user"]
                go_to("Overview")
                st.rerun()

        st.button("New to MoodMentor? Create account", key="login_to_register",
                  on_click=go_to, args=("Register",))
    with right:
        auth_panel()


def page_register():
    left, right = st.columns(2, gap="large")
    with left:
        st.write("")
        eyebrow("Join MoodMentor")
        st.markdown('<div class="page-title">Create your account</div>', unsafe_allow_html=True)
        st.caption("It takes less than a minute.")

        with st.form("register_form"):
            username = st.text_input("Username", help="3-30 letters, numbers or underscore")
            email = st.text_input("Email address")
            password = st.text_input("Password", type="password", help="At least 8 characters")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account →", type="primary", use_container_width=True)

        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            elif call_api(api_register, username, email, password):
                st.session_state["flash"] = "Account created. Please sign in."
                go_to("Login")
                st.rerun()

        st.button("Already have an account? Sign in", key="register_to_login",
                  on_click=go_to, args=("Login",))
    with right:
        auth_panel()


# ---------- Logged-in pages ----------
def page_overview():
    username = st.session_state["user"]["username"]
    page_header("Your overview", f"Welcome, {username} 👋",
                "Make space for your thoughts and see how your writing feels.")
    data = call_api(api_overview)
    if data is None:
        return

    total = data["total"]
    counts = data["counts"]
    avg = data["average_compound"]
    most_common = max(counts, key=counts.get) if total else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Journal entries", total)
    c2.metric("Average sentiment", "—" if avg is None else f"{avg:+.2f}")
    c3.metric("Most common tone", most_common)
    st.write("")

    if total == 0:
        st.info("No entries yet. Add your first check-in to see your charts.")
        st.button("＋ New check-in", type="primary", on_click=go_to, args=("Analyze text",))
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Your emotional rhythm**")
            st.caption("Compound score of your latest entries, oldest to newest.")
            rhythm = pd.DataFrame(
                {"Compound score": [r["compound"] for r in data["recent"]]},
                index=range(1, len(data["recent"]) + 1),
            )
            st.line_chart(rhythm, height=280)
    with right:
        with st.container(border=True):
            st.markdown("**Your sentiment mix**")
            st.caption("Share of Positive, Neutral and Negative entries.")
            mix = pd.DataFrame(
                [{"label": k, "count": v} for k, v in counts.items() if v > 0]
            )
            st.vega_lite_chart(
                mix,
                {
                    "mark": {"type": "arc", "innerRadius": 60},
                    "encoding": {
                        "theta": {"field": "count", "type": "quantitative"},
                        "color": {
                            "field": "label",
                            "type": "nominal",
                            "scale": {
                                "domain": list(MIX_COLORS),
                                "range": list(MIX_COLORS.values()),
                            },
                            "legend": {"title": None, "orient": "bottom"},
                        },
                    },
                    "height": 260,
                },
                use_container_width=True,
            )
    st.button("＋ New check-in", type="primary", on_click=go_to, args=("Analyze text",))


def page_analyze():
    page_header("Sentiment analyzer", "How are you feeling today?",
                "Type your text, or upload a .txt or .csv file.")
    tab_text, tab_txt, tab_csv = st.tabs(["✍️ Type text", "📄 Upload .txt", "📊 Upload .csv"])

    with tab_text:
        with st.container(border=True):
            text = st.text_area(
                "Your check-in",
                height=200,
                max_chars=50_000,
                placeholder="Today I felt...",
            )
            clicked = st.button("Analyze my text", type="primary", key="analyze_text_btn")
        if clicked:
            report = call_api(api_analyze_text, text)
            if report:
                st.caption("✅ Saved to your journal.")
                show_text_report(report)

    with tab_txt:
        uploaded = st.file_uploader("Choose a .txt file", type=["txt"], key="txt_upload")
        if st.button("Analyze .txt file", type="primary", key="analyze_txt_btn"):
            if uploaded is None:
                st.warning("Please choose a file first.")
            else:
                data = call_api(api_analyze_file, uploaded, None)
                if data:
                    st.caption("✅ Saved to your journal.")
                    show_text_report(data["report"])

    with tab_csv:
        st.caption("Every row is scored on its own. Max 2 MB and 500 rows.")
        uploaded = st.file_uploader("Choose a .csv file", type=["csv"], key="csv_upload")
        column = st.text_input(
            "Column with the text (optional)",
            help="Leave blank to auto-detect a column named text, comment, review, etc.",
        ).strip() or None
        if st.button("Analyze .csv file", type="primary", key="analyze_csv_btn"):
            if uploaded is None:
                st.warning("Please choose a file first.")
            else:
                data = call_api(api_analyze_file, uploaded, column)
                if data:
                    st.caption(f"✅ {data['saved_entries']} rows saved to your journal.")
                    show_csv_report(data)


def page_journal():
    page_header("Your saved entries", "My journal")
    rows = call_api(api_history)
    if rows is None:
        return
    if not rows:
        st.info("No saved entries yet. Analyze some text first.")
        return
    table = pd.DataFrame(rows)[
        ["created_at", "source", "text", "label", "compound", "positive", "negative", "neutral"]
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)


def _index(options, value):
    return options.index(value) if value in options else 0


def page_profile():
    page_header("Your account", "My profile",
                "Tell MoodMentor a little about you and how you like to journal.")
    me = call_api(api_me)
    if me is None:
        return

    flash = st.session_state.pop("profile_flash", None)
    if flash:
        st.success(flash)

    with st.container(border=True):
        st.markdown(f"### {me.get('full_name') or me['username']}")
        st.caption(f"@{me['username']}  ·  {me['email']}")
        if me.get("bio"):
            st.write(me["bio"])

    with st.container(border=True):
        st.markdown("**Edit your details**")
        with st.form("profile_form"):
            full_name = st.text_input("Full name", value=me.get("full_name") or "", max_chars=100)
            bio = st.text_area(
                "About me", value=me.get("bio") or "", max_chars=500, height=110,
                placeholder="A few words about you...",
            )
            goal = st.text_area(
                "What I want from journaling", value=me.get("journaling_goal") or "",
                max_chars=300, height=90,
                placeholder="e.g. Notice my mood, feel calmer, build a daily habit",
            )
            c1, c2 = st.columns(2)
            frequency = c1.selectbox(
                "How often I want to write", FREQUENCY_OPTIONS,
                index=_index(FREQUENCY_OPTIONS, me.get("journaling_frequency")),
            )
            best_time = c2.selectbox(
                "Best time to write", TIME_OPTIONS,
                index=_index(TIME_OPTIONS, me.get("preferred_time")),
            )
            saved = st.form_submit_button("Save changes", type="primary")

    if saved:
        payload = {
            "full_name": full_name,
            "bio": bio,
            "journaling_goal": goal,
            "journaling_frequency": "" if frequency == "Not set" else frequency,
            "preferred_time": "" if best_time == "Not set" else best_time,
        }
        if call_api(api_update_profile, payload):
            st.session_state["profile_flash"] = "Profile updated."
            st.rerun()

    st.write("")
    st.button("Log out", on_click=logout_click)


# ---------- Navigation ----------
PUBLIC_PAGES = {"Home": page_home, "Login": page_login, "Register": page_register}
USER_PAGES = {
    "Overview": page_overview,
    "Analyze text": page_analyze,
    "My journal": page_journal,
    "My profile": page_profile,
}
NAV_ICONS = {"Overview": "🏠", "Analyze text": "✨", "My journal": "📖", "My profile": "👤"}

pages = USER_PAGES if is_logged_in() else PUBLIC_PAGES
default_page = "Overview" if is_logged_in() else "Home"

if "goto" in st.session_state:
    target = st.session_state.pop("goto")
    if target in pages:
        st.session_state["page"] = target
if st.session_state.get("page") not in pages:
    st.session_state["page"] = default_page
current = st.session_state["page"]

if is_logged_in():
    with st.sidebar:
        st.markdown(logo_html(), unsafe_allow_html=True)
        st.markdown('<div class="side-label">YOUR SPACE</div>', unsafe_allow_html=True)
        for name in USER_PAGES:
            st.button(
                f"{NAV_ICONS[name]}   {name}",
                key=f"nav_{name}",
                on_click=go_to,
                args=(name,),
                type="primary" if name == current else "secondary",
                use_container_width=True,
            )
        st.divider()
        st.caption(f"Signed in as **{st.session_state['user']['username']}**")
        st.button("Log out", key="side_logout", on_click=logout_click, use_container_width=True)
else:
    topbar()

pages[current]()