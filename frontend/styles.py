"""Look and feel for MoodMentor: CSS plus small HTML helpers."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

.stApp { background: linear-gradient(160deg, #F5F8FF 0%, #EEF3FF 60%, #F9F5FF 100%); }
.block-container { max-width: 1150px; padding-top: 1.2rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3EAFB; }
[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start; text-align: left; border: none; background: transparent;
  color: #3A4668; box-shadow: none; padding: .6rem .9rem;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #F0F4FF; transform: none; box-shadow: none; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #E8EFFF; color: #2F58C8; box-shadow: none;
}
.side-label { letter-spacing: .14em; font-size: .7rem; font-weight: 700; color: #8A97BB; margin: 1.2rem 0 .4rem .3rem; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 600; border: 1px solid #D5DEF5;
  padding: .55rem 1rem; transition: all .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,111,224,.18); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, #3B6FE0, #5B8CFF); color: #fff; border: none;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #fff; border-radius: 18px; border: 1px solid #E3EAFB;
  box-shadow: 0 6px 24px rgba(40,70,140,.07); padding: .4rem;
}
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #E3EAFB; border-radius: 16px; padding: 1rem 1.2rem;
  box-shadow: 0 6px 24px rgba(40,70,140,.06);
}

/* Logo */
.logo { display: flex; align-items: center; gap: .6rem; font-weight: 700; font-size: 1.25rem; color: #1B2540; }
.logo-mark {
  width: 38px; height: 38px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #2F58C8, #5B8CFF); color: #fff; font-size: 1.1rem;
}

/* Headings */
.eyebrow { letter-spacing: .14em; font-size: .72rem; font-weight: 700; color: #6B7BA8; text-transform: uppercase; }
.page-title { font-size: 2.2rem; font-weight: 800; color: #1B2540; margin: .1rem 0 .2rem 0; line-height: 1.15; }
.page-sub { color: #5C6A8F; margin-bottom: 1.2rem; }

/* Landing */
.headline { text-align: center; margin: 1.6rem 0 1.4rem 0; }
.headline h1 { font-size: 2.6rem; font-weight: 800; color: #1B2540; margin: 0; }
.headline p { color: #5C6A8F; font-size: 1.1rem; margin: .4rem 0 0 0; }
.grid-title { letter-spacing: .14em; font-size: .72rem; font-weight: 700; color: #6B7BA8; margin: 1.6rem 0 .7rem 0; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; }
.fcard {
  background: #fff; border: 1px solid #E3EAFB; border-radius: 20px; padding: 1.5rem;
  box-shadow: 0 6px 24px rgba(40,70,140,.07); transition: all .18s ease;
}
.fcard:hover { transform: translateY(-4px); box-shadow: 0 14px 34px rgba(40,70,140,.14); }
.ficon {
  width: 46px; height: 46px; border-radius: 14px; background: #E8EFFF; display: flex;
  align-items: center; justify-content: center; font-size: 1.3rem; margin-bottom: .9rem;
}
.fcard.alt .ficon { background: #E3F7EA; }
.fcard h3 { margin: 0 0 .35rem 0; font-size: 1.08rem; color: #1B2540; }
.fcard p { margin: 0; color: #5C6A8F; font-size: .93rem; line-height: 1.5; }
.footnote { text-align: center; color: #8A97BB; font-size: .82rem; margin: 2rem 0 .5rem 0; }

/* Split-screen auth panel */
.auth-panel {
  background: linear-gradient(150deg, #2447B0 0%, #3B6FE0 55%, #6D8CFF 100%); color: #fff;
  border-radius: 26px; padding: 2.6rem 2.2rem; min-height: 520px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.auth-panel .heart {
  width: 52px; height: 52px; border-radius: 16px; background: rgba(255,255,255,.18);
  display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
}
.auth-panel h2 { color: #fff; font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 1.2rem 0 .6rem 0; }
.auth-panel p { color: #DDE6FF; margin: 0; }
.quote { background: rgba(255,255,255,.14); border-radius: 16px; padding: 1rem 1.2rem; margin-top: 2rem; }
.quote small { letter-spacing: .12em; font-weight: 700; color: #C9D6FF; }
.quote div { font-size: 1.15rem; font-weight: 600; margin-top: .3rem; }

/* Sentiment result */
.big-label { font-size: 2.4rem; font-weight: 800; margin: .1rem 0 0 0; }
.big-label.Positive { color: #1E7D45; }
.big-label.Negative { color: #B3261E; }
.big-label.Neutral  { color: #4A5578; }
.scorebar { position: relative; height: 14px; border-radius: 999px; margin: 1rem 0 .3rem 0;
  background: linear-gradient(90deg, #E5675B 0%, #D9DEEC 50%, #2E9E5B 100%); }
.scorebar .marker { position: absolute; top: -5px; width: 24px; height: 24px; margin-left: -12px;
  border-radius: 50%; background: #fff; border: 4px solid #1B2540; box-shadow: 0 2px 8px rgba(0,0,0,.25); }
.scale { display: flex; justify-content: space-between; font-size: .75rem; color: #8A97BB; }

@media (max-width: 900px) { .grid3 { grid-template-columns: 1fr; } }
</style>
"""

# Logged out: no sidebar on the public pages.
HIDE_SIDEBAR = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
</style>
"""

# Logged in: force the sidebar open.
SHOW_SIDEBAR = """
<style>
[data-testid="stSidebar"] {
  display: flex !important; visibility: visible !important;
  transform: none !important; min-width: 280px !important;
}
</style>
"""

ABOUT_CARDS = [
    ("🧠", "Reflect with intention",
     "A private space to put your thoughts into words and see how they read."),
    ("📊", "Understand your tone",
     "VADER sentiment analysis tells you if your writing leans positive, negative or neutral."),
    ("🔒", "Private by design",
     "Your account is password protected and only you can see your saved entries."),
]
FEATURE_CARDS = [
    ("✍️", "Write or paste text",
     "Type how you feel and get an instant sentiment report."),
    ("📄", "Upload .txt or .csv files",
     "Analyze a whole journal file, or every row of a CSV one by one."),
    ("🎯", "Get clear scores",
     "See compound, positive, negative and neutral scores with a simple label."),
]


def inject_css(hide_sidebar: bool = False):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HIDE_SIDEBAR if hide_sidebar else SHOW_SIDEBAR, unsafe_allow_html=True)


def logo_html():
    return '<div class="logo"><div class="logo-mark">💙</div>MoodMentor</div>'


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def page_header(eyebrow_text, title, subtitle=""):
    html = f'<div class="eyebrow">{eyebrow_text}</div><div class="page-title">{title}</div>'
    if subtitle:
        html += f'<div class="page-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def _cards(items, alt=False):
    css = "fcard alt" if alt else "fcard"
    return "".join(
        f'<div class="{css}"><div class="ficon">{icon}</div><h3>{title}</h3><p>{text}</p></div>'
        for icon, title, text in items
    )


def landing_grids():
    st.markdown(
        '<div class="grid-title">WHAT MOODMENTOR IS ABOUT</div>'
        f'<div class="grid3">{_cards(ABOUT_CARDS)}</div>'
        '<div class="grid-title">WHAT YOU CAN DO</div>'
        f'<div class="grid3">{_cards(FEATURE_CARDS, alt=True)}</div>',
        unsafe_allow_html=True,
    )


def auth_panel():
    st.markdown(
        '<div class="auth-panel"><div>'
        '<div class="heart">💙</div>'
        '<h2>Make room for how you really feel.</h2>'
        '<p>A simple place to see the tone of your words and keep your entries together.</p>'
        '</div>'
        '<div class="quote"><small>✦ YOUR DAILY REMINDER</small>'
        '<div>“Progress begins with noticing.”</div></div></div>',
        unsafe_allow_html=True,
    )


def score_bar(compound: float):
    pct = (compound + 1) / 2 * 100
    st.markdown(
        f'<div class="scorebar"><div class="marker" style="left:{pct:.1f}%"></div></div>'
        '<div class="scale"><span>-1 Negative</span><span>0 Neutral</span><span>+1 Positive</span></div>',
        unsafe_allow_html=True,
    )