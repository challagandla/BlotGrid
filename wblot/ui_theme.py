"""BlotGrid visual theme.

Colours, type and spacing follow the Linear DESIGN.md from
VoltAgent/awesome-design-md (MIT): a near-black canvas, a four-step charcoal
surface ladder with hairline borders, one lavender accent (#5e6ad2) used only
for primary actions, focus and selection, 8 px control radii and 4 px spacing.
The 4 px scale and the soft light-mode shadows follow the OpenUI react-ui
theme tokens (thesysdev/openui, MIT).

Base colours live in ``.streamlit/config.toml`` (``[theme.dark]`` and
``[theme.light]``); this module only adds what Streamlit's config cannot
express. Everything is local, so no web fonts or other network requests.
"""
from __future__ import annotations

FONT_STACK = (
    'Inter, "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    "system-ui, Roboto, sans-serif"
)
MONO_STACK = (
    '"JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace'
)

# Linear tokens (DESIGN.md front matter).
LINEAR = {
    "primary": "#5e6ad2",
    "primary_hover": "#828fff",
    "primary_focus": "#5e69d1",
    "canvas": "#010102",
    "surface_1": "#0f1011",
    "surface_2": "#141516",
    "surface_3": "#18191a",
    "hairline": "#23252a",
    "hairline_strong": "#34343a",
    "ink": "#f7f8f8",
    "ink_muted": "#d0d6e0",
    "ink_subtle": "#8a8f98",
}

CSS = """
<style>
:root {
  --bg-accent: #5e6ad2;
  --bg-accent-hover: #828fff;
  --bg-radius-sm: 6px;
  --bg-radius-md: 8px;
  --bg-radius-lg: 12px;
}

/* Typography: Linear's measured negative tracking on headings. */
h1, h2, h3 { font-weight: 600 !important; }
h1 { letter-spacing: -0.6px; }
h2 { letter-spacing: -0.4px; }
h3 { letter-spacing: -0.2px; }
[data-testid="stCaptionContainer"], .stCaption { letter-spacing: 0; }

/* Compact chrome: a desktop tool, not a marketing page. */
[data-testid="stAppViewContainer"] .block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1680px;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* Tabs read as a step bar: muted labels, accent underline on the active step. */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--bg-hairline, rgba(128,128,128,.25)); }
[data-baseweb="tab"] {
  height: 38px;
  padding: 0 14px;
  border-radius: var(--bg-radius-md) var(--bg-radius-md) 0 0;
  font-weight: 500;
  font-size: 14px;
  opacity: .72;
}
[data-baseweb="tab"][aria-selected="true"] { opacity: 1; }
[data-baseweb="tab-highlight"] { background-color: var(--bg-accent) !important; height: 2px; }

/* Buttons: Linear compact spec, 8px radius, 8x14 padding, weight 500. */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: var(--bg-radius-md);
  padding: 6px 14px;
  font-weight: 500;
  font-size: 14px;
  transition: background-color .12s ease, border-color .12s ease;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover { background-color: var(--bg-accent-hover); border-color: var(--bg-accent-hover); }

/* Focus ring: 2px lavender at 50% opacity, on every focusable control. */
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible, [data-baseweb="select"]:focus-within {
  outline: 2px solid rgba(94, 105, 209, .5) !important;
  outline-offset: 1px;
}

/* Panels: surface lift with hairline borders, no heavy shadows. */
[data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  border-radius: var(--bg-radius-lg);
}
[data-testid="stExpander"] summary { font-weight: 500; }
[data-testid="stAlert"] { border-radius: var(--bg-radius-md); }
/* Info notes are neutral surfaces with an accent rule: Linear allows one
   chromatic accent, so blue info boxes become charcoal/grey panels. Warnings
   and errors keep their semantic colours. */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
  background: rgba(128, 128, 128, .08);
  border: 1px solid rgba(128, 128, 128, .22);
  border-left: 2px solid var(--bg-accent);
  color: inherit;
}
[data-testid="stAlertContentInfo"] { color: inherit; }
[data-testid="stMetric"] {
  padding: 12px 16px;
  border-radius: var(--bg-radius-lg);
  border: 1px solid rgba(128,128,128,.22);
}
[data-testid="stDataFrame"], [data-testid="stImage"] img { border-radius: var(--bg-radius-md); }

/* Tables and numeric read-outs line up. */
[data-testid="stDataFrame"], [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* Sidebar: denser labels. */
section[data-testid="stSidebar"] label p { font-size: 13px; }

/* Respect reduced motion. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
</style>
"""


def apply() -> None:
    """Inject the theme CSS into the current Streamlit page."""
    import streamlit as st

    st.html(CSS)
