import os
import re
import time
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path

try:
    from .config_env import load_project_environment
except ImportError:
    from config_env import load_project_environment

load_project_environment(Path(__file__).resolve().parent)

import requests
import streamlit as st
from r7r13_observer import execute_governed
from r7r17_activity import r17_authorized_invocation, r17_child_headers


def _rerun() -> None:
    """Rerun on both current and older supported Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000")
TENANT_HEADER = os.getenv("TENANT_HEADER", "X-Tenant-ID")
UI_LANGUAGE = os.getenv("UI_LANGUAGE", "auto").strip().lower()

st.set_page_config(
    page_title="Hebrew-English RAG Application",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_LTR_BRACKET_RE = re.compile(
    r"\[(?:\d+|[^\]]*(?:\.pdf|chunk)[^\]]*)\]",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LEGACY_TIMING_RE = re.compile(r"^Thought for ([^\n]+)\n+", re.IGNORECASE)
_HEB_ORDINAL_PREFIX_RE = re.compile(
    r"^(?:ו?ראשית|ו?שנית|ו?שלישית|ו?רביעית|ו?חמישית|ו?שישית|"
    r"ו?שביעית|ו?שמינית|ו?תשיעית|ו?עשירית|ו?לבסוף|בסוף)\s*[,:\-]\s*"
)
_HEB_CONTINUATION_RE = re.compile(
    r"^(?:חברות אלה|החברות האלה|זה|זאת|זוהי|זוהו|הדבר|תהליך זה|"
    r"כתוצאה מכך|ולכן)\b"
)


_TEXT = {
    "English": {
        "brand_tag": "KNOWLEDGE ASSISTANT",
        "title": "Document question answering",
        "subtitle": "Ask questions across your organization’s knowledge and get clear, source-grounded answers.",
        "sidebar_title": "Workspace",
        "language": "Interface language / שפת ממשק",
        "connection": "Connection",
        "api_url": "Chat API URL",
        "api_help": "The address of the Hebrew-English RAG Application backend.",
        "tenant": "Tenant ID",
        "tenant_help": "Keeps retrieval and chat data isolated to this workspace.",
        "health": "Test connection",
        "clear": "New conversation",
        "ready": "Connected",
        "not_ready": "Backend not ready",
        "not_checked": "Connection not tested",
        "health_error": "Could not reach the backend",
        "health_details": "Connection details",
        "conversation": "Conversation",
        "messages": "messages",
        "download": "Download transcript",
        "debug": "Developer details",
        "show_raw": "Show raw conversation state",
        "endpoint": "Endpoint",
        "tenant_header": "Tenant header",
        "empty_title": "What would you like to understand?",
        "empty_body": "Start with a question, ask for a summary, or compare details found in your documents.",
        "starter_1": "Summarize the most important information",
        "starter_2": "Find the policy and explain it simply",
        "starter_3": "Compare the relevant options",
        "starter_help": "Use this example as your next question",
        "input": "Ask a question about your documents…",
        "thinking": "Searching your knowledge and preparing an answer…",
        "answered_in": "Answered in",
        "sources": "Sources",
        "source_details": "Source details",
        "unknown_source": "Unknown source",
        "error_title": "I could not complete that request.",
        "timeout": "The backend took too long to respond. Please try again.",
        "offline": "The backend is unavailable. Check the connection settings and try again.",
        "request_error": "The backend returned an error. Please try again.",
        "invalid_response": "The backend returned an invalid response.",
        "empty_answer": "No answer was returned. Try rephrasing your question.",
        "test_answer": "Test received — the interface is working. ✅",
        "privacy": "Answers are generated from the knowledge available to the selected tenant.",
        "user_label": "You",
        "assistant_label": "Assistant",
        "transcript_title": "Hebrew-English RAG Application conversation",
        "question_count": "questions",
    },
    "עברית": {
        "brand_tag": "עוזר ידע ארגוני",
        "title": "שאלות ותשובות על מסמכים",
        "subtitle": "שאלו על הידע הארגוני וקבלו תשובות ברורות ומבוססות מקורות.",
        "sidebar_title": "סביבת עבודה",
        "language": "שפת ממשק / Interface language",
        "connection": "חיבור",
        "api_url": "כתובת ה־Chat API",
        "api_help": "הכתובת של שרת Hebrew-English RAG Application.",
        "tenant": "מזהה ארגון",
        "tenant_help": "שומר את האחזור והשיחה מבודדים לסביבת העבודה הזאת.",
        "health": "בדיקת חיבור",
        "clear": "שיחה חדשה",
        "ready": "מחובר",
        "not_ready": "השרת עדיין לא מוכן",
        "not_checked": "החיבור טרם נבדק",
        "health_error": "לא ניתן להגיע לשרת",
        "health_details": "פרטי החיבור",
        "conversation": "שיחה",
        "messages": "הודעות",
        "download": "הורדת תמלול",
        "debug": "פרטים למפתחים",
        "show_raw": "הצגת מצב השיחה הגולמי",
        "endpoint": "כתובת שרת",
        "tenant_header": "כותרת ארגון",
        "empty_title": "מה תרצו להבין?",
        "empty_body": "אפשר להתחיל בשאלה, לבקש סיכום או להשוות פרטים שמופיעים במסמכים.",
        "starter_1": "סכם את המידע החשוב ביותר",
        "starter_2": "מצא את המדיניות והסבר אותה בפשטות",
        "starter_3": "השווה בין האפשרויות הרלוונטיות",
        "starter_help": "השתמשו בדוגמה הזאת כשאלה הבאה",
        "input": "שאלו שאלה על המסמכים שלכם…",
        "thinking": "מחפש במקורות הידע ומכין תשובה…",
        "answered_in": "התשובה נוצרה בתוך",
        "sources": "מקורות",
        "source_details": "פרטי מקורות",
        "unknown_source": "מקור לא ידוע",
        "error_title": "לא הצלחתי להשלים את הבקשה.",
        "timeout": "השרת לא השיב בזמן. נסו שוב.",
        "offline": "השרת אינו זמין. בדקו את הגדרות החיבור ונסו שוב.",
        "request_error": "השרת החזיר שגיאה. נסו שוב.",
        "invalid_response": "התקבלה תשובה לא תקינה מהשרת.",
        "empty_answer": "לא התקבלה תשובה. נסו לנסח את השאלה מחדש.",
        "test_answer": "הבדיקה התקבלה — הממשק עובד. ✅",
        "privacy": "התשובות נוצרות מהידע הזמין לארגון שנבחר.",
        "user_label": "אתם",
        "assistant_label": "העוזר",
        "transcript_title": "שיחת Hebrew-English RAG Application",
        "question_count": "שאלות",
    },
}


def is_hebrew(text: str) -> bool:
    return bool(text and _HEBREW_RE.search(str(text)))


def clean_text(text: str) -> str:
    """Return display-safe plain text while preserving useful paragraphs."""
    if not text:
        return ""
    value = html_unescape(str(text))
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\uFEFF]", "", value)
    return value.strip()


def rtl_mixed_html(text: str) -> str:
    """Escape answer text and isolate LTR citation-like fragments."""
    output = []
    last = 0
    for match in _LTR_BRACKET_RE.finditer(text):
        output.append(html_escape(text[last : match.start()]))
        output.append(
            "<span dir='ltr' class='ltr-isolate'>"
            + html_escape(match.group(0))
            + "</span>"
        )
        last = match.end()
    output.append(html_escape(text[last:]))
    return "".join(output)


def pretty_layout(text: str) -> str:
    if not text:
        return ""
    if text.count("•") >= 2 and "\n" not in text:
        text = text.replace(" • ", "\n• ")
    if "\n" in text or "•" in text:
        return text
    compact = re.sub(r"\s+", " ", text).strip()
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(compact) if part.strip()]
    if len(parts) < 2:
        parts = [part.strip() for part in re.split(r"\s*;\s*", compact) if part.strip()]
    if len(parts) < 2:
        return text
    if len(parts) >= 4:
        return f"{parts[0]}\n\n" + "\n".join(f"• {part}" for part in parts[1:])
    return "\n\n".join(parts)


def normalize_hebrew_bullets(text: str) -> str:
    if not text:
        return ""
    output = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw.startswith("•"):
            output.append(line)
            continue
        content = _HEB_ORDINAL_PREFIX_RE.sub("", raw.lstrip("•").strip()).strip()
        if output and output[-1].strip().startswith("•") and _HEB_CONTINUATION_RE.match(content):
            output[-1] = output[-1].rstrip() + " " + content
        else:
            output.append("• " + content)
    return "\n".join(output).strip()


def _safe_citation_value(value) -> str:
    if value is None:
        return ""
    return html_escape(clean_text(str(value)), quote=True)


def _api_endpoint(base_url: str, path: str) -> str:
    base = clean_text(base_url).strip().rstrip("/")
    return f"{base}{path}"


def _format_duration(seconds) -> str:
    try:
        elapsed = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return ""
    if elapsed < 10:
        return f"{elapsed:.1f}s"
    total = int(round(elapsed))
    minutes, remaining = divmod(total, 60)
    return f"{minutes}m {remaining:02d}s" if minutes else f"{remaining}s"


def _legacy_timing(content: str):
    match = _LEGACY_TIMING_RE.match(content)
    if not match:
        return "", content
    return match.group(1).strip(), content[match.end() :].lstrip()


def _transcript(chat, labels) -> str:
    lines = [labels["transcript_title"], "=" * len(labels["transcript_title"]), ""]
    for message in chat:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "assistant")
        label = labels["user_label"] if role == "user" else labels["assistant_label"]
        _, body = _legacy_timing(clean_text(message.get("content", "")))
        lines.extend((f"{label}:", body, ""))
        citations = message.get("citations", [])
        if isinstance(citations, list) and citations:
            files = [clean_text(item.get("file", "")) for item in citations if isinstance(item, dict)]
            files = [name for name in files if name]
            if files:
                lines.extend((f"{labels['sources']}: " + ", ".join(files), ""))
    return "\n".join(lines).strip() + "\n"


def _render_message_text(text: str) -> None:
    body = normalize_hebrew_bullets(pretty_layout(clean_text(text)))
    if is_hebrew(body):
        st.markdown(
            "<div class='message-copy rtl-copy' dir='rtl'>"
            + rtl_mixed_html(body)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(body)


def render_citations(citations, labels) -> None:
    """Render compact, escaped source cards without trusting backend HTML."""
    if not isinstance(citations, list):
        return
    items = [item for item in citations if isinstance(item, dict)]
    if not items:
        return

    pills = []
    for index, citation in enumerate(items, start=1):
        citation_id = _safe_citation_value(citation.get("id") or index)
        filename = _safe_citation_value(citation.get("file") or labels["unknown_source"])
        location = _safe_citation_value(citation.get("location"))
        location_text = f"<span class='source-location'> · {location}</span>" if location else ""
        pills.append(
            "<span class='source-pill' dir='auto'>"
            f"<span class='source-index'>{citation_id}</span> {filename}{location_text}"
            "</span>"
        )
    st.markdown(
        "<div class='source-wrap' dir='auto'>"
        f"<div class='source-label'>{html_escape(labels['sources'])}</div>"
        + "".join(pills)
        + "</div>",
        unsafe_allow_html=True,
    )

    has_details = any(
        item.get("snippet")
        or item.get("score") is not None
        or item.get("chunk_id") is not None
        or item.get("chunk_index") is not None
        for item in items
    )
    if not has_details:
        return

    with st.expander(f"{labels['source_details']} ({len(items)})", expanded=False):
        for index, citation in enumerate(items, start=1):
            citation_id = _safe_citation_value(citation.get("id") or index)
            filename = _safe_citation_value(citation.get("file") or labels["unknown_source"])
            location = _safe_citation_value(citation.get("location"))
            snippet = _safe_citation_value(citation.get("snippet"))
            metadata = []
            if location:
                metadata.append(location)
            chunk = citation.get("chunk_id", citation.get("chunk_index"))
            if chunk is not None:
                metadata.append(f"chunk {_safe_citation_value(chunk)}")
            if citation.get("score") is not None:
                metadata.append(f"score {_safe_citation_value(citation.get('score'))}")
            meta_html = (
                "<div class='source-meta' dir='auto'>" + " · ".join(metadata) + "</div>"
                if metadata
                else ""
            )
            snippet_html = (
                f"<div class='source-snippet' dir='auto'>{snippet}</div>" if snippet else ""
            )
            st.markdown(
                "<article class='source-detail'>"
                f"<div class='source-title' dir='auto'><span>{citation_id}</span> {filename}</div>"
                + meta_html
                + snippet_html
                + "</article>",
                unsafe_allow_html=True,
            )


def _apply_styles(rtl_interface: bool) -> None:
    interface_rules = """
    [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    [data-testid="stSidebar"] input { direction: ltr; text-align: left; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
    [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
    """ if rtl_interface else ""
    st.markdown(
        f"""
        <style>
        :root {{
            --px-accent: var(--primary-color, #6d5dfc);
            --px-surface: var(--secondary-background-color, #f5f7fb);
            --px-text: var(--text-color, #172033);
            --px-bg: var(--background-color, #ffffff);
            --px-border: color-mix(in srgb, var(--px-text) 13%, transparent);
            --px-muted: color-mix(in srgb, var(--px-text) 62%, transparent);
            --px-accent-soft: color-mix(in srgb, var(--px-accent) 12%, var(--px-bg));
            --px-shadow: 0 18px 50px color-mix(in srgb, #172033 10%, transparent);
        }}

        html {{ scroll-behavior: smooth; }}
        .stApp {{
            background:
                radial-gradient(circle at 85% -10%, color-mix(in srgb, var(--px-accent) 13%, transparent), transparent 28rem),
                radial-gradient(circle at 5% 22%, color-mix(in srgb, #36b6a5 8%, transparent), transparent 24rem),
                var(--px-bg);
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 8rem;
        }}
        [data-testid="stSidebar"] {{
            border-inline-end: 1px solid var(--px-border);
            background: color-mix(in srgb, var(--px-surface) 88%, var(--px-bg));
        }}
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 1.25rem; }}

        .brand-card {{
            padding: 1rem 1.05rem;
            margin: 0 0 1.15rem;
            border: 1px solid var(--px-border);
            border-radius: 18px;
            background: linear-gradient(145deg, var(--px-accent-soft), color-mix(in srgb, var(--px-bg) 92%, transparent));
            box-shadow: 0 9px 24px color-mix(in srgb, #172033 7%, transparent);
        }}
        .brand-row {{ display: flex; align-items: center; gap: .7rem; }}
        .brand-mark {{
            width: 2.35rem; height: 2.35rem; flex: 0 0 2.35rem;
            display: grid; place-items: center; border-radius: 12px;
            color: white; font-size: 1.25rem;
            background: linear-gradient(145deg, var(--px-accent), #3f8efc);
            box-shadow: 0 7px 18px color-mix(in srgb, var(--px-accent) 34%, transparent);
        }}
        .brand-name {{ font-size: 1.02rem; font-weight: 750; line-height: 1.1; }}
        .brand-subtitle {{ color: var(--px-muted); font-size: .72rem; margin-top: .22rem; letter-spacing: .045em; }}

        .hero {{
            position: relative; overflow: hidden;
            padding: clamp(1.5rem, 3.5vw, 2.7rem);
            margin-bottom: 1.5rem;
            border: 1px solid var(--px-border);
            border-radius: 28px;
            background: linear-gradient(125deg, color-mix(in srgb, var(--px-bg) 91%, var(--px-accent)), color-mix(in srgb, var(--px-bg) 96%, #36b6a5));
            box-shadow: var(--px-shadow);
        }}
        .hero::after {{
            content: "✦"; position: absolute; inset-inline-end: 4%; top: -2.5rem;
            color: color-mix(in srgb, var(--px-accent) 11%, transparent);
            font-size: clamp(10rem, 22vw, 17rem); line-height: 1; pointer-events: none;
        }}
        .hero-content {{ position: relative; z-index: 1; max-width: 750px; }}
        .eyebrow {{
            color: var(--px-accent); font-size: .72rem; font-weight: 800;
            letter-spacing: .13em; margin-bottom: .55rem;
        }}
        .hero h1 {{ margin: 0; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.08; letter-spacing: -.035em; }}
        .hero p {{ max-width: 650px; margin: .8rem 0 1rem; color: var(--px-muted); font-size: clamp(.96rem, 1.5vw, 1.08rem); line-height: 1.65; }}
        .hero-meta {{ display: flex; gap: .55rem; flex-wrap: wrap; align-items: center; }}
        .meta-chip {{
            display: inline-flex; align-items: center; gap: .4rem;
            min-height: 1.85rem; padding: .28rem .7rem; border-radius: 999px;
            border: 1px solid var(--px-border); background: color-mix(in srgb, var(--px-bg) 78%, transparent);
            font-size: .76rem; font-weight: 650;
        }}
        .status-dot {{ width: .48rem; height: .48rem; border-radius: 50%; background: #98a2b3; }}
        .status-dot.ready {{ background: #17a673; box-shadow: 0 0 0 4px color-mix(in srgb, #17a673 15%, transparent); }}
        .status-dot.warning {{ background: #e69c25; }}
        .status-dot.error {{ background: #e05260; }}

        .empty-state {{ text-align: center; max-width: 650px; margin: 2.3rem auto 1.15rem; }}
        .empty-icon {{
            width: 3.25rem; height: 3.25rem; display: grid; place-items: center;
            margin: 0 auto .8rem; border-radius: 16px; color: var(--px-accent); font-size: 1.45rem;
            background: var(--px-accent-soft); border: 1px solid color-mix(in srgb, var(--px-accent) 22%, transparent);
        }}
        .empty-state h2 {{ margin: 0 0 .35rem; font-size: 1.32rem; }}
        .empty-state p {{ margin: 0; color: var(--px-muted); line-height: 1.55; }}

        [data-testid="stChatMessage"] {{
            max-width: 920px; margin: .7rem auto; padding: 1rem 1.05rem;
            border: 1px solid var(--px-border); border-radius: 20px;
            background: color-mix(in srgb, var(--px-bg) 94%, transparent);
            box-shadow: 0 7px 22px color-mix(in srgb, #172033 5%, transparent);
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: color-mix(in srgb, var(--px-accent) 8%, var(--px-bg));
        }}
        .message-copy {{ white-space: pre-wrap; line-height: 1.72; overflow-wrap: anywhere; }}
        .rtl-copy {{ direction: rtl; unicode-bidi: plaintext; text-align: right; }}
        .ltr-isolate {{ direction: ltr; unicode-bidi: isolate; }}
        .timing-chip {{
            display: inline-flex; align-items: center; gap: .3rem; margin-bottom: .55rem;
            color: var(--px-muted); font-size: .7rem; font-weight: 650;
        }}

        .source-wrap {{ margin-top: .85rem; padding-top: .72rem; border-top: 1px solid var(--px-border); }}
        .source-label {{ color: var(--px-muted); font-size: .7rem; font-weight: 750; letter-spacing: .06em; text-transform: uppercase; margin-bottom: .42rem; }}
        .source-pill {{
            display: inline-flex; align-items: center; gap: .3rem; max-width: 100%;
            padding: .3rem .62rem; margin: .15rem .3rem .15rem 0;
            border: 1px solid var(--px-border); border-radius: 999px;
            background: var(--px-surface); font-size: .72rem; overflow-wrap: anywhere;
        }}
        .source-index {{
            display: inline-grid; place-items: center; min-width: 1.12rem; height: 1.12rem;
            border-radius: 50%; color: white; background: var(--px-accent); font-size: .62rem; font-weight: 800;
        }}
        .source-location, .source-meta {{ color: var(--px-muted); }}
        .source-detail {{
            border-inline-start: 3px solid var(--px-accent); margin: .5rem 0;
            padding: .35rem .7rem; background: color-mix(in srgb, var(--px-accent) 4%, transparent);
            border-radius: 0 10px 10px 0;
        }}
        .source-title {{ font-size: .82rem; font-weight: 720; }}
        .source-title > span {{ color: var(--px-accent); }}
        .source-meta {{ font-size: .69rem; margin-top: .15rem; }}
        .source-snippet {{ font-size: .77rem; line-height: 1.55; margin-top: .35rem; white-space: pre-wrap; }}

        .connection-card {{
            margin: .65rem 0 .9rem; padding: .65rem .75rem; border: 1px solid var(--px-border);
            border-radius: 12px; background: color-mix(in srgb, var(--px-bg) 76%, transparent);
            font-size: .78rem;
        }}
        .connection-card strong {{ display: block; margin-bottom: .1rem; }}
        .connection-card span {{ color: var(--px-muted); overflow-wrap: anywhere; }}
        .sidebar-stat {{ display: flex; justify-content: space-between; color: var(--px-muted); font-size: .8rem; margin: .2rem 0 .7rem; }}

        .stButton > button, .stDownloadButton > button {{
            min-height: 2.65rem; border-radius: 12px; border-color: var(--px-border);
            transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px); border-color: var(--px-accent);
            box-shadow: 0 6px 16px color-mix(in srgb, var(--px-accent) 15%, transparent);
        }}
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
        input:focus-visible, textarea:focus-visible {{ outline: 3px solid color-mix(in srgb, var(--px-accent) 35%, transparent) !important; outline-offset: 2px; }}
        [data-testid="stChatInput"] {{
            border: 1px solid var(--px-border); border-radius: 18px;
            box-shadow: 0 12px 35px color-mix(in srgb, #172033 12%, transparent);
            overflow: hidden;
        }}
        .privacy-note {{ text-align: center; color: var(--px-muted); font-size: .68rem; margin-top: .65rem; }}
        hr {{ border-color: var(--px-border) !important; }}

        @media (max-width: 700px) {{
            [data-testid="stMainBlockContainer"] {{ padding: 1rem .8rem 7.5rem; }}
            .hero {{ border-radius: 20px; padding: 1.35rem; }}
            .hero::after {{ opacity: .55; }}
            [data-testid="stChatMessage"] {{ padding: .8rem; border-radius: 16px; }}
            .source-location {{ display: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            html {{ scroll-behavior: auto; }}
            *, *::before, *::after {{ transition: none !important; animation: none !important; }}
        }}
        {interface_rules}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Initialize state before widgets and migrate old stored content safely.
if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "tenant" not in st.session_state:
    st.session_state["tenant"] = ""
if "api_url" not in st.session_state:
    st.session_state["api_url"] = API_URL
if "backend_status" not in st.session_state:
    st.session_state["backend_status"] = None
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "עברית" if UI_LANGUAGE in {"he", "heb", "hebrew", "עברית"} else "English"

cleaned_chat = []
for stored_message in st.session_state["chat"]:
    if isinstance(stored_message, dict):
        message = dict(stored_message)
        message["content"] = clean_text(message.get("content", ""))
        cleaned_chat.append(message)
    else:
        cleaned_chat.append({"role": "assistant", "content": clean_text(stored_message)})
st.session_state["chat"] = cleaned_chat

language = st.session_state["ui_language"]
labels = _TEXT.get(language, _TEXT["English"])
rtl_interface = language == "עברית"
_apply_styles(rtl_interface)


with st.sidebar:
    st.markdown(
        "<div class='brand-card'><div class='brand-row'>"
        "<div class='brand-mark' aria-hidden='true'>✦</div>"
        "<div><div class='brand-name'>Hebrew-English RAG Application</div>"
        f"<div class='brand-subtitle'>{html_escape(labels['brand_tag'])}</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.selectbox(
        labels["language"],
        options=("English", "עברית"),
        key="ui_language",
    )
    st.subheader(labels["sidebar_title"])
    st.markdown(f"**{labels['connection']}**")
    api_url = st.text_input(
        labels["api_url"],
        key="api_url",
        help=labels["api_help"],
        placeholder="http://localhost:8000",
    )
    tenant = st.text_input(
        labels["tenant"],
        key="tenant",
        help=labels["tenant_help"],
    )

    action_columns = st.columns(2)
    with action_columns[0]:
        check_connection = st.button(labels["health"], use_container_width=True)
    with action_columns[1]:
        if st.button(labels["clear"], use_container_width=True):
            st.session_state["chat"] = []
            st.session_state.pop("pending_query", None)
            _rerun()

    if check_connection:
        try:
            response = requests.get(
                _api_endpoint(api_url, "/readyz"),
                headers={TENANT_HEADER: clean_text(tenant)},
                timeout=30,
            )
            response.raise_for_status()
            health_data = response.json()
            ready = bool(health_data.get("ready"))
            st.session_state["backend_status"] = {
                "kind": "ready" if ready else "warning",
                "detail": labels["ready"] if ready else labels["not_ready"],
                "data": health_data,
            }
        except Exception as exc:
            st.session_state["backend_status"] = {
                "kind": "error",
                "detail": labels["health_error"],
                "error": clean_text(str(exc)),
            }

    backend_status = st.session_state.get("backend_status")
    if backend_status:
        status_kind = backend_status.get("kind", "warning")
        status_detail = clean_text(backend_status.get("detail", labels["not_ready"]))
    else:
        status_kind = "neutral"
        status_detail = labels["not_checked"]
    st.markdown(
        f"<div class='connection-card'><strong><span class='status-dot {status_kind}'></span> "
        f"{html_escape(status_detail)}</strong>"
        f"<span>{html_escape(clean_text(api_url) or API_URL)}</span></div>",
        unsafe_allow_html=True,
    )

    message_count = len(st.session_state["chat"])
    question_count = sum(
        1 for item in st.session_state["chat"] if isinstance(item, dict) and item.get("role") == "user"
    )
    st.markdown(
        f"<div class='sidebar-stat'><span>{html_escape(labels['conversation'])}</span>"
        f"<strong>{question_count} {html_escape(labels['question_count'])} · "
        f"{message_count} {html_escape(labels['messages'])}</strong></div>",
        unsafe_allow_html=True,
    )
    if st.session_state["chat"]:
        st.download_button(
            labels["download"],
            data=_transcript(st.session_state["chat"], labels),
            file_name="hebrew-english-rag-conversation.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )

    with st.expander(labels["debug"], expanded=False):
        st.caption(f"{labels['endpoint']}: {_api_endpoint(api_url, '/chat')}")
        st.caption(f"{labels['tenant_header']}: {TENANT_HEADER}")
        if backend_status and backend_status.get("data"):
            with st.expander(labels["health_details"], expanded=False):
                st.json(backend_status["data"])
        if backend_status and backend_status.get("error"):
            st.caption(backend_status["error"])
        if st.checkbox(labels["show_raw"], value=False):
            st.json(st.session_state["chat"])


backend_status = st.session_state.get("backend_status")
if backend_status:
    hero_status_kind = backend_status.get("kind", "warning")
    hero_status_text = clean_text(backend_status.get("detail", labels["not_ready"]))
else:
    hero_status_kind = "neutral"
    hero_status_text = labels["not_checked"]

hero_direction = "rtl" if rtl_interface else "ltr"
st.markdown(
    f"<section class='hero' dir='{hero_direction}'>"
    "<div class='hero-content'>"
    f"<div class='eyebrow'>{html_escape(labels['brand_tag'])}</div>"
    f"<h1>{html_escape(labels['title'])}</h1>"
    f"<p>{html_escape(labels['subtitle'])}</p>"
    "<div class='hero-meta'>"
    f"<span class='meta-chip'><span class='status-dot {hero_status_kind}'></span>{html_escape(hero_status_text)}</span>"
    f"<span class='meta-chip' dir='ltr'>◉ {html_escape(clean_text(tenant) or '—')}</span>"
    "</div></div></section>",
    unsafe_allow_html=True,
)


for message in st.session_state["chat"]:
    role = message.get("role", "assistant")
    content = clean_text(message.get("content", ""))
    avatar = "👤" if role == "user" else "✦"
    with st.chat_message(role if role in {"user", "assistant"} else "assistant", avatar=avatar):
        if role == "user":
            _render_message_text(content)
            continue

        legacy_duration, body = _legacy_timing(content)
        duration = _format_duration(message.get("elapsed_seconds")) or legacy_duration
        if duration:
            st.markdown(
                f"<div class='timing-chip' dir='auto'>◷ {html_escape(labels['answered_in'])} "
                f"<span dir='ltr'>{html_escape(duration)}</span></div>",
                unsafe_allow_html=True,
            )
        if message.get("is_error"):
            st.error(body, icon="⚠️")
        else:
            _render_message_text(body)
        render_citations(message.get("citations", []), labels)


suggested_question = None
if not st.session_state["chat"]:
    st.markdown(
        f"<div class='empty-state' dir='{hero_direction}'>"
        "<div class='empty-icon' aria-hidden='true'>✦</div>"
        f"<h2>{html_escape(labels['empty_title'])}</h2>"
        f"<p>{html_escape(labels['empty_body'])}</p></div>",
        unsafe_allow_html=True,
    )
    starter_columns = st.columns(3)
    for column, key in zip(starter_columns, ("starter_1", "starter_2", "starter_3")):
        with column:
            if st.button(
                labels[key],
                key=f"prompt_{key}",
                help=labels["starter_help"],
                use_container_width=True,
            ):
                suggested_question = labels[key]


typed_question = st.chat_input(labels["input"])
user_question = clean_text(typed_question or suggested_question or "")
if user_question and not st.session_state.get("pending_query"):
    st.session_state["chat"].append({"role": "user", "content": user_question})
    st.session_state["pending_query"] = user_question
    _rerun()

st.markdown(
    f"<div class='privacy-note' dir='{hero_direction}'>{html_escape(labels['privacy'])}</div>",
    unsafe_allow_html=True,
)


pending_question = st.session_state.pop("pending_query", None)
if pending_question:
    normalized = pending_question.strip().lower().rstrip(".!?")
    if normalized in {"test", "this is a test", "just a test", "ping"}:
        st.session_state["chat"].append(
            {"role": "assistant", "content": labels["test_answer"], "elapsed_seconds": 0.0}
        )
        _rerun()

    history = st.session_state["chat"][:-1][-6:]
    if history and history[0].get("role") == "assistant":
        history = history[1:]
    history_payload = [
        {"role": turn.get("role", "assistant"), "content": clean_text(turn.get("content", ""))}
        for turn in history
        if isinstance(turn, dict)
    ]
    payload = {"query": pending_question, "history": history_payload}
    headers = {TENANT_HEADER: clean_text(st.session_state.get("tenant", ""))}
    api_url = _api_endpoint(st.session_state.get("api_url", API_URL), "/chat")
    def input_factory(binding):
        outbound_headers = dict(headers)
        outbound_headers["X-Request-ID"] = binding.request_identity
        outbound_headers.update(r17_child_headers(
            surface_id="chat-api-request",
            dimension_id="REAL_SUBJECT_EXECUTION_COUNT",
            operation_identity="candidate-runtime-chat-endpoint",
        ))
        return {"method": "POST", "url": api_url, "headers": outbound_headers,
                "payload": payload, "timeout": 1000}

    started = time.monotonic()
    try:
        with st.spinner(labels["thinking"]):
            with r17_authorized_invocation(
                dimension="REAL_SUBJECT_EXECUTION_COUNT",
                surface="webui-streamlit-request",
                operation_identity="streamlit-user-post-chat",
                input_factory=input_factory,
            ) as r17_input:
                actual_headers = dict(r17_input["headers"]) if r17_input is not None else headers
                response = execute_governed(
                    dimension="REAL_SUBJECT_EXECUTION_COUNT",
                    surface="webui-streamlit-request",
                    operation="streamlit-user-post-chat",
                    target=lambda: requests.post(
                        api_url, json=payload, headers=actual_headers, timeout=1000,
                    ),
                    r17_input=r17_input,
                )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ValueError(labels["invalid_response"])
        answer = clean_text(data.get("answer", "")) or labels["empty_answer"]
        citations = data.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        st.session_state["chat"].append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    except requests.Timeout:
        error_message = labels["timeout"]
        st.session_state["chat"].append(
            {"role": "assistant", "content": error_message, "is_error": True}
        )
    except requests.ConnectionError:
        error_message = labels["offline"]
        st.session_state["chat"].append(
            {"role": "assistant", "content": error_message, "is_error": True}
        )
    except requests.RequestException:
        error_message = labels["request_error"]
        st.session_state["chat"].append(
            {"role": "assistant", "content": error_message, "is_error": True}
        )
    except (TypeError, ValueError):
        st.session_state["chat"].append(
            {"role": "assistant", "content": labels["invalid_response"], "is_error": True}
        )
    _rerun()
