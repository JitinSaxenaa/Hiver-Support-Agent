import json
import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

st.set_page_config(
    page_title="Hiver Support Copilot",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

def configure_streamlit_secrets():
    secret_paths = (
        Path(".streamlit/secrets.toml"),
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if not any(path.exists() for path in secret_paths):
        return
    try:
        secrets = st.secrets
        for key in ("LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
            if key in secrets and secrets[key]:
                os.environ[key] = str(secrets[key])
    except FileNotFoundError:
        # Local runs do not need a secrets.toml file because local mode is the default.
        return

configure_streamlit_secrets()

SCENARIOS = {
    "Battery drain": "@AppleSupport updated to iOS 11 and now my battery dies in 2 hours on my iPhone 7!",
    "Billing dispute": "@AppleSupport I was charged twice for Apple Music. This is fraud, please refund my money.",
    "Account locked": "@AppleSupport My Apple ID has been locked and I am not receiving the verification code.",
    "AirPods disconnect": "@AppleSupport my AirPods disconnect every 30 seconds during phone calls on iPhone X.",
    "Store complaint": "@AppleSupport Terrible experience at Covent Garden store today, staff was rude and made me wait 2 hours.",
}

INTENT_LABELS = {
    "software_update_os": "Software and updates",
    "battery_power_charging": "Battery and charging",
    "account_appleid_icloud": "Apple ID and iCloud",
    "billing_subscription_store": "Billing and subscriptions",
    "hardware_display_audio": "Hardware and audio",
    "network_connectivity": "Network and connectivity",
    "general_complaint_store": "Store or service complaint",
    "other": "Other",
}

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
    .brand-title { font-size: 1.85rem; font-weight: 750; letter-spacing: -0.03em; color: #f8fafc !important; margin-bottom: .35rem; }
    .brand-subtitle { color: #cbd5e1 !important; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.15rem; }
    .step { color: #cbd5e1 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .reply-card { background: #1f2937; color: #f8fafc !important; border: 1px solid #475569; border-left: 5px solid #60a5fa; border-radius: 10px; padding: 1rem; font-size: 0.96rem; line-height: 1.5; }
    .reply-card strong { color: #bfdbfe !important; }
    .tweet-card { background: #111827; color: #f9fafb; border-radius: 10px; padding: 1rem 1.2rem; line-height: 1.5; }
    .small-note { color: #cbd5e1; font-size: .85rem; }
    div[data-testid="stMetric"] { background: #1f2937; border: 1px solid #475569; padding: .65rem; border-radius: 10px; min-height: 104px; }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #f8fafc !important; }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: .82rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: .78rem !important; }
    .metric-card { background: #1f2937; border: 1px solid #475569; border-radius: 10px; padding: .75rem .85rem; min-height: 104px; }
    .metric-label { color: #cbd5e1; font-size: .82rem; line-height: 1.25; }
    .metric-value { color: #f8fafc; font-size: 1.45rem; font-weight: 700; line-height: 1.2; margin-top: .55rem; overflow-wrap: anywhere; }
    .metric-help { color: #94a3b8; font-size: .78rem; line-height: 1.25; margin-top: .45rem; }
    [data-testid="stCaptionContainer"] { font-size: .82rem !important; }
    h3 { font-size: 1.35rem !important; margin-top: 1.7rem !important; margin-bottom: .7rem !important; }
    h4 { font-size: 1.05rem !important; }
    [data-testid="stAlert"] { margin-top: .35rem; margin-bottom: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_eval_results():
    path = Path("eval/eval_results.json")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_golden_data():
    path = Path("eval/golden_set.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def set_scenario(name):
    st.session_state["tweet_text"] = SCENARIOS[name]
    st.session_state["last_result"] = None


def render_metric_card(label, value, help_text):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("## Hiver Support Copilot")
    st.sidebar.caption("A grounded triage assistant for @AppleSupport")
    page = st.sidebar.radio(
        "Go to",
        ["Agent demo", "Evaluation", "Intent guide", "Golden set"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown("### System status")
    st.sidebar.success("Ready for local demo")
    st.sidebar.caption("Brand: @AppleSupport")
    st.sidebar.caption("Grounding: 20,000 resolved threads")
    st.sidebar.caption("Safety: human escalation enabled")
    st.sidebar.divider()
    st.sidebar.markdown("### How to use")
    st.sidebar.caption("1. Choose a scenario or write a tweet")
    st.sidebar.caption("2. Click Process message")
    st.sidebar.caption("3. Explain the four output cards")
    return page


def render_result(result):
    st.divider()
    st.markdown("### 2. Agent decision")
    st.caption("The pipeline classifies the message, finds similar resolved cases, drafts a reply, and applies safety rules.")

    intent = INTENT_LABELS.get(result["intent"], result["intent"].replace("_", " ").title())
    decision = result["escalation"]["decision"]
    decision_label = "AUTO-HANDLE" if decision == "auto" else "ESCALATE TO HUMAN"
    grounding = result["grounding_score"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Intent", intent, "What is the customer asking about?")
    with m2:
        render_metric_card("Confidence", f"{result['confidence']:.0%}", "Classifier certainty")
    with m3:
        render_metric_card("Grounding", f"{grounding:.2f}", "Similarity to resolved cases")
    with m4:
        render_metric_card("Decision", decision_label, "Safety routing outcome")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="step">Suggested response</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="reply-card"><strong>@AppleSupport</strong><br>{result["draft_reply"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption("This is a draft for an agent to review, not an automatic customer send.")
    with right:
        st.markdown('<div class="step">Why this route?</div>', unsafe_allow_html=True)
        if decision == "auto":
            st.success(result["escalation"]["reason"])
        else:
            st.error(result["escalation"]["reason"])
        st.markdown("**Classifier reasoning**")
        st.caption(result["classification_reasoning"])

    with st.expander("Show grounding evidence", expanded=True):
        cases = result.get("retrieved_cases", [])
        if not cases:
            st.info("No sufficiently similar historical case was found.")
        for index, case in enumerate(cases, 1):
            st.markdown(f"**Case {index}** - similarity `{case['similarity_score']:.2f}`")
            st.markdown(f"Customer: {case['inbound_text']}")
            st.markdown(f"Historical reply: {case['brand_reply_text']}")
            if index != len(cases):
                st.divider()


def render_agent():
    st.markdown('<div class="brand-title">Support message triage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Turn one noisy customer tweet into an explainable, grounded support action.</div>',
        unsafe_allow_html=True,
    )
    st.info("This demo is intentionally human-in-the-loop: the agent drafts and recommends; a support specialist approves the final action.")

    st.markdown("### 1. Give the agent a customer message")
    st.caption("Use a prepared example for the fastest demo, or type your own message below.")
    cols = st.columns(len(SCENARIOS))
    for column, name in zip(cols, SCENARIOS):
        column.button(name, key=f"scenario_{name}", on_click=set_scenario, args=(name,), use_container_width=True)

    if "tweet_text" not in st.session_state:
        st.session_state["tweet_text"] = SCENARIOS["Battery drain"]
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    tweet = st.text_area(
        "Customer tweet",
        key="tweet_text",
        height=125,
        placeholder="@AppleSupport tell us what the customer needs help with...",
    )
    process = st.button("Process message", type="primary", use_container_width=False)
    if process:
        if not tweet.strip():
            st.warning("Enter a customer message before processing it.")
        else:
            with st.spinner("Running classification, retrieval, reply drafting, and safety triage..."):
                from src.pipeline import TwitterSupportPipeline

                # Local mode is the safe default; Streamlit Secrets can opt into a hosted LLM.
                provider = os.getenv("LLM_PROVIDER", "local").lower()
                if provider == "ollama":
                    provider = "local"
                st.session_state["last_result"] = TwitterSupportPipeline(provider=provider).process_tweet(tweet.strip())

    if st.session_state["last_result"]:
        render_result(st.session_state["last_result"])
    else:
        st.markdown("### What you will see")
        preview = st.columns(4)
        preview[0].info("Intent\n\nBattery, billing, account, network, and more.")
        preview[1].info("Evidence\n\nSimilar historical Apple Support cases.")
        preview[2].info("Reply\n\nA concise, brand-aligned draft.")
        preview[3].info("Decision\n\nAuto-handle or escalate with a reason.")


def render_evaluation():
    results = load_eval_results()
    st.markdown('<div class="brand-title">Evaluation at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Evidence that the system is better than simple baselines.</div>', unsafe_allow_html=True)
    classification = results.get("intent_classification", {})
    system = classification.get("system_classifier", {})
    simple = classification.get("simple_baseline_tfidf", {})
    trivial = classification.get("trivial_baseline", {})
    a, b, c, d = st.columns(4)
    with a:
        render_metric_card("Held-out examples", results.get("total_examples_evaluated", 200), "Golden evaluation set")
    with b:
        render_metric_card("System accuracy", f"{system.get('accuracy', 0):.1%}", "Intent classification")
    with c:
        render_metric_card("System macro F1", f"{system.get('macro_f1', 0):.3f}", "Balanced intent score")
    with d:
        render_metric_card("Runtime", f"{results.get('execution_runtime_seconds', 0):.2f}s", "Evaluation runtime")
    st.markdown("### Baseline comparison")
    table = pd.DataFrame([
        {"Model": "Trivial majority class", "Accuracy": trivial.get("accuracy", 0), "Macro F1": trivial.get("macro_f1", 0)},
        {"Model": "TF-IDF + Logistic Regression", "Accuracy": simple.get("accuracy", 0), "Macro F1": simple.get("macro_f1", 0)},
        {"Model": "Hiver support classifier", "Accuracy": system.get("accuracy", 0), "Macro F1": system.get("macro_f1", 0)},
    ])
    st.dataframe(table.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("Metrics are read from eval/eval_results.json and reproduced by python -m eval.run_eval.")
    st.warning(
        "Evaluation transparency: intent and triage metrics are computed on the checked-in golden set. "
        "Reply quality uses an LLM-as-judge when a provider is configured, otherwise a deterministic rubric fallback. "
        "The repository's human-agreement calibration is rubric-generated, not a fresh independent human study."
    )


def render_intent_guide():
    from src.taxonomy import OUT_OF_SCOPE_DECLARATION, TAXONOMY_DEFINITIONS

    st.markdown('<div class="brand-title">Intent guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">The eight labels used by the classifier, in plain language.</div>', unsafe_allow_html=True)
    for intent, definition in TAXONOMY_DEFINITIONS.items():
        with st.expander(INTENT_LABELS[intent.value]):
            st.write(definition)
    st.markdown("### Explicit boundaries")
    for key, value in OUT_OF_SCOPE_DECLARATION.items():
        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")


def render_golden_set():
    df = load_golden_data()
    st.markdown('<div class="brand-title">Golden evaluation set</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">The hand-labelled examples used to measure quality.</div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("The golden set was not found.")
        return
    intent_options = ["All"] + sorted(df["true_intent"].dropna().unique().tolist())
    selected = st.selectbox("Filter by true intent", intent_options)
    view = df if selected == "All" else df[df["true_intent"] == selected]
    st.caption(f"Showing {len(view)} of {len(df)} examples")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False), "golden_set_export.csv", "text/csv")


page = render_sidebar()
if page == "Agent demo":
    render_agent()
elif page == "Evaluation":
    render_evaluation()
elif page == "Intent guide":
    render_intent_guide()
else:
    render_golden_set()
