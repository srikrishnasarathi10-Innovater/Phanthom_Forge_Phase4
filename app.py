
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import IsolationForest

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Phantom Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# BACKEND CONFIGURATION
# Set BACKEND_URL in Render for cloud deployment.
# The default is the existing backend URL from your code.
# =========================================================
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://phanthom-forge-phase4.onrender.com",
).rstrip("/")


# =========================================================
# DESIGN SYSTEM — WHITE / PURPLE / PINK
# =========================================================
st.markdown(
    """
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"] {
    background: #FFFFFF;
    color: #171525;
}

.block-container {
    max-width: 1500px;
    padding: 2.5rem 2.5rem 3rem;
    margin: 0 auto;
    overflow: visible;
}

[data-testid="stMainBlockContainer"] {
    min-width: 0;
}

h1, h2, h3, h4, p, label, li {
    color: #171525;
}

h1 {
    font-size: clamp(1.8rem, 3vw, 2.6rem) !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    letter-spacing: -0.8px;
    overflow-wrap: anywhere;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 750 !important;
    margin-top: 1.2rem !important;
}

h3 {
    font-size: 1.12rem !important;
    font-weight: 700 !important;
}

/* Main page header */
.main-page-header {
    display: block;
    box-sizing: border-box;
    width: 100%;
    margin: 0 0 24px;
    padding: 12px 0 0;
    overflow-wrap: anywhere;
}

.main-page-title {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    color: #171525;
    font-size: clamp(1.55rem, 2vw, 2.1rem);
    font-weight: 850;
    line-height: 1.4;
    letter-spacing: -0.5px;
    padding: 2px 0 5px;
}

.main-title-shield {
    flex: 0 0 auto;
    font-size: 2rem;
}

.main-title-accent {
    color: #8051E8;
}

.main-page-subtitle {
    margin: 5px 0 0 58px;
    color: #756D84;
    font-size: clamp(1rem, 1.5vw, 1.2rem);
    line-height: 1.45;
}

.main-page-tagline {
    margin-top: 18px;
    color: #756D84;
    font-size: 1rem;
    line-height: 1.6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F8F6FF;
    border-right: 1px solid #E5D9FF;
    min-width: 250px;
}

section[data-testid="stSidebar"] > div {
    padding: 1.4rem 1rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #201A35 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 8px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #FFFFFF;
    border: 1px solid #E5D9FF;
    border-radius: 10px;
    padding: 11px 12px;
    width: 100%;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #F0E9FF;
    border-color: #8B5CF6;
}

section[data-testid="stSidebar"] [data-testid="stRadio"]
label > div:first-child {
    display: none;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #6D3FE8 !important;
    color: #FFFFFF !important;
    border: 1px solid #6D3FE8 !important;
    border-radius: 10px !important;
    min-height: 43px;
    padding: 0.6rem 1.1rem !important;
    font-weight: 650 !important;
    opacity: 1 !important;
    box-shadow: 0 4px 12px rgba(109, 63, 232, 0.15);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #5426C9 !important;
    border-color: #5426C9 !important;
    color: #FFFFFF !important;
}

.stButton > button:focus,
.stDownloadButton > button:focus {
    color: #FFFFFF !important;
    border-color: #EC4899 !important;
    box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.18) !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #171525 !important;
    border-color: #D8C9F5 !important;
}

.stTextInput label,
.stSelectbox label,
.stNumberInput label {
    color: #302744 !important;
    font-weight: 600 !important;
}

/* Metric cards */
.pd-card {
    background: #FFFFFF;
    border: 1px solid #E5D9FF;
    border-radius: 16px;
    padding: 21px;
    min-height: 135px;
    box-shadow: 0 5px 18px rgba(45, 25, 90, 0.045);
    overflow-wrap: anywhere;
}

.pd-card-label {
    font-size: 0.88rem;
    color: #655D76;
    margin-bottom: 13px;
    font-weight: 550;
}

.pd-card-value {
    font-size: clamp(1.5rem, 2.6vw, 2.3rem);
    font-weight: 800;
    color: #171525;
    line-height: 1.2;
    overflow-wrap: anywhere;
}

.pd-card-note {
    margin-top: 10px;
    color: #756D84;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* Score cards */
.score-card {
    background: #FFFFFF;
    border: 1px solid #E5D9FF;
    border-top: 5px solid #7C4DFF;
    border-radius: 16px;
    padding: 24px;
    min-height: 245px;
    box-shadow: 0 6px 20px rgba(45, 25, 90, 0.055);
    overflow-wrap: anywhere;
}

.score-card.ml {
    border-top-color: #D946A6;
}

.score-method {
    color: #786B8D;
    font-size: 0.77rem;
    font-weight: 750;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.score-description {
    color: #706780;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 10px 0 20px;
}

.score-number {
    font-size: clamp(2.5rem, 4vw, 3.5rem);
    font-weight: 850;
    line-height: 1.2;
    margin: 12px 0;
}

.score-denominator {
    font-size: 1rem;
    font-weight: 650;
}

.score-badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 30px;
    font-weight: 750;
    font-size: 0.84rem;
    margin-top: 5px;
}

.low {
    color: #278447 !important;
    background: #E7F7EC;
}

.medium {
    color: #285CC5 !important;
    background: #EAF1FF;
}

.high {
    color: #C93636 !important;
    background: #FDEAEA;
}

/* Information banners */
.demo-banner {
    background: #F0EBFF;
    border: 1px solid #D8C8FF;
    border-left: 5px solid #8051E8;
    color: #302345;
    border-radius: 12px;
    padding: 16px 19px;
    line-height: 1.65;
    margin: 18px 0 26px;
}

.info-box {
    background: #FFF1F8;
    border: 1px solid #F6C9E2;
    border-radius: 12px;
    padding: 15px;
    color: #542541;
    line-height: 1.6;
}

.success-box {
    background: #EAF8EE;
    border: 1px solid #C5E8CF;
    border-radius: 12px;
    padding: 15px;
    color: #245D35;
}

.section-caption {
    color: #756D84;
    font-size: 0.92rem;
    line-height: 1.6;
    margin-top: -5px;
    margin-bottom: 20px;
}

hr {
    border-color: #EDE5FA !important;
    margin: 24px 0 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E5D9FF;
    border-radius: 12px;
    overflow: hidden;
}

@media (max-width: 900px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    .pd-card {
        padding: 16px;
    }

    .score-card {
        padding: 18px;
    }

    .main-page-subtitle {
        margin-left: 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# DEMONSTRATION SENSOR DATA
# These readings are generated, not live hardware readings.
# =========================================================
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)

    sensor_names = [
        "Pressure Sensor",
        "Temperature Sensor",
        "Voltage Sensor",
        "Current Sensor",
    ]

    base_values = {
        "Pressure Sensor": 50.0,
        "Temperature Sensor": 70.0,
        "Voltage Sensor": 220.0,
        "Current Sensor": 12.0,
    }

    noise_values = {
        "Pressure Sensor": 2.0,
        "Temperature Sensor": 2.5,
        "Voltage Sensor": 4.0,
        "Current Sensor": 0.8,
    }

    rows = []
    start = datetime.now() - timedelta(minutes=119)

    for sensor in sensor_names:
        noise = noise_values[sensor]
        values = base_values[sensor] + rng.normal(0, noise, 30)

        anomaly_indices = rng.choice(30, size=3, replace=False)
        values[anomaly_indices] += (
            rng.choice([-1, 1], size=3)
            * noise
            * rng.uniform(4, 8, size=3)
        )

        for i, value in enumerate(values):
            rows.append(
                {
                    "Timestamp": start + timedelta(minutes=i * 4),
                    "Sensor": sensor,
                    "Reading": round(float(value), 2),
                    "Demo label": (
                        "Artificial anomaly"
                        if i in anomaly_indices
                        else "Generated reading"
                    ),
                }
            )

    df = (
        pd.DataFrame(rows)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    statistical_scores = []
    model_scores = []

    for sensor in sensor_names:
        mask = df["Sensor"] == sensor
        values = df.loc[mask, "Reading"].to_numpy()

        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad < 1e-9:
            robust_z = np.zeros(len(values))
        else:
            robust_z = 0.6745 * np.abs(values - median) / mad

        stat_scores = np.clip((robust_z / 6.0) * 100, 0, 100)

        model = IsolationForest(
            n_estimators=150,
            contamination=0.10,
            random_state=42,
        )
        model.fit(values.reshape(-1, 1))

        raw_anomaly = -model.decision_function(
            values.reshape(-1, 1)
        )

        ranks = (
            pd.Series(raw_anomaly)
            .rank(pct=True)
            .to_numpy()
        )
        ml_scores = np.clip(ranks * 100, 0, 100)

        statistical_scores.extend(stat_scores.tolist())
        model_scores.extend(ml_scores.tolist())

    df["Statistical Score"] = np.round(statistical_scores, 1)
    df["ML Model Score"] = np.round(model_scores, 1)

    df["Review Score"] = df[
        ["Statistical Score", "ML Model Score"]
    ].max(axis=1)

    df["Review Status"] = df["Review Score"].apply(
        lambda score: (
            "High" if score >= 60
            else "Medium" if score >= 30
            else "Low"
        )
    )

    return df.sort_values(
        "Timestamp", ascending=False
    ).reset_index(drop=True)


df = create_demo_data()


# =========================================================
# BACKEND API — SUMMARY AND INVESTIGATION CASES
# =========================================================
def fetch_backend_json(endpoint, params=None, timeout=10):
    """Fetch JSON from the configured backend."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"Invalid JSON response: {exc}"


backend_summary, summary_error = fetch_backend_json(
    "/summary",
    timeout=10,
)

backend_connected = (
    isinstance(backend_summary, dict)
    and summary_error is None
)

if not backend_connected:
    backend_summary = {}

cases_data, cases_error = fetch_backend_json(
    "/cases",
    params={"limit": 500},
    timeout=15,
)

backend_cases = []

if isinstance(cases_data, dict):
    raw_cases = cases_data.get("cases", [])
    if isinstance(raw_cases, list):
        backend_cases = raw_cases

cases_connected = (
    cases_error is None
    and isinstance(cases_data, dict)
    and isinstance(cases_data.get("cases", []), list)
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def score_class(score):
    if score >= 60:
        return "high", "High"
    if score >= 30:
        return "medium", "Medium"
    return "low", "Low"


def score_card_html(title, method, description, score, model=False):
    css_class, level = score_class(score)
    card_class = "score-card ml" if model else "score-card"

    return f"""
    <div class="{card_class}">
        <div class="score-method">{method}</div>
        <h3>{title}</h3>
        <div class="score-description">{description}</div>
        <div class="score-number {css_class}">
            {score:.1f}<span class="score-denominator"> / 100</span>
        </div>
        <span class="score-badge {css_class}">{level} score</span>
        <div class="pd-card-note">Score range: 0–100</div>
    </div>
    """


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="pd-card">
            <div class="pd-card-label">{label}</div>
            <div class="pd-card-value">{value}</div>
            <div class="pd-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_guide():
    st.markdown("### Score interpretation")
    st.markdown(
        '<span class="score-badge low">● 0–29.9 · Low</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="score-badge medium">● 30–59.9 · Medium</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="score-badge high">● 60–100 · High</span>',
        unsafe_allow_html=True,
    )


def flag_is_true(value):
    """Interpret common boolean representations from API records."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def get_priority_counts():
    counts = backend_summary.get("priority_counts", {})
    return counts if isinstance(counts, dict) else {}


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 20px 4px;">
            <div style="font-size:38px;">🛡️</div>
            <div style="font-size:24px;font-weight:850;line-height:1.3;">
                Phantom Detection <span style="color:#8051E8;">AI</span>
            </div>
            <div style="color:#756D84;font-size:14px;margin-top:7px;">
                Enterprise Anomaly Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### NAVIGATION")

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Sensor Monitoring",
            "Investigation Centre",
            "ERP Reconciliation",
        ],
        label_visibility="collapsed",
        key="page_navigation",
    )

    st.divider()
    st.markdown("### System status")

    if backend_connected and cases_connected:
        st.markdown(
            """
            <div class="success-box">
                <strong>Backend connected</strong><br>
                <span style="font-size:13px;">
                    API summary and cases available
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif backend_connected or cases_connected:
        st.warning("Backend partially available")
    else:
        st.error("Backend unavailable")

    st.caption("Sensor readings: generated demonstration data")
    st.divider()
    render_score_guide()


# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(
    """
    <div class="main-page-header">
        <div class="main-page-title">
            <span class="main-title-shield">🛡️</span>
            <span>Phantom Detection
                <span class="main-title-accent">AI</span>
            </span>
        </div>
        <div class="main-page-subtitle">
            Enterprise Anomaly Intelligence
        </div>
        <div class="main-page-tagline">
            Statistical analysis meets machine-learning anomaly detection.
        </div>
    </div>

    <div class="demo-banner">
        <strong>DEMO MODE</strong> — Sensor readings on this dashboard
        are generated sample data. Isolation Forest is fitted to these
        demonstration readings at runtime. This is not the saved sensor
        model's evaluation. ERP investigation records are also synthetic
        demonstration data, and no verified production ERP integration
        is claimed.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "Overview":

    st.markdown("## System Overview")
    st.markdown(
        '<div class="section-caption">'
        'Operational summary and two independent anomaly scores.'
        '</div>',
        unsafe_allow_html=True,
    )

    total_readings = len(df)
    high_count = int((df["Review Score"] >= 60).sum())
    medium_count = int(
        (
            (df["Review Score"] >= 30)
            & (df["Review Score"] < 60)
        ).sum()
    )
    low_count = int((df["Review Score"] < 30).sum())

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        metric_card(
            "Readings analysed",
            total_readings,
            "Across 4 demonstration sensors",
        )
    with c2:
        metric_card(
            "High-priority readings",
            high_count,
            "Review score ≥ 60",
        )
    with c3:
        metric_card(
            "Medium-priority readings",
            medium_count,
            "Review score 30–59.9",
        )
    with c4:
        metric_card(
            "Low-priority readings",
            low_count,
            "Review score below 30",
        )

    st.markdown("## Latest Reading — Two Independent Scores")

    sensor_options = sorted(df["Sensor"].unique().tolist())

    selected_sensor = st.selectbox(
        "Select a sensor to inspect",
        sensor_options,
        index=sensor_options.index(df.iloc[0]["Sensor"]),
    )

    sensor_data = df[df["Sensor"] == selected_sensor].copy()
    reading = sensor_data.iloc[0]

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            score_card_html(
                "Statistical Anomaly Score",
                "METHOD 01 · ROBUST STATISTICS",
                "Measures how far this reading deviates from its "
                "sensor's median using Median Absolute Deviation (MAD).",
                float(reading["Statistical Score"]),
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            score_card_html(
                "ML Model Anomaly Score",
                "METHOD 02 · ISOLATION FOREST",
                "Ranks the reading's relative unusualness using an "
                "Isolation Forest fitted to demonstration readings.",
                float(reading["ML Model Score"]),
                model=True,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("## Backend Investigation Summary")

    if backend_connected:
        counts = get_priority_counts()
        b1, b2, b3 = st.columns(3, gap="medium")

        with b1:
            metric_card(
                "Total Orders",
                backend_summary.get("total_orders", 0),
                "From backend API",
            )
        with b2:
            metric_card(
                "ERP Discrepancies",
                backend_summary.get("erp_discrepancy_cases", 0),
                "Orders requiring investigation",
            )
        with b3:
            metric_card(
                "High-Priority Cases",
                counts.get("HIGH", 0),
                "Prioritised for human review",
            )
    else:
        st.warning(
            "The backend summary is unavailable. "
            "Check the backend URL and server status."
        )
        with st.expander("Backend connection details"):
            st.code(BACKEND_URL)
            st.caption(summary_error or "No response from backend.")

    st.markdown("## Reading Details")

    d1, d2, d3 = st.columns(3)

    with d1:
        metric_card(
            "Sensor",
            selected_sensor,
            "Selected demonstration sensor",
        )

    with d2:
        metric_card(
            "Latest reading",
            f'{reading["Reading"]:.2f}',
            "Illustrative sensor units",
        )

    with d3:
        level_class, level_name = score_class(
            float(reading["Review Score"])
        )
        st.markdown(
            f"""
            <div class="pd-card">
                <div class="pd-card-label">Review priority</div>
                <div class="pd-card-value {level_class}">
                    {level_name}
                </div>
                <div class="pd-card-note">
                    Based on the higher of the two scores. This is a
                    review priority, not a confirmed equipment fault.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Timestamp: "
        + pd.Timestamp(reading["Timestamp"]).strftime(
            "%d %b %Y, %I:%M:%S %p"
        )
    )

    st.markdown(
        """
        <div class="info-box">
            <strong>How to interpret the scores</strong><br>
            Statistical score: robust deviation from the sensor's median.
            ML score: relative anomaly ranking from Isolation Forest.
            These methods may disagree. Investigate the underlying readings
            before deciding whether a fault exists.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE 2 — SENSOR MONITORING
# =========================================================
elif page == "Sensor Monitoring":

    st.markdown("## Sensor Monitoring")
    st.markdown(
        '<div class="section-caption">'
        'Inspect generated readings and compare the two scoring methods.'
        '</div>',
        unsafe_allow_html=True,
    )

    sensor_filter = st.selectbox(
        "Filter by sensor",
        ["All sensors"] + sorted(df["Sensor"].unique().tolist()),
    )

    status_filter = st.selectbox(
        "Filter by review priority",
        ["All priorities", "High", "Medium", "Low"],
    )

    filtered = df.copy()

    if sensor_filter != "All sensors":
        filtered = filtered[filtered["Sensor"] == sensor_filter]

    if status_filter != "All priorities":
        filtered = filtered[
            filtered["Review Status"] == status_filter
        ]

    display_df = filtered[
        [
            "Timestamp",
            "Sensor",
            "Reading",
            "Statistical Score",
            "ML Model Score",
            "Review Score",
            "Review Status",
            "Demo label",
        ]
    ].copy()

    display_df["Timestamp"] = display_df["Timestamp"].dt.strftime(
        "%d %b %Y %I:%M %p"
    )

    st.markdown(f"**{len(display_df)} readings match your filters.**")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download readings as CSV",
        data=csv_data,
        file_name="phantom_detection_demo_readings.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 3 — INVESTIGATION CENTRE
# =========================================================
elif page == "Investigation Centre":

    st.markdown("## Investigation Centre")
    st.markdown(
        '<div class="section-caption">'
        'Review readings that may deserve further attention. A high score '
        'alone does not establish a real sensor fault.'
        '</div>',
        unsafe_allow_html=True,
    )

    candidates = df[df["Review Score"] >= 30].copy()
    candidates = candidates.sort_values(
        "Review Score", ascending=False
    )

    if candidates.empty:
        st.success(
            "No readings currently meet the review threshold."
        )
    else:
        display_candidates = candidates[
            [
                "Timestamp",
                "Sensor",
                "Reading",
                "Statistical Score",
                "ML Model Score",
                "Review Score",
                "Review Status",
                "Demo label",
            ]
        ].copy()

        display_candidates["Timestamp"] = (
            display_candidates["Timestamp"].dt.strftime(
                "%d %b %Y %I:%M %p"
            )
        )

        st.markdown("### Readings requiring review")

        st.dataframe(
            display_candidates,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Investigate a reading")

        candidate_indices = candidates.index.tolist()

        chosen_index = st.selectbox(
            "Select a reading",
            candidate_indices,
            format_func=lambda idx: (
                f'{df.loc[idx, "Sensor"]} · '
                f'{df.loc[idx, "Timestamp"].strftime("%d %b %I:%M %p")} · '
                f'Score {df.loc[idx, "Review Score"]:.1f}'
            ),
        )

        chosen = df.loc[chosen_index]

        a, b = st.columns(2)

        with a:
            metric_card(
                "Statistical score",
                f'{chosen["Statistical Score"]:.1f}/100',
                "Robust deviation method",
            )

        with b:
            metric_card(
                "ML model score",
                f'{chosen["ML Model Score"]:.1f}/100',
                "Isolation Forest ranking",
            )

        st.markdown("### Investigation checklist")

        st.checkbox(
            "Compare the reading with recent sensor history",
            key="check_history",
        )
        st.checkbox(
            "Check timestamp gaps and repeated readings",
            key="check_timestamps",
        )
        st.checkbox(
            "Review maintenance and equipment records",
            key="check_maintenance",
        )
        st.checkbox(
            "Compare against independent ERP evidence",
            key="check_erp",
        )

        st.markdown(
            """
            <div class="info-box">
                <strong>Investigation outcome:</strong> Not confirmed.<br>
                Record the evidence and have a responsible reviewer determine
                whether the event is a sensor fault, operational change,
                maintenance event, or another cause.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 4 — ERP RECONCILIATION
# =========================================================
elif page == "ERP Reconciliation":

    st.markdown("## ERP Reconciliation")
    st.markdown(
        '<div class="section-caption">'
        'Review ERP discrepancy flags alongside sensor anomaly indicators.'
        '</div>',
        unsafe_allow_html=True,
    )

    if cases_connected:
        st.markdown(
            """
            <div class="info-box">
                <strong>Backend connected</strong><br>
                Investigation cases are loaded from the FastAPI backend.
                ERP records and sensor-to-machine mappings are synthetic
                demonstration data, not a verified production ERP
                integration.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Could not load investigation cases. "
            "Check the FastAPI backend connection."
        )
        with st.expander("Connection details"):
            st.code(BACKEND_URL)
            st.caption(cases_error or "No valid cases response received.")

    case_df = pd.DataFrame(backend_cases)

    if not case_df.empty:

        st.markdown("### Reconciliation summary")

        e1, e2, e3 = st.columns(3)
        counts = get_priority_counts()

        with e1:
            metric_card(
                "Demo orders reviewed",
                backend_summary.get("total_orders", len(case_df)),
                "Synthetic investigation cases",
            )

        with e2:
            discrepancy_count = backend_summary.get(
                "erp_discrepancy_cases",
                (
                    int(
                        case_df["erp_discrepancy_flag"].apply(
                            flag_is_true
                        ).sum()
                    )
                    if "erp_discrepancy_flag" in case_df.columns
                    else 0
                ),
            )
            metric_card(
                "ERP discrepancy flags",
                discrepancy_count,
                "Requires independent verification",
            )

        with e3:
            high_priority_count = counts.get(
                "HIGH",
                (
                    int(
                        (
                            case_df["investigation_priority"]
                            .astype(str)
                            .str.upper() == "HIGH"
                        ).sum()
                    )
                    if "investigation_priority" in case_df.columns
                    else 0
                ),
            )
            metric_card(
                "High-priority cases",
                high_priority_count,
                "Prioritised for human review",
            )

        st.markdown("### Filter investigation cases")

        f1, f2 = st.columns(2)

        with f1:
            selected_priority = st.selectbox(
                "Investigation priority",
                ["All", "HIGH", "MEDIUM", "LOW"],
            )

        with f2:
            discrepancy_only = st.checkbox(
                "Show only ERP discrepancy flags"
            )

        filtered_df = case_df.copy()

        if (
            selected_priority != "All"
            and "investigation_priority" in filtered_df.columns
        ):
            filtered_df = filtered_df[
                filtered_df["investigation_priority"]
                .astype(str)
                .str.upper() == selected_priority
            ]

        if discrepancy_only:
            if "erp_discrepancy_flag" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["erp_discrepancy_flag"].apply(
                        flag_is_true
                    )
                ]
            else:
                filtered_df = filtered_df.iloc[0:0]
                st.warning(
                    "The backend response does not contain an "
                    "ERP discrepancy flag column."
                )

        st.markdown("### Investigation case register")

        preferred_columns = [
            "order_id",
            "machine_id",
            "product_id",
            "expected_quantity",
            "recorded_quantity",
            "verified_quantity",
            "quantity_difference",
            "discrepancy_percentage",
            "erp_discrepancy_flag",
            "elevated_sensor_anomaly_flag",
            "investigation_priority",
            "investigation_status",
            "priority_reason",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in filtered_df.columns
        ]

        if not visible_columns:
            visible_columns = filtered_df.columns.tolist()

        st.caption(
            f"Showing {len(filtered_df)} of {len(case_df)} "
            "investigation cases."
        )

        st.dataframe(
            filtered_df[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

        csv_export = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered investigation cases",
            data=csv_export,
            file_name="phantom_erp_investigation_cases.csv",
            mime="text/csv",
        )

        st.markdown("### Evidence and interpretation")

        st.info(
            "An ERP discrepancy flag identifies a record that needs "
            "checking. A sensor anomaly is a separate warning signal. "
            "The mapping between sensor machines and ERP cases is "
            "simulated, so neither signal alone proves fraud or confirms "
            "a real-world relationship."
        )

        fraud_count = backend_summary.get(
            "fraud_confirmed_cases", 0
        )
        st.metric("Confirmed fraud cases", fraud_count)

        st.caption(
            "Status: Human review required. "
            "This prototype does not independently confirm fraud."
        )

    else:
        st.warning(
            "No investigation cases are available. Check the backend "
            "and its investigation case data."
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#81778F;font-size:12px;line-height:1.8;">
        <strong style="color:#6D3FE8;">PHANTOM DETECTION AI</strong><br>
        Enterprise Anomaly Intelligence · Demonstration Prototype<br>
        Statistical and ML scores are decision-support signals,
        not proof of fraud or equipment failure.
    </div>
    """,
    unsafe_allow_html=True,
)
