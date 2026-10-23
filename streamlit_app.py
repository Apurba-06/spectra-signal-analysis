"""SPECTRA - live web version of the COMINT Signal Analysis Workbench (same backend as the desktop app).

Run locally:   streamlit run streamlit_app.py
Deploy:        Hugging Face Spaces (Docker SDK) or Render - see Dockerfile / README.
"""
import csv
import json
import os
import tempfile
import time

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import soundfile as sf

import benchmark as BM
import blind as BL
import correlation as COR
import deinterleave as DI
from analysis import (analyze_signal, compute_spectrum, compute_waterfall, constellation_points,
                      flatten_params, offline_analyst)
from report_export import build_report
from signal_loader import IQ_FORMATS, MAX_SAMPLES, load_signal

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = {
    "demo.wav - QPSK + Conv K=7 + Block interleaver": ("demo.wav", 48000, "cf32"),
    "demo_bpsk_rs.wav - BPSK + RS(255,223)": ("demo_bpsk_rs.wav", 48000, "cf32"),
    "demo_qpsk_concat.iq - QPSK + RS+Conv + Forney interleaver": ("demo_qpsk_concat.iq", 250000, "cf32"),
    "demo_fsk.wav - 2-FSK + Conv + Diagonal interleaver": ("demo_fsk.wav", 48000, "cf32"),
    "demo_16qam_uncoded.wav - 16-QAM uncoded": ("demo_16qam_uncoded.wav", 48000, "cf32"),
}

st.set_page_config(page_title="SPECTRA | Signal Analysis", page_icon="S", layout="wide")
THEMES = {
    "Light": {
        "paper": "#f4f1e9",
        "panel": "#ebe7dc",
        "surface": "#f8f6f0",
        "ink": "#292b25",
        "muted": "#73756b",
        "line": "#d7d2c5",
        "accent": "#526b4f",
        "accent_hover": "#40573e",
        "signal": "#526b4f",
        "secondary": "#a15d43",
        "heatmap": "Cividis",
        "plotly_template": "plotly_white",
    },
    "Dark": {
        "paper": "#171a1b",
        "panel": "#202425",
        "surface": "#252a2b",
        "ink": "#e7e4da",
        "muted": "#a7aaa0",
        "line": "#3b4240",
        "accent": "#a6b98c",
        "accent_hover": "#8fa474",
        "signal": "#a6b98c",
        "secondary": "#d58c6e",
        "heatmap": "Cividis",
        "plotly_template": "plotly_dark",
    },
}
with st.sidebar:
    theme_name = st.selectbox("Appearance", list(THEMES), key="appearance")
palette = THEMES[theme_name]
st.markdown(
    """
    <style>
    :root {
        --paper: #f4f1e9;
        --paper-deep: #ebe7dc;
        --ink: #292b25;
        --muted: #73756b;
        --line: #d7d2c5;
        --moss: #526b4f;
        --rust: #a15d43;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
    }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1440px;
        padding-top: 2.6rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: var(--paper-deep);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }
    .spectra-header {
        border-bottom: 1px solid var(--line);
        padding: 0.3rem 0 1.45rem;
        margin-bottom: 1.65rem;
    }
    .spectra-kicker, .spectra-side-kicker {
        color: var(--moss);
        font: 600 0.7rem/1.4 "Segoe UI", Arial, sans-serif;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .spectra-header h1 {
        color: var(--ink);
        font: 500 clamp(2.2rem, 5vw, 3.25rem)/1.05 Georgia, "Times New Roman", serif;
        letter-spacing: -0.045em;
        margin: 0.7rem 0 0.45rem;
    }
    .spectra-header p {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0;
    }
    [data-testid="stSidebar"] h2 {
        color: var(--ink);
        font: 500 1.55rem/1.2 Georgia, "Times New Roman", serif;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stWidgetLabel"] p {
        color: var(--ink);
        font-size: 0.86rem;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: var(--surface);
        border-color: var(--line);
        border-radius: 4px;
    }
    [data-testid="stFileUploader"] section {
        background: var(--surface);
        border: 1px dashed #aaa999;
        border-radius: 4px;
    }
    [data-testid="stFileUploader"] section small {
        color: var(--muted);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: var(--moss);
        border: 1px solid var(--moss);
        border-radius: 4px;
        color: #fffdf7;
        font-weight: 600;
        min-height: 2.8rem;
        transition: background 120ms ease, border-color 120ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: #40573e;
        border-color: #40573e;
        color: #fffdf7;
    }
    [data-testid="stTabs"] [role="tablist"] {
        display: flex;
        flex-wrap: wrap !important;
        gap: 0.5rem;
        overflow: visible !important;
        width: 100%;
        border: 0;
        padding: 0.25rem 0 0.75rem;
    }
    [data-testid="stTabs"] [role="tab"] {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        min-height: 2.4rem;
        padding: 0.45rem 0.85rem;
        border: 1px solid var(--line);
        border-radius: 5px;
        background: var(--surface);
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 500;
        transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        border-color: var(--moss);
        color: var(--ink);
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: var(--moss);
        border-color: var(--moss);
        color: var(--paper);
        font-weight: 650;
    }
    [data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"],
    [data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }
    [data-testid="stTabs"] [role="tab"]:focus-visible {
        outline: 2px solid var(--moss);
        outline-offset: 2px;
    }
    [data-testid="stAlert"] {
        border-radius: 4px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
    }
    </style>
    <header class="spectra-header">
        <div class="spectra-kicker">SIH 2026 / PS 26147 · Signal analysis</div>
        <h1>SPECTRA</h1>
        <p>Automated model for analysis of .IQ and .wav files along with signal parameter extraction</p>
    </header>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"dark" if theme_name == "Dark" else "light"};
        --paper: {palette["paper"]} !important;
        --paper-deep: {palette["panel"]} !important;
        --surface: {palette["surface"]} !important;
        --ink: {palette["ink"]} !important;
        --muted: {palette["muted"]} !important;
        --line: {palette["line"]} !important;
        --moss: {palette["accent"]} !important;
        --rust: {palette["secondary"]} !important;
    }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        background: var(--paper) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] {{ background: var(--paper-deep) !important; }}
    [data-testid="stSidebar"] [data-testid="stButton"] button {{
        background: {palette["accent"]} !important;
        border-color: {palette["accent"]} !important;
        color: {palette["paper"]} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
        background: {palette["accent_hover"]} !important;
        border-color: {palette["accent_hover"]} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stFileUploader"] section {{
        background: var(--surface) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        background: var(--surface) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background: var(--surface) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {{
        background: var(--surface) !important;
        border-color: var(--line) !important;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {{
        background: var(--surface) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }}
    [data-baseweb="select"] input,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{
        color: var(--ink) !important;
        caret-color: var(--ink);
    }}
    [data-baseweb="select"] svg,
    [data-baseweb="input"] svg,
    [data-testid="stNumberInput"] button svg {{
        fill: var(--ink);
        color: var(--ink);
    }}
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {{
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28) !important;
    }}
    [role="option"],
    [data-baseweb="menu"] li {{
        background: var(--surface) !important;
        color: var(--ink) !important;
    }}
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover {{
        background: var(--paper-deep) !important;
        color: var(--ink) !important;
    }}
    [role="option"][aria-selected="true"] {{
        box-shadow: inset 3px 0 var(--moss);
    }}
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {{
        background: var(--surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stButton"] button:hover:not(:disabled),
    [data-testid="stDownloadButton"] button:hover:not(:disabled),
    [data-testid="stFormSubmitButton"] button:hover:not(:disabled) {{
        background: var(--paper-deep) !important;
        border-color: var(--moss) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stButton"] button:disabled,
    [data-testid="stDownloadButton"] button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled {{
        background: var(--paper-deep) !important;
        border-color: var(--line) !important;
        color: var(--muted) !important;
        opacity: 1 !important;
        cursor: not-allowed;
    }}
    [data-testid="stButton"] button:disabled *,
    [data-testid="stDownloadButton"] button:disabled *,
    [data-testid="stFormSubmitButton"] button:disabled * {{
        color: var(--muted) !important;
    }}
    [data-testid="stBaseButton-primary"] {{
        background: {palette["accent"]} !important;
        border-color: {palette["accent"]} !important;
        color: {palette["paper"]} !important;
    }}
    [data-testid="stBaseButton-primary"]:hover:not(:disabled) {{
        background: {palette["accent_hover"]} !important;
        border-color: {palette["accent_hover"]} !important;
        color: {palette["paper"]} !important;
    }}
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label, [data-testid="stCaptionContainer"] {{
        color: var(--ink);
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] button *,
    [data-testid="stBaseButton-primary"] * {{
        color: {palette["paper"]} !important;
    }}
    [data-testid="stButton"] button:disabled *,
    [data-testid="stDownloadButton"] button:disabled *,
    [data-testid="stFormSubmitButton"] button:disabled * {{
        color: var(--muted) !important;
    }}
    [data-testid="stFileUploader"] section {{
        background: var(--surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {{
        color: var(--ink);
    }}
    [data-testid="stFileUploader"] section button {{
        background: var(--paper-deep) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {{
        color: var(--ink) !important;
    }}
    [data-testid="stCheckbox"] input,
    [data-testid="stRadio"] input {{
        accent-color: var(--moss);
    }}
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary {{
        background: var(--surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        color: var(--ink) !important;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        background: var(--surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{
        border-color: {palette["accent"]} !important;
        color: var(--ink) !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: {palette["accent"]} !important;
        border-color: {palette["accent"]} !important;
        color: {palette["paper"]} !important;
    }}
    [data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"],
    [data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"] p {{
        color: inherit !important;
    }}
    [data-testid="stAlert"] {{ background: var(--surface); color: var(--ink); }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def run(path_or_bytes, name, fs_iq, fmt, byte_order, mod, sps):
    temporary_path = None
    if isinstance(path_or_bytes, bytes):
        suffix = os.path.splitext(name)[1] or ".iq"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(path_or_bytes)
            path = fh.name
            temporary_path = path
    else:
        path = path_or_bytes
    try:
        started = time.perf_counter()
        x, fs, ftype = load_signal(path, fs_iq, fmt, byte_order)
        file_size = os.path.getsize(path)
        if ftype == "wav":
            source_samples = int(sf.info(path).frames)
        else:
            source_samples = file_size // (IQ_FORMATS[fmt][0].itemsize * IQ_FORMATS[fmt][1])
        loader_truncated = source_samples > MAX_SAMPLES
        x = x[:600000]
        res = analyze_signal(x, fs, mod=None if mod == "Auto" else mod, sps=sps)
        return x, fs, ftype, res, source_samples, file_size, loader_truncated, time.perf_counter() - started
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def clear_previous_result():
    st.session_state.ran = False


if "ran" not in st.session_state:
    st.session_state.ran = False
if "benchmark_curves" not in st.session_state:
    st.session_state.benchmark_curves = None
if "benchmark_metadata" not in st.session_state:
    st.session_state.benchmark_metadata = None


# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.markdown('<div class="spectra-side-kicker">01 / Capture setup</div>', unsafe_allow_html=True)
    st.header("Input")
    src = st.radio("Signal source", ["Bundled demo capture", "Upload my own file"], on_change=clear_previous_result)
    fs_iq, fmt, byte_order, path, data, name = 2_000_000.0, "cf32", "little", None, None, ""
    if src == "Bundled demo capture":
        label = st.selectbox("Capture", list(DEMOS), on_change=clear_previous_result)
        fn, fs_iq, fmt = DEMOS[label]
        path, name = os.path.join(HERE, fn), fn
    else:
        up = st.file_uploader("Upload .wav / .iq / .bin", type=["wav", "iq", "bin", "raw", "dat"], on_change=clear_previous_result)
        fs_iq = st.number_input("Sample rate for raw IQ (Hz)", min_value=1.0, value=2_000_000.0, step=1000.0, on_change=clear_previous_result)
        fmt = st.selectbox("IQ format", ["cf32", "ci16", "ci8", "cu8"], on_change=clear_previous_result)
        byte_order = st.selectbox(
            "Raw IQ byte order",
            ["little", "big"],
            help="Applies to multi-byte formats (cf32 and ci16).",
            on_change=clear_previous_result,
        )
        center_frequency_hz = st.number_input(
            "Capture center frequency (Hz, optional)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Recorded for context only; carrier offset is measured relative to the supplied baseband samples.",
        )
        if up is not None:
            data, name = up.getvalue(), up.name
    if src == "Bundled demo capture":
        center_frequency_hz = 0.0
    with st.expander("Manual override (optional)"):
        mod = st.selectbox("Modulation", ["Auto", "BPSK", "QPSK", "16-QAM", "FSK"], on_change=clear_previous_result)
        sps_txt = st.text_input("Samples/symbol (auto or number)", "auto", on_change=clear_previous_result)
    run_clicked = st.button("Run analysis", type="primary", width="stretch")

sps = None
if sps_txt.strip().lower() not in ("", "auto"):
    try:
        sps = float(sps_txt)
        if not np.isfinite(sps) or sps <= 0:
            raise ValueError("Samples per symbol must be a finite number greater than zero.")
    except ValueError as e:
        st.sidebar.error(f"Invalid samples/symbol value: {e}")
        st.stop()
if run_clicked:
    st.session_state.ran = (path is not None) or (data is not None)
    if not st.session_state.ran:
        st.sidebar.error("Choose a demo capture or upload a file before running the analysis.")
src_ready = (path is not None) or (data is not None)

tabs = st.tabs([
    "Overview",
    "Parameters",
    "Spectrum",
    "Waterfall",
    "Constellation",
    "I/Q waveform",
    "Demodulation",
    "De-interleaving",
    "FEC",
    "Bit correlation",
    "AI analyst",
    "BER benchmark",
])

with tabs[0]:
    st.subheader("Project overview")
    st.markdown(
        """
        **SPECTRA IS AN automated model for analysis of .IQ and .wav files along with signal parameter extraction.**
        It analyzes capture samples without requiring a supplied modulation or coding profile, estimates
        signal parameters, recovers symbols and bits, searches for frame synchronization, then tests
        supported interleavers and error-correcting codes.
        """
    )
    st.markdown("**Receiver workflow**")
    st.write("1. Load a WAV recording or raw IQ capture.")
    st.write("2. Estimate modulation, symbol rate, bandwidth, carrier offset, and signal quality.")
    st.write("3. Recover symbols and bits, then search for known frame markers.")
    st.write("4. Test supported interleavers and FEC decoders.")
    st.write("5. Check recovered frames with CRC and inspect the BER benchmark.")
    st.markdown("**How to read the result**")
    st.write(
        "CRC-16 OK means the recovered frame passed its integrity check. Signal measurements are estimates "
        "from the samples. FEC and interleaver labels are receiver hypotheses, not proof of the transmitter "
        "configuration. A receiver lock without a valid frame can indicate noise, encryption, an unsupported "
        "scheme, or a sync word outside the current library."
    )
    st.caption(
        "Current coverage: BPSK, QPSK, 16-QAM, and 2-FSK; convolutional K=7, RS(255,223), "
        "and built-in LDPC decoding."
    )

if st.session_state.ran and src_ready:
    try:
        with st.spinner("Running blind receiver (baud → modulation → timing/carrier → sync → interleaver/FEC detection)..."):
            x, fs, ftype, res, source_samples, file_size, loader_truncated, processing_seconds = run(
                data if data is not None else path, name, fs_iq, fmt, byte_order, mod, sps
            )
        params = flatten_params(res, fs, len(x))
        rx, sy, ch = res["rx"], res["sync"], res["chain"]
    except Exception as e:
        st.error(f"Could not process this file: {e}")
        st.stop()

    st.caption(
        f"Input: {ftype.upper()} · {file_size:,} bytes · {source_samples:,} source samples · "
        f"{len(x):,} analyzed samples · {processing_seconds:.2f} s"
        + (f" · center {center_frequency_hz:,.0f} Hz" if center_frequency_hz else "")
    )
    if loader_truncated or source_samples > len(x):
        st.warning(
            f"Analysis used {len(x):,} samples. The browser receiver is capped at 600,000 analyzed samples "
            f"and the file loader at {MAX_SAMPLES:,}; longer captures are truncated from the end."
        )

    with tabs[0]:
        if res["frame"]:
            st.success(f"Payload recovered, CRC-16 OK: “{res['frame']['payload'].decode('ascii', 'replace')}”")
        elif rx is None:
            st.error(f"Receiver could not lock: {res['error']}")
        else:
            st.warning("Receiver locked but no CRC-valid frame was found (stream may be random/encrypted).")
        st.info("Open the Parameters tab for measured receiver values. FEC and interleaver details are hypotheses; a CRC pass confirms frame integrity.")
        st.subheader("Export analysis report")
        report_stem = os.path.splitext(os.path.basename(name))[0] or "capture"
        report_args = {
            "source_name": name,
            "file_type": ftype,
            "sample_rate": fs,
            "file_size": file_size,
            "source_samples": source_samples,
            "analyzed_samples": len(x),
            "runtime_seconds": processing_seconds,
            "center_frequency_hz": center_frequency_hz,
            "params": params,
            "result": res,
        }
        pdf_col, word_col, text_col = st.columns(3)
        pdf_col.download_button(
            "Download PDF report",
            data=build_report("pdf", **report_args),
            file_name=f"SPECTRA_{report_stem}_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
        word_col.download_button(
            "Download Word report",
            data=build_report("docx", **report_args),
            file_name=f"SPECTRA_{report_stem}_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width="stretch",
        )
        text_col.download_button(
            "Download text report",
            data=build_report("txt", **report_args),
            file_name=f"SPECTRA_{report_stem}_report.txt",
            mime="text/plain",
            width="stretch",
        )

    with tabs[1]:
        if res["frame"]:
            st.success(f"Payload recovered, CRC-16 OK: “{res['frame']['payload'].decode('ascii', 'replace')}”")
        elif rx is None:
            st.error(f"Receiver could not lock: {res['error']}")
        else:
            st.warning("Receiver locked but no CRC-valid frame was found (stream may be random/encrypted).")
        c1, c2 = st.columns(2)
        items = list(params.items())
        half = (len(items) + 1) // 2
        for col, part in ((c1, items[:half]), (c2, items[half:])):
            _ = col.markdown("\n".join(f"**{k}**: {v}  " for k, v in part))
        if rx is not None:
            classifier = (
                f"Classifier's heuristic score: {rx.cls['confidence'] * 100:.0f}%."
                if rx.cls
                else "Modulation was supplied manually; no automatic classifier score is available."
            )
            st.info(
                f"Evidence: receiver lock is {'present' if rx.rec['locked'] else 'weak'}, "
                f"known frame sync is {'present' if sy['found'] else 'not found'}, and CRC-16 is "
                f"{'valid' if res['frame'] else 'not validated'}. {classifier} "
                "The classifier score is heuristic, not a statistically calibrated probability; "
                "no confidence interval is claimed."
            )

    with tabs[2]:
        st.caption("Power spectral density shows signal power across frequency. Hover for values; drag to zoom and use the chart toolbar to pan or reset.")
        f, db = compute_spectrum(x, fs)
        fig = go.Figure(go.Scatter(x=f / 1e3, y=db, mode="lines", name="Power spectral density", line={"color": palette["signal"], "width": 1.4}))
        fig.update_layout(
            template=palette["plotly_template"],
            title="Power spectral density",
            xaxis_title="Frequency (kHz)",
            yaxis_title="Power (dB)",
            height=520,
            dragmode="zoom",
            margin={"t": 60, "r": 25, "b": 55, "l": 65},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}, key="spectrum_plot")

    with tabs[3]:
        st.caption("The waterfall displays how signal power changes over time and frequency. Use zoom to inspect a region.")
        fw, tw, Sw = compute_waterfall(x, fs)
        fig = go.Figure(
            go.Heatmap(
                x=tw,
                y=fw / 1e3,
                z=Sw,
                colorscale=palette["heatmap"],
                colorbar={"title": "Power (dB)"},
                name="Power",
                hovertemplate="Time: %{x:.4f} s<br>Frequency: %{y:.2f} kHz<br>Power: %{z:.1f} dB<extra></extra>",
            )
        )
        fig.update_layout(
            template=palette["plotly_template"],
            title="Signal power over time",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (kHz)",
            height=580,
            dragmode="zoom",
            margin={"t": 60, "r": 40, "b": 55, "l": 65},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}, key="waterfall_plot")

    with tabs[4]:
        st.caption("Recovered symbols cluster by modulation state. Tight groups usually indicate cleaner symbol recovery; a diffuse cloud can indicate noise or residual synchronization error.")
        pts = constellation_points(None, symbols=rx.rec["symbols"]) if rx is not None and rx.rec.get("symbols") is not None else constellation_points(x)
        if rx is not None and rx.mod == "FSK":
            st.info("2-FSK is represented by soft bit decisions rather than an I/Q constellation.")
        else:
            fig = go.Figure(go.Scattergl(x=pts.real, y=pts.imag, mode="markers", name="Recovered symbols", marker={"size": 4, "color": palette["signal"], "opacity": 0.62}))
            fig.update_layout(
                template=palette["plotly_template"],
                title="Recovered constellation",
                xaxis_title="In-phase (I)",
                yaxis_title="Quadrature (Q)",
                yaxis={"scaleanchor": "x", "scaleratio": 1},
                height=580,
                dragmode="zoom",
                margin={"t": 60, "r": 40, "b": 55, "l": 65},
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}, key="constellation_plot")
    with tabs[5]:
        st.caption("The in-phase (I) and quadrature (Q) components show the complex signal over a short sample window. Drag to zoom into a section.")
        seg = x[:800]
        waveform = go.Figure()
        waveform.add_trace(go.Scatter(x=np.arange(len(seg)), y=seg.real, mode="lines", name="I", line={"color": palette["signal"], "width": 1}))
        waveform.add_trace(go.Scatter(x=np.arange(len(seg)), y=seg.imag, mode="lines", name="Q", line={"color": palette["secondary"], "width": 1}))
        waveform.update_layout(
            template=palette["plotly_template"],
            title="I/Q waveform (first 800 samples)",
            xaxis_title="Sample",
            yaxis_title="Amplitude",
            height=520,
            dragmode="zoom",
            legend={"orientation": "h"},
        )
        st.plotly_chart(waveform, width="stretch", config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}, key="waveform_plot")

    with tabs[6]:
        st.caption("Demodulation recovers symbols and bits from the sampled waveform. Sync is a match against known marker patterns; a missing match does not mean the capture contains no signal.")
        if rx is None:
            st.error(res["error"])
        else:
            r = rx.rec
            st.write(f"**Modulation:** {rx.mod}  ·  **Symbol rate:** {rx.baud:,.2f} baud ({rx.sps:.2f} samples/symbol)  ·  "
                     f"**SNR:** {r['snr_db']:.1f} dB  ·  **Lock:** {'LOCKED' if r['locked'] else 'WEAK'}")
            if sy["found"]:
                h = sy["hit"]
                st.write(f"**Frame sync:** {h['pattern']} at raw bit {h['position']} ({sy['variant']['name']}"
                         f"{', inverted' if h['inverted'] else ''}, {h['mismatches']} bit errors) - phase ambiguity resolved.")
            else:
                st.write("**Frame sync:** no known sync word found (raw stream shown).")
            st.code(COR.format_bit_preview(sy["bits"], 512), language=None)

    with tabs[7]:
        st.caption("Interleavers rearrange transmitted bits to spread burst errors. The receiver ranks supported arrangements by how well they fit the tested code structure.")
        if rx is not None:
            st.write(f"**Blind interleaver search** - chosen: `{ch['interleaver']}`")
            if ch.get("ranking"):
                st.table([{"Hypothesis": r["mode"], "Parameters": str(r["params"]), "Code mismatch %": round(r["score"] * 100, 1)} for r in ch["ranking"]])
            mode, kw = ch.get("interleaver_mode", "None"), ch.get("interleaver_kw", {})
            out = DI.deinterleave(sy["bits"], mode, **kw) if mode != "None" else sy["bits"]
            st.code(COR.format_bit_preview(out, 512), language=None)

    with tabs[8]:
        st.caption("Forward-error correction (FEC) adds redundancy so a receiver can correct errors. These are the best-supported decoding hypotheses for this capture.")
        if rx is not None:
            st.write(f"**Blind FEC detection:** {ch['fec']}")
            st.json({k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in (res["fec_info"] or {}).items()})
            st.code(COR.format_bit_preview(res["decoded_bits"], 512), language=None)

    with tabs[9]:
        st.caption("Correlation checks decoded bits for known sync markers, CRC-valid frames, periodicity, and stream entropy.")
        if rx is not None:
            bits = res["decoded_bits"]
            fr = COR.parse_frame(bits)
            lib = COR.search_sync(sy.get("raw_bits", sy["bits"]))
            st.write("**Sync-word library matches (raw stream):** " + (", ".join(f"{h['pattern']} @ {h['position']} ({h['mismatches']}/{h['length']} errors)" for h in lib[:3]) or "none"))
            if fr and fr["crc_ok"]:
                hx, asc = COR.hex_ascii(fr["payload"], 48)
                st.success(f"Frame found: length field {fr['length']} bytes, CRC-16 OK (0x{fr['crc_rx']:04X})")
                st.code(f"HEX   : {hx}\nASCII : {fr['payload'].decode('ascii', 'replace')}", language=None)
            else:
                hx, asc = COR.hex_ascii(COR.bits_to_bytes(bits, 32), 32)
                st.info("No CRC-valid frame. First bytes of the stream:")
                st.code(f"HEX   : {hx}\nASCII : {asc}", language=None)
            lag, strength = COR.estimate_frame_period(bits)
            st.write(f"Stream entropy: **{COR.bit_entropy(bits):.3f}** (1.000 = random) · periodicity: "
                     f"{'frame period ~%d bits' % lag if lag else 'none detected'}")

    with tabs[10]:
        st.caption("The offline analyst summarizes the measured receiver output using local rules; it does not send your capture to an AI service.")
        for label in ("Summarize Signal", "Explain Pipeline Settings", "Tactical Assessment"):
            if st.button(label):
                st.code(offline_analyst(label, params), language=None)
else:
    with tabs[0]:
        st.info("Choose a bundled capture or upload a file in the sidebar, then run the analysis.")

with tabs[11]:
    st.write("Bit-error rate (BER) measured through the receiver chain with pulse shaping, carrier and timing offsets, and added noise. Lower values mean fewer bit errors.")
    st.caption("The x-axis is Eb/N0 (signal energy per bit relative to noise). Hollow points indicate receiver lock failures; downward triangles show points where no errors were observed in the tested bits. Drag to zoom and use the chart toolbar to pan or reset.")
    benchmark_seed = st.number_input(
        "Reproducible run seed",
        min_value=0,
        max_value=2**32 - 1,
        value=1234,
        step=1,
        key="benchmark_seed",
        help="The same seed and settings reproduce the same simulated bit streams.",
    )
    csv_path = os.path.join(HERE, "results", "ber_vs_snr.csv")
    curves = None
    if st.button("Run quick benchmark now (seconds)"):
        started = time.perf_counter()
        with st.spinner("Simulating..."):
            curves = BM.run_benchmark(
                BM.POINTS_QUICK,
                n_info=2400,
                min_errors=40,
                max_trials=3,
                min_trials=3,
                seed=int(benchmark_seed),
            )
        metadata = BM.benchmark_metadata(
            seed=int(benchmark_seed),
            points=BM.POINTS_QUICK,
            n_info=2400,
            min_errors=40,
            max_trials=3,
            min_trials=3,
            runtime_seconds=time.perf_counter() - started,
        )
        metadata["run_type"] = "quick"
        st.session_state.benchmark_curves = curves
        st.session_state.benchmark_metadata = metadata
    elif st.session_state.benchmark_curves is not None:
        curves = st.session_state.benchmark_curves
        metadata = st.session_state.benchmark_metadata
    elif os.path.isfile(csv_path):
        curves = {}
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                c = curves.setdefault(r["curve"], {"ebn0": [], "ber": [], "bits": [], "lost": [], "frames": []})
                c["ebn0"].append(float(r["ebn0_db"])); c["ber"].append(float(r["ber"])); c["bits"].append(int(r["bits_tested"]))
                c["lost"].append(int(r["frames_lost"])); c["frames"].append(int(r["frames"]))
        metadata_path = os.path.join(HERE, "results", "ber_vs_snr_metadata.json")
        metadata = None
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
        st.caption("Showing stored full benchmark results.")
    if curves:
        if metadata:
            st.caption(
                f"Seed {metadata.get('seed', 'unknown')} · {metadata.get('runtime_seconds') or 'runtime not recorded'} "
                f"seconds · software/settings are included in the JSON download."
            )
        else:
            st.warning("These stored curves predate run metadata; their seed and exact settings are unknown.")
        fig = go.Figure()
        theory_x = np.linspace(0, 10, 200)
        fig.add_trace(go.Scatter(x=theory_x, y=BM.theory_qpsk(theory_x), mode="lines", name="Uncoded QPSK (theory)", line={"color": palette["muted"], "dash": "dash"}))
        curve_colors = {
            "Uncoded QPSK": palette["secondary"],
            "Conv K=7 hard": "#b78b3d" if theme_name == "Light" else "#e0b75e",
            "Conv K=7 soft": "#527f9b" if theme_name == "Light" else "#8ab7d4",
            "LDPC soft": palette["signal"],
        }
        plot_floor = 1.0
        for curve_name, curve in curves.items():
            points = [
                (ebn0, ber, bits, lost)
                for ebn0, ber, bits, lost in zip(curve["ebn0"], curve["ber"], curve["bits"], curve["lost"])
                if bits > 0 and np.isfinite(ber)
            ]
            measured = [(ebn0, ber, bits, lost) for ebn0, ber, bits, lost in points if ber > 0]
            zero_error = [(ebn0, bits) for ebn0, ber, bits, _ in points if ber == 0]
            color = curve_colors.get(curve_name, palette["signal"])
            if measured:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in measured],
                        y=[p[1] for p in measured],
                        mode="lines+markers",
                        name=curve_name,
                        line={"color": color, "width": 2},
                        marker={"size": 8, "symbol": ["circle-open" if p[3] else "circle" for p in measured]},
                        customdata=[[p[2], p[3]] for p in measured],
                        hovertemplate="Eb/N0: %{x} dB<br>BER: %{y:.3g}<br>Bits tested: %{customdata[0]}<br>Frames lost: %{customdata[1]}<extra>%{fullData.name}</extra>",
                    )
                )
                plot_floor = min(plot_floor, *(p[1] for p in measured))
            if zero_error:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in zero_error],
                        y=[1 / p[1] for p in zero_error],
                        mode="markers",
                        name=f"{curve_name} (no errors observed)",
                        marker={"size": 10, "symbol": "triangle-down-open", "color": color, "line": {"width": 2}},
                        customdata=[[p[1]] for p in zero_error],
                        hovertemplate="Eb/N0: %{x} dB<br>No errors observed<br>Detection limit: %{y:.3g}<br>Bits tested: %{customdata[0]}<extra>%{fullData.name}</extra>",
                    )
                )
                plot_floor = min(plot_floor, *(1 / p[1] for p in zero_error))
        fig.update_layout(
            template=palette["plotly_template"],
            title="BER vs Eb/N0 — measured through the SPECTRA receiver",
            xaxis_title="Eb/N0 (dB)",
            yaxis_title="Bit error rate",
            yaxis_type="log",
            xaxis={"range": [0, 10], "gridcolor": palette["line"]},
            yaxis={"range": [np.log10(max(min(plot_floor / 3, 0.1), 1e-6)), 0], "gridcolor": palette["line"]},
            height=560,
            hovermode="closest",
            dragmode="zoom",
            legend={"orientation": "h", "y": -0.22},
            margin={"t": 65, "r": 30, "b": 100, "l": 65},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}, key="ber_plot")
        rows = []
        for curve_name, curve in curves.items():
            for ebn0, ber, bits, lost, frames in zip(
                curve["ebn0"], curve["ber"], curve["bits"], curve["lost"], curve["frames"]
            ):
                rows.append(
                    {
                        "Curve": curve_name,
                        "Eb/N0 (dB)": ebn0,
                        "BER": ber,
                        "Bits tested": bits,
                        "Frames lost": lost,
                        "Frames": frames,
                        "Lost-frame rate": lost / frames if frames else 0.0,
                    }
                )
        st.dataframe(rows, width="stretch", hide_index=True)
        dl_csv, dl_json = st.columns(2)
        dl_csv.download_button(
            "Download benchmark CSV",
            data=BM.curves_to_csv(curves),
            file_name="spectra_ber_results.csv",
            mime="text/csv",
            width="stretch",
        )
        if metadata:
            dl_json.download_button(
                "Download run metadata",
                data=json.dumps(metadata, indent=2),
                file_name="spectra_ber_metadata.json",
                mime="application/json",
                width="stretch",
            )
        else:
            dl_json.info("No recorded metadata is available for these stored curves.")
