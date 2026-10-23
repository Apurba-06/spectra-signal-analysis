"""Build portable PDF, Word, and plain-text analysis reports."""

from __future__ import annotations

from datetime import datetime
import io
import textwrap
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zipfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_TITLE = "SPECTRA Signal Analysis Report"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def report_lines(
    *,
    source_name: str,
    file_type: str,
    sample_rate: float,
    file_size: int,
    source_samples: int,
    analyzed_samples: int,
    runtime_seconds: float,
    center_frequency_hz: float,
    params: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    """Prepare report content from the receiver output and capture metadata."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        REPORT_TITLE,
        f"Generated: {generated_at}",
        "",
        "=== INPUT CAPTURE ===",
        f"Source file: {source_name}",
        f"File type: {file_type.upper()}",
        f"File size: {file_size:,} bytes",
        f"Sample rate: {sample_rate:,.2f} Hz",
        f"Source samples: {source_samples:,}",
        f"Samples analyzed: {analyzed_samples:,}",
        f"Processing time: {runtime_seconds:.2f} seconds",
    ]
    if center_frequency_hz:
        lines.append(
            f"Capture center frequency (context only): {center_frequency_hz:,.2f} Hz"
        )
    if analyzed_samples < source_samples:
        lines.append(
            "Note: The capture was truncated to the configured analysis sample limit."
        )

    rx = result.get("rx")
    sync = result.get("sync")
    chain = result.get("chain")
    frame = result.get("frame")
    lines.extend(["", "=== RECEIVER RESULT ==="])
    if rx is None:
        lines.append(f"Receiver status: Could not lock ({result.get('error') or 'unknown error'})")
    else:
        lines.extend(
            [
                f"Receiver status: {'LOCKED' if rx.rec.get('locked') else 'WEAK'}",
                f"Detected modulation: {rx.mod}",
                f"Estimated symbol rate: {rx.baud:,.2f} baud",
                f"Estimated SNR: {rx.rec.get('snr_db', float('nan')):.2f} dB",
            ]
        )
        if rx.cls:
            lines.append(
                f"Modulation classifier score: {rx.cls['confidence'] * 100:.1f}% "
                "(heuristic; not a calibrated probability)"
            )
    if sync and sync.get("found"):
        hit = sync["hit"]
        lines.append(
            f"Frame synchronization: {hit['pattern']} at bit {hit['position']} "
            f"({hit['mismatches']} mismatches)"
        )
    else:
        lines.append("Frame synchronization: No known sync word found")
    if chain:
        lines.append(f"FEC hypothesis: {chain.get('fec', 'Not available')}")
        lines.append(f"Interleaver hypothesis: {chain.get('interleaver', 'Not available')}")
    if frame:
        lines.append(f"Frame integrity: CRC-16 valid ({frame['length']} payload bytes)")
        payload = frame["payload"].decode("utf-8", "replace")
        lines.extend(["", "=== DECODED PAYLOAD ===", payload])
    else:
        lines.append("Frame integrity: No CRC-16-valid payload recovered")
        if result.get("error"):
            lines.append(f"Receiver message: {result['error']}")

    lines.extend(["", "=== MEASURED PARAMETERS ==="])
    lines.extend(f"{key}: {value}" for key, value in params.items())
    lines.extend(
        [
            "",
            "Interpretation note: FEC and interleaver labels are receiver hypotheses. "
            "A classifier score is heuristic. A valid CRC indicates the decoded frame "
            "passed its integrity check.",
        ]
    )
    return lines


def build_text_report(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_word_report(lines: list[str]) -> bytes:
    """Create a minimal valid DOCX package using only the Python standard library."""
    ET.register_namespace("w", WORD_NS)
    ET.register_namespace("r", REL_NS)
    document = ET.Element(f"{{{WORD_NS}}}document")
    body = ET.SubElement(document, f"{{{WORD_NS}}}body")

    for index, line in enumerate(lines):
        paragraph = ET.SubElement(body, f"{{{WORD_NS}}}p")
        run = ET.SubElement(paragraph, f"{{{WORD_NS}}}r")
        if index == 0 or line.startswith("=== "):
            properties = ET.SubElement(run, f"{{{WORD_NS}}}rPr")
            ET.SubElement(properties, f"{{{WORD_NS}}}b")
            if index == 0:
                size = ET.SubElement(properties, f"{{{WORD_NS}}}sz")
                size.set(f"{{{WORD_NS}}}val", "32")
        text = ET.SubElement(run, f"{{{WORD_NS}}}t")
        text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        text.text = line or " "

    section = ET.SubElement(body, f"{{{WORD_NS}}}sectPr")
    page_size = ET.SubElement(section, f"{{{WORD_NS}}}pgSz")
    page_size.set(f"{{{WORD_NS}}}w", "12240")
    page_size.set(f"{{{WORD_NS}}}h", "15840")

    content_types = ET.Element(f"{{{CONTENT_NS}}}Types")
    default = ET.SubElement(content_types, f"{{{CONTENT_NS}}}Default")
    default.set("Extension", "rels")
    default.set("ContentType", "application/vnd.openxmlformats-package.relationships+xml")
    default = ET.SubElement(content_types, f"{{{CONTENT_NS}}}Default")
    default.set("Extension", "xml")
    default.set("ContentType", "application/xml")
    override = ET.SubElement(content_types, f"{{{CONTENT_NS}}}Override")
    override.set("PartName", "/word/document.xml")
    override.set(
        "ContentType",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
    )

    relationships = ET.Element(f"{{{PKG_REL_NS}}}Relationships")
    relationship = ET.SubElement(relationships, f"{{{PKG_REL_NS}}}Relationship")
    relationship.set("Id", "rId1")
    relationship.set(
        "Type",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
    )
    relationship.set("Target", "word/document.xml")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        archive.writestr("_rels/.rels", ET.tostring(relationships, encoding="utf-8", xml_declaration=True))
        archive.writestr("word/document.xml", ET.tostring(document, encoding="utf-8", xml_declaration=True))
    return output.getvalue()


def build_pdf_report(lines: list[str]) -> bytes:
    """Render a readable multi-page PDF report using the installed Matplotlib backend."""
    output = io.BytesIO()
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=96, replace_whitespace=False) or [""])

    with PdfPages(output, metadata={"Title": REPORT_TITLE, "Author": "SPECTRA"}) as pdf:
        for start in range(0, max(1, len(wrapped)), 48):
            page_lines = wrapped[start : start + 48]
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(
                0.08,
                0.96,
                "\n".join(page_lines),
                va="top",
                ha="left",
                family="DejaVu Sans Mono",
                fontsize=8.5,
                linespacing=1.45,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return output.getvalue()


def build_report(
    report_format: str,
    *,
    source_name: str,
    file_type: str,
    sample_rate: float,
    file_size: int,
    source_samples: int,
    analyzed_samples: int,
    runtime_seconds: float,
    center_frequency_hz: float,
    params: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bytes:
    """Return an analysis report as PDF, DOCX, or plain UTF-8 text bytes."""
    lines = report_lines(
        source_name=source_name,
        file_type=file_type,
        sample_rate=sample_rate,
        file_size=file_size,
        source_samples=source_samples,
        analyzed_samples=analyzed_samples,
        runtime_seconds=runtime_seconds,
        center_frequency_hz=center_frequency_hz,
        params=params,
        result=result,
    )
    normalized_format = report_format.lower().lstrip(".")
    if normalized_format == "pdf":
        return build_pdf_report(lines)
    if normalized_format in ("docx", "word"):
        return build_word_report(lines)
    if normalized_format in ("txt", "text"):
        return build_text_report(lines)
    raise ValueError("Report format must be 'pdf', 'docx'/'word', or 'txt'/'text'.")
