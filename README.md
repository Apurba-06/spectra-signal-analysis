# SPECTRA - COMINT Signal Analysis Workbench (SIH 2026, PS 26147)

Blind analysis of `.wav` / `.iq` intercepts: every parameter is **measured from the samples**.

Pipeline: burst detection -> baud rate (cyclostationary) -> modulation classification (BPSK / QPSK / 16-QAM / 2-FSK) ->
matched filter + timing tracking -> carrier-offset + phase recovery -> soft demapping -> sync-word search
(all phase ambiguities / spectrum inversion) -> blind interleaver + FEC detection by trial decoding ->
soft Viterbi / Reed-Solomon / concatenated / LDPC decoding -> CRC-checked frame and payload.

## Run
    pip install -r requirements.txt
    python main.py                # GUI  (click Load File, then "Run Full Pipeline")
    python test_pipeline.py       # self-test: must print ALL PASS
    python demo_signals.py        # regenerate the demo captures

## Demo files (ground truth in *.truth.json)
| file | modulation | FEC | interleaver | IQ settings |
|---|---|---|---|---|
| demo.wav | QPSK 2400 baud | Conv K=7 | Block 16x16 | WAV |
| demo_bpsk_rs.wav | BPSK 1200 | RS(255,223) | none | WAV |
| demo_qpsk_concat.iq | QPSK 20 kbaud | RS + Conv | Forney depth 12 | 250000 Hz, cf32 |
| demo_fsk.wav | 2-FSK 1200 | Conv K=7 | Diagonal 8x32 | WAV |
| demo_16qam_uncoded.wav | 16-QAM 2400 | none | none | WAV |

## AI Analyst
Set your own key (never commit it):  `set GEMINI_API_KEY=your_key`  (or put `GEMINI_API_KEY=...` in a git-ignored `.env`).
Optional `GEMINI_MODEL` overrides the model name. Without a key the tab uses an offline rule-based analyst.

## Honest limits
Supported demodulators: BPSK, QPSK, 16-QAM, 2-FSK (8-PSK is detected only). FEC: conv K=7 r=1/2 (171,133),
RS(255,223) (reedsolo defaults, not the CCSDS dual-basis form), built-in LDPC n=1020. Pseudo-random interleavers
need a known seed. Sync library: CCSDS ASM and a few common markers - add yours in the hex box.

## BER vs Eb/N0 (BER benchmark tab and `python benchmark.py`)
Measured through the real receiver (RRC pulse, 120 Hz carrier offset, timing offset, AWGN -> timing/carrier recovery ->
soft demap -> decoder). Curves: uncoded QPSK (with theory overlay), Conv K=7 hard, Conv K=7 soft, LDPC soft.
Results are stored in `results/ber_vs_snr.png`, `ber_vs_snr_dark.png` (for slides), `ber_vs_snr.csv`, and
`ber_vs_snr_metadata.json`. The CSV includes tested bits, lost frames, total frames, and lost-frame rate; the JSON
records the seed, receiver settings, software versions, and run configuration. Frames with >30 % bit errors are
counted as receiver lock failures (hollow markers), not as BER. `python benchmark.py --quick` finishes in seconds;
the browser tab exposes the seed and downloads both the curve data and its metadata.

## Web showcase (Vercel)
`web/` is a static site (no backend) that displays the receiver's measured results on the five demo captures plus the BER curves.
Regenerate its data with `python build_site.py`; deploy on Vercel with **Root Directory = `web`** (framework: Other).
The live, interactive tool is the desktop app (`python main.py`).

## Browser app
Install the web requirements with `pip install -r requirements-web.txt`, then run `streamlit run streamlit_app.py`.
The browser app accepts WAV and raw IQ files, explains how to interpret receiver estimates and CRC results, and includes
light/dark appearance choices. Signal and BER charts are interactive: drag to zoom, then use the chart toolbar to pan,
reset the axes, or save a plot. For raw IQ, set the sample rate, IQ format, and byte order to match the capture; the
optional center frequency is recorded for context and does not alter baseband processing. The Windows desktop input
controls offer the same IQ sample-format and byte-order choices. The loader rejects empty, misaligned, non-finite, or
invalid-rate inputs. Browser analysis uses at most 600,000 samples (the loader accepts up to 1,500,000); the interface
reports the file size, source and analyzed sample counts, truncation, and processing time.
The application header describes the model as “Automated model for analysis of .IQ and .wav files along with signal
parameter extraction.” After analysis, the Overview tab provides a report download in PDF, Word (`.docx`), or
plain-text format with capture metadata, measured parameters, receiver and CRC status, and any recovered payload.

## Reliability checks
Run `python test_pipeline.py` for end-to-end decode, modulation, and baud-rate checks against the bundled synthetic
ground-truth captures. Its baud error summary describes only those five generated examples and is not a statistical
confidence interval for arbitrary recordings. Run `python -m unittest test_reliability.py` for malformed WAV/IQ
inputs, byte-order handling, low-SNR error handling, and seeded benchmark reproducibility.

## For judges - three ways to see the full application
1. **Live web version (no install):** `streamlit_app.py` runs the same backend in a browser - upload a .wav/.iq or pick a bundled capture,
   click *Run analysis* and use 12 tabs: overview, parameters, spectrum, waterfall, constellation, I/Q waveform, demodulation,
   de-interleaving, FEC, bit correlation, offline analysis, and BER benchmarking. The Overview tab also downloads the
   analysis report as PDF, Word, or text. It explains processing stages, measured estimates, inferred FEC/interleaver
   hypotheses, and how to read the results. Deployed from
   this repo via the included `Dockerfile` (Hugging Face Spaces / Render).
2. **Windows app:** download `SPECTRA-Windows.zip` from the *Releases* page (built by GitHub Actions), unzip, run `SPECTRA.exe`.
3. **From source:** `pip install -r requirements.txt` then `python main.py` (desktop GUI) or `streamlit run streamlit_app.py` (web).
