# SPECTRA

Automated analysis of `.wav` and raw `.iq` captures, with signal parameter extraction and receiver diagnostics.

| Resource | Link |
| --- | --- |
| Live application | [spectra-comint.onrender.com](https://spectra-comint.onrender.com/) |
| Application health check | [Render health endpoint](https://spectra-comint.onrender.com/_stcore/health) |
| Public source repository | [github.com/Apurba-06/spectra-signal-analysis](https://github.com/Apurba-06/spectra-signal-analysis) |

## Project overview

SPECTRA is a digital-signal analysis workbench. It accepts WAV recordings and raw IQ captures, estimates signal characteristics from their samples, and attempts to recover symbols, frames, and payloads. The browser interface presents each analysis operation in its own tab and explains how to interpret measured values and receiver hypotheses.

The receiver workflow includes:

1. Load a bundled example or upload a WAV or raw IQ capture.
2. Estimate modulation, baud rate, bandwidth, carrier offset, and signal quality.
3. Recover symbols and bits, then search for supported frame synchronization markers.
4. Test supported interleavers and forward-error-correction decoders.
5. Check recovered frames with CRC and inspect the BER benchmark.

CRC-16 OK indicates that a recovered frame passed its integrity check. Signal parameters are estimates from the capture. FEC and interleaver identifications are receiver hypotheses, not proof of the transmitter configuration.

## Try the live application

Open the [live SPECTRA application](https://spectra-comint.onrender.com/). Choose a bundled capture or select **Upload my own file** in the sidebar, then click **Run analysis**. The app accepts `.wav`, `.iq`, `.bin`, `.raw`, and `.dat` files up to 100 MB. Raw IQ uploads require the sample rate, IQ format, and byte order to match the capture.

The interface includes 12 analysis tabs: Overview, Parameters, Spectrum, Waterfall, Constellation, I/Q waveform, Demodulation, De-interleaving, FEC, Bit correlation, AI analyst, and BER benchmark. The charts support zooming, panning, and resetting the view. After an analysis, download the results as PDF, Word (`.docx`), or plain text from the Overview tab.

Render's free service can take a little time to wake after a period without traffic.

## Run locally

Python 3.11 or newer is recommended. From the project directory, install the browser-app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements-web.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed in the terminal. To run the Windows desktop interface instead, install the project dependencies and start the desktop app:

```powershell
python -m pip install -r requirements.txt
python main.py
```

## Example captures

The bundled captures have corresponding `.truth.json` files describing their generated ground truth.

| Capture | Modulation | Coding | Interleaving | Input |
| --- | --- | --- | --- | --- |
| `demo.wav` | QPSK, 2400 baud | Convolutional K=7 | Block 16 x 16 | WAV |
| `demo_bpsk_rs.wav` | BPSK, 1200 baud | RS(255,223) | None | WAV |
| `demo_qpsk_concat.iq` | QPSK, 20 kbaud | RS + convolutional | Forney, depth 12 | Raw IQ, 250 kHz, cf32 |
| `demo_fsk.wav` | 2-FSK, 1200 baud | Convolutional K=7 | Diagonal 8 x 32 | WAV |
| `demo_16qam_uncoded.wav` | 16-QAM, 2400 baud | None | None | WAV |

Regenerate the synthetic demo captures with:

```powershell
python demo_signals.py
```

## AI analyst

The AI analyst tab works with its offline, rule-based analysis when no API key is configured. To enable Gemini, set `GEMINI_API_KEY` in your local environment or in a git-ignored `.env` file. Never commit API keys or other credentials. `GEMINI_MODEL` can optionally select a model.

## Tests and reliability

Run the end-to-end checks against the bundled synthetic captures:

```powershell
python test_pipeline.py
```

Run the reliability and report-export tests:

```powershell
python -m unittest test_reliability.py
```

The pipeline test checks five generated ground-truth captures. Its baud-error summary describes only those examples; it is not a statistical confidence interval for arbitrary real-world signals.

## BER benchmark

The BER benchmark evaluates the receiver using generated signals with pulse shaping, carrier and timing offsets, and additive noise. Run the full benchmark or its quicker smoke run with:

```powershell
python benchmark.py
python benchmark.py --quick
```

The benchmark writes its curves and CSV results under `results/`. The browser tab also supports seeded runs and downloads benchmark data and run metadata.

## Supported signal processing

The receiver supports BPSK, QPSK, 16-QAM, and 2-FSK demodulation; 8-PSK is detection-only. Supported coding includes convolutional K=7 rate 1/2, RS(255,223), concatenated decoding, and built-in LDPC decoding. Pseudo-random interleavers require a known seed. The frame-sync library covers CCSDS ASM and selected common markers; additional sync words can be supplied in the interface.

## Deployment

The repository includes a `Dockerfile` and Render Blueprint in `render.yaml`. Render builds and runs the Streamlit web application from this repository, using `/_stcore/health` as its health-check endpoint. See the [live application](https://spectra-comint.onrender.com/) to use the deployed model.
