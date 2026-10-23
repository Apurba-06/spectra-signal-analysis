"""Focused tests for raw-capture validation and reproducible benchmark output."""

import csv
import io
import json
import os
import tempfile
import unittest
import zipfile

import numpy as np
import soundfile as sf

import benchmark as BM
from analysis import analyze_signal, flatten_params
from demo_signals import make_signal
from report_export import build_report
from signal_loader import load_iq, load_signal, load_wav


class CaptureValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def path(self, name):
        return os.path.join(self.temp_dir.name, name)

    def test_cf32_respects_byte_order(self):
        expected = np.array([0.25 - 0.5j, -0.75 + 0.125j], dtype=np.complex64)
        values = np.empty(expected.size * 2, dtype=">f4")
        values[0::2], values[1::2] = expected.real, expected.imag
        path = self.path("capture.iq")
        values.tofile(path)

        actual, sample_rate = load_iq(path, 48000, "cf32", "big")

        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertEqual(sample_rate, 48000.0)

    def test_ci16_respects_byte_order_and_normalizes(self):
        path = self.path("capture.bin")
        np.array([16384, -8192], dtype=">i2").tofile(path)

        actual, _ = load_iq(path, 100000, "ci16", "big")

        np.testing.assert_allclose(actual, [0.5 - 0.25j])

    def test_eight_bit_formats_decode_as_expected(self):
        ci8_path = self.path("signed.iq")
        np.array([64, -64], dtype=np.int8).tofile(ci8_path)
        ci8, _ = load_iq(ci8_path, 48000, "ci8")
        np.testing.assert_allclose(ci8, [0.5 - 0.5j])

        cu8_path = self.path("unsigned.iq")
        np.array([255, 0], dtype=np.uint8).tofile(cu8_path)
        cu8, _ = load_iq(cu8_path, 48000, "cu8")
        np.testing.assert_allclose(cu8, [1.0 - 1.0j])

    def test_empty_and_incomplete_iq_files_are_rejected(self):
        empty_path = self.path("empty.iq")
        open(empty_path, "wb").close()
        with self.assertRaisesRegex(ValueError, "empty"):
            load_signal(empty_path, 48000, "cf32")

        partial_path = self.path("partial.iq")
        with open(partial_path, "wb") as fh:
            fh.write(b"\x00" * 7)
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_iq(partial_path, 48000, "cf32")

    def test_invalid_sample_rate_format_and_byte_order_are_rejected(self):
        path = self.path("capture.iq")
        np.array([0.0, 1.0], dtype=np.float32).tofile(path)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "Sample rate"):
                load_iq(path, rate)
        with self.assertRaisesRegex(ValueError, "Unknown IQ format"):
            load_iq(path, 48000, "unknown")
        with self.assertRaisesRegex(ValueError, "Byte order"):
            load_iq(path, 48000, "cf32", "native")

    def test_nonfinite_cf32_samples_are_rejected(self):
        path = self.path("nonfinite.iq")
        np.array([np.nan, 0.0], dtype=np.float32).tofile(path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_iq(path, 48000, "cf32")

    def test_empty_and_nonfinite_wav_files_are_rejected(self):
        empty_path = self.path("empty.wav")
        sf.write(empty_path, np.array([], dtype=np.float32), 48000)
        with self.assertRaisesRegex(ValueError, "no audio samples"):
            load_wav(empty_path)

        nonfinite_path = self.path("nonfinite.wav")
        sf.write(nonfinite_path, np.array([0.0, np.nan, 0.0], dtype=np.float32), 48000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_wav(nonfinite_path)

    def test_low_snr_capture_returns_analysis_result_without_raising(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 256).astype(np.uint8)
        samples = make_signal(bits, "QPSK", 48000, 2400, snr_db=-8, seed=43, lead=1200, trail=800)

        result = analyze_signal(samples, 48000)

        self.assertIn("rx", result)
        self.assertIn("error", result)
        self.assertIsNone(result["frame"])


class BenchmarkReproducibilityTests(unittest.TestCase):
    def test_seed_repeats_curves_and_exports_run_statistics(self):
        settings = {
            "points": [3],
            "n_info": 128,
            "min_errors": 1,
            "max_trials": 1,
            "min_trials": 1,
            "seed": 2026,
        }
        first = BM.run_benchmark(**settings)
        second = BM.run_benchmark(**settings)

        self.assertEqual(BM.curves_to_csv(first), BM.curves_to_csv(second))
        rows = list(csv.DictReader(BM.curves_to_csv(first).splitlines()))
        self.assertTrue(rows)
        self.assertTrue(all("lost_frame_rate" in row for row in rows))
        self.assertTrue(all(int(row["bits_tested"]) >= 0 for row in rows))

        metadata = BM.benchmark_metadata(
            seed=settings["seed"],
            points=settings["points"],
            n_info=settings["n_info"],
            min_errors=settings["min_errors"],
            max_trials=settings["max_trials"],
            min_trials=settings["min_trials"],
            runtime_seconds=1.25,
        )
        self.assertEqual(metadata["seed"], 2026)
        self.assertEqual(metadata["receiver"]["alignment"], "known-preamble best-match alignment")
        self.assertEqual(metadata["runtime_seconds"], 1.25)
        self.assertIn("numpy", metadata["software"])
        with tempfile.TemporaryDirectory() as output_dir:
            image_path = BM.save_results(first, output_dir, metadata)
            metadata_path = os.path.join(output_dir, "ber_vs_snr_metadata.json")
            csv_path = os.path.join(output_dir, "ber_vs_snr.csv")
            self.assertTrue(os.path.isfile(image_path))
            with open(metadata_path, encoding="utf-8") as fh:
                saved_metadata = json.load(fh)
            self.assertEqual(saved_metadata["seed"], 2026)
            with open(csv_path, newline="") as fh:
                saved_rows = list(csv.DictReader(fh))
            self.assertTrue(saved_rows)
            self.assertIn("lost_frame_rate", saved_rows[0])


class ReportExportTests(unittest.TestCase):
    def setUp(self):
        self.report_args = {
            "source_name": "demo.wav",
            "file_type": "wav",
            "sample_rate": 48000.0,
            "file_size": 4096,
            "source_samples": 1024,
            "analyzed_samples": 1024,
            "runtime_seconds": 0.75,
            "center_frequency_hz": 145000000.0,
            "params": {
                "Modulation (estimated)": "QPSK",
                "Frame Sync": "CCSDS ASM at bit 0",
            },
            "result": {
                "rx": type(
                    "Receiver",
                    (),
                    {
                        "mod": "QPSK",
                        "baud": 2400.0,
                        "rec": {"locked": True, "snr_db": 14.0},
                        "cls": {"confidence": 0.95},
                    },
                )(),
                "sync": {"found": True, "hit": {"pattern": "ASM", "position": 0, "mismatches": 0}},
                "chain": {"fec": "Conv K=7", "interleaver": "Block"},
                "frame": {
                    "length": 16,
                    "payload": b"Test payload",
                    "crc_ok": True,
                },
                "error": None,
            },
        }

    def test_text_report_includes_result_and_capture_metadata(self):
        report = build_report("txt", **self.report_args).decode("utf-8")

        self.assertIn("Source file: demo.wav", report)
        self.assertIn("Capture center frequency (context only): 145,000,000.00 Hz", report)
        self.assertIn("Modulation classifier score: 95.0% (heuristic; not a calibrated probability)", report)
        self.assertIn("Frame integrity: CRC-16 valid (16 payload bytes)", report)
        self.assertIn("Test payload", report)

    def test_text_report_describes_receiver_failure(self):
        args = dict(self.report_args)
        args["result"] = {
            "rx": None,
            "sync": None,
            "chain": None,
            "frame": None,
            "error": "could not estimate the symbol rate",
        }
        report = build_report("txt", **args).decode("utf-8")

        self.assertIn("Receiver status: Could not lock (could not estimate the symbol rate)", report)
        self.assertIn("Frame integrity: No CRC-16-valid payload recovered", report)

    def test_word_report_is_a_valid_docx_package(self):
        report = build_report("docx", **self.report_args)

        with zipfile.ZipFile(io.BytesIO(report)) as archive:
            self.assertIn("[Content_Types].xml", archive.namelist())
            self.assertIn("word/document.xml", archive.namelist())
            document = archive.read("word/document.xml")
        self.assertIn(b"SPECTRA Signal Analysis Report", document)
        self.assertIn(b"Test payload", document)

    def test_pdf_report_has_pdf_signature_and_text(self):
        report = build_report("pdf", **self.report_args)

        self.assertTrue(report.startswith(b"%PDF-"))
        self.assertGreater(len(report), 1000)

    def test_all_formats_serialize_a_real_receiver_result(self):
        root = os.path.dirname(os.path.abspath(__file__))
        capture = os.path.join(root, "demo.wav")
        samples, sample_rate, file_type = load_signal(capture, 48000, "cf32")
        result = analyze_signal(samples, sample_rate)
        args = {
            "source_name": os.path.basename(capture),
            "file_type": file_type,
            "sample_rate": sample_rate,
            "file_size": os.path.getsize(capture),
            "source_samples": len(samples),
            "analyzed_samples": len(samples),
            "runtime_seconds": 1.0,
            "center_frequency_hz": 0.0,
            "params": flatten_params(result, sample_rate, len(samples)),
            "result": result,
        }

        pdf = build_report("pdf", **args)
        word = build_report("docx", **args)
        text = build_report("txt", **args).decode("utf-8")

        self.assertTrue(pdf.startswith(b"%PDF-"))
        with zipfile.ZipFile(io.BytesIO(word)) as archive:
            self.assertIsNone(archive.testzip())
        self.assertIn("Frame integrity: CRC-16 valid", text)
        self.assertIn(result["frame"]["payload"].decode("ascii"), text)

    def test_unknown_report_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Report format"):
            build_report("csv", **self.report_args)


if __name__ == "__main__":
    unittest.main()
