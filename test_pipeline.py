"""End-to-end self-test. Run: python test_pipeline.py"""
import os
import sys
import tempfile

import numpy as np
import demo_signals as DS
from signal_loader import load_signal
from analysis import analyze_signal


def main():
    ok_all = True
    baud_errors = []
    with tempfile.TemporaryDirectory() as tmp:
        truth = DS.generate_all(tmp)
        for t in truth.values():
            path = os.path.join(tmp, t["file"])
            x, fs, _ = load_signal(path, t["sample_rate"], "cf32")
            res = analyze_signal(x, fs)
            fr, rx = res["frame"], res["rx"]
            expected_mod = "FSK" if t["modulation"] == "FSK" else t["modulation"]
            modulation_ok = bool(rx and rx.mod == expected_mod)
            baud_error = abs(rx.baud - t["baud"]) / t["baud"] if rx else float("inf")
            if np.isfinite(baud_error):
                baud_errors.append(baud_error)
            ok = bool(
                fr
                and fr["crc_ok"]
                and fr["payload"].decode() == t["message"]
                and modulation_ok
                and baud_error <= 0.10
            )
            ok_all &= ok
            print(
                f"{'PASS' if ok else 'FAIL'}  {t['file']:26s} "
                f"detected={rx.mod if rx else res['error']}  "
                f"baud error={baud_error * 100:.2f}%  "
                f"fec={res['chain']['fec'][:38] if res['chain'] else '-'}  "
                f"inter={res['chain']['interleaver'] if res['chain'] else '-'}"
            )
    if baud_errors:
        print(
            "Bundled-capture baud validation: "
            f"median absolute error {np.median(baud_errors) * 100:.2f}%, "
            f"maximum {max(baud_errors) * 100:.2f}% "
            "(five synthetic ground-truth captures; not a real-world confidence interval)"
        )
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
