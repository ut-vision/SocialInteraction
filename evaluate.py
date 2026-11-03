"""Compare two prediction files (e.g. baseline vs. ours) produced by run_zeroshot.py."""

import argparse
import json
from collections import defaultdict


def normalize_answer(text):  # same as social_attention.data.normalize_answer (kept inline: no torch needed here)
    return " ".join(text.strip().lower().replace("speaker", "speaker ").split())


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def accuracy(preds):
    c, n = defaultdict(int), defaultdict(int)
    for r in preds:
        t = r.get("task") or "all"
        n[t] += 1
        c[t] += int(normalize_answer(r["predict"]) == normalize_answer(r["label"]))
    return {t: (c[t], n[t]) for t in n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", nargs="+", help="one or more jsonl files from run_zeroshot.py")
    args = p.parse_args()
    for path in args.predictions:
        preds = load(path)
        print(path)
        for t, (c, n) in sorted(accuracy(preds).items()):
            print(f"  {t}: {c / n:.2%} ({c}/{n})")
    if len(args.predictions) == 2:
        a, b = load(args.predictions[0]), load(args.predictions[1])
        flips = [
            ra["id"]
            for ra, rb in zip(a, b)
            if normalize_answer(ra["predict"]) != normalize_answer(ra["label"])
            and normalize_answer(rb["predict"]) == normalize_answer(rb["label"])
        ]
        print(f"wrong in {args.predictions[0]} but correct in {args.predictions[1]}: {len(flips)} samples")
        print(" ", flips[:50])


if __name__ == "__main__":
    main()
